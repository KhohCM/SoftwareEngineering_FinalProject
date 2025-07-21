from fpdf import FPDF
from docx import Document
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, jsonify, send_file, redirect
import torch
import os
import logging
import random
import re
import markdown2
import requests
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from abc import ABC, abstractmethod
from typing import Type
from dotenv import load_dotenv
load_dotenv()

# === Groq API Setup ===
API_KEY = os.getenv("GROQ_API_KEY")
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-8b-8192"

# === Flask App Setup ===
app = Flask(__name__, static_folder='static', template_folder='templates')
app.template_filter('markdown')(markdown2.markdown)

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === GPT2 Local Model Setup ===
model_path = "gpt2-ramsay-finetuned2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, model_exists = None, None, False

# File upload configuration
UPLOAD_FOLDER = "uploads"
EXPORT_FOLDER = "exports"
ALLOWED_EXTENSIONS = {"pdf", "docx"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)

# === File Extractor Classes ===
class FileExtractor(ABC):
    """Common interface for all file parsers"""
    def __init__(self, path: str):
        self._path = path

    @abstractmethod
    def extract_text(self) -> str:
        """Extract file text content"""
        raise NotImplementedError

class PdfExtractor(FileExtractor):
    def extract_text(self) -> str:
        from PyPDF2 import PdfReader
        reader = PdfReader(self._path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)

class DocxExtractor(FileExtractor):
    def extract_text(self) -> str:
        from docx import Document
        doc = Document(self._path)
        return "\n".join(p.text for p in doc.paragraphs)

class ExtractorFactory:
    _registry: dict[str, Type[FileExtractor]] = {
        ".pdf": PdfExtractor,
        ".docx": DocxExtractor,
    }

    @classmethod
    def get_extractor(cls, path: str) -> FileExtractor:
        ext = os.path.splitext(path)[1].lower()
        if ext not in cls._registry:
            raise ValueError(f"Unsupported file type: {ext}")
        return cls._registry[ext](path)

def save_to_pdf(filename, ingredients=None, steps=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    if ingredients:
        pdf.set_font("Arial", "B", size=14)
        pdf.cell(200, 10, "Ingredients", ln=True)
        pdf.set_font("Arial", size=12)
        for line in ingredients:
            pdf.cell(200, 10, f"- {line}", ln=True)

    if steps:
        pdf.set_font("Arial", "B", size=14)
        pdf.cell(200, 10, "Steps", ln=True)
        pdf.set_font("Arial", size=12)
        for idx, step in enumerate(steps, 1):
            pdf.cell(200, 10, f"{idx}. {step}", ln=True)

    output_path = os.path.join(EXPORT_FOLDER, secure_filename(filename))
    pdf.output(output_path)
    return output_path

def save_to_word(filename, ingredients=None, steps=None):
    doc = Document()

    if ingredients:
        doc.add_heading("Ingredients", level=1)
        for item in ingredients:
            doc.add_paragraph(item, style='List Bullet')

    if steps:
        doc.add_heading("Steps", level=1)
        for idx, step in enumerate(steps, 1):
            doc.add_paragraph(f"{idx}. {step}")

    output_path = os.path.join(EXPORT_FOLDER, secure_filename(filename))
    doc.save(output_path)
    return output_path

def parse_bot_response_sections(bot_response):
    # Parses text for sections marked **Ingredients:** and **Steps:**
    ingredients, steps = [], []
    if "**Ingredients:**" in bot_response:
        parts = bot_response.split("**Ingredients:**")
        if len(parts) > 1:
            rest = parts[1].split("**Steps:**")
            ingredients = [line.strip("- ").strip() for line in rest[0].strip().split("\\n") if line.strip()]
            if len(rest) > 1:
                steps = [line.strip("0123456789. ").strip() for line in rest[1].strip().split("\\n") if line.strip()]
    return ingredients, steps

#=====
if os.path.exists(model_path):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        model.eval()
        model_exists = True
    except Exception as e:
        logger.error(f"Failed to load custom GPT-2 model: {e}")
else:
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        model.eval()
        model_exists = True
    except Exception as e:
        logger.error(f"Failed to load base GPT-2 model: {e}")

# === Global Conversation History ===
conversation_history = [{
    "sender": "Gordon Ramsay",
    "text": "Where is the bloody duck?? Tell me what ingredients you have or what recipe you want to learn, you donut!"
}]

# === Utility Functions ===
def format_user_input(user_input):
    formatted_input = re.sub(r'\\s+', ' ', user_input).strip()
    if re.search(r'how (to|do|can|should) .+[^?]$', formatted_input.lower()):
        formatted_input += "?"
    return formatted_input

def build_conversation_context(user_input, include_history=True, max_history=3):
    formatted_input = format_user_input(user_input)
    context_parts = []

    for msg in conversation_history:
        if msg["sender"] == "Memory":
            context_parts.append(f"(Reference context from uploaded material: {msg['text']})")

    if include_history:
        history_pairs = []
        relevant_history = [
            msg for msg in conversation_history if msg["sender"] in ["User", "Gordon Ramsay"]
        ][-max_history * 2:]

        for i in range(0, len(relevant_history), 2):
            if i + 1 < len(relevant_history):
                history_pairs.append(
                    f"User: {relevant_history[i]['text']}\n### Bot: {relevant_history[i+1]['text']}"
                )

        context_parts.extend(history_pairs)

    # current user prompt
    context_parts.append(f"User: {formatted_input}\n### Bot:")

    return "\n\n".join(context_parts)

def process_response(response_text):
    if "### Bot:" in response_text:
        bot_response = response_text.split("### Bot:")[-1].strip()
    else:
        lines = response_text.split('\\n')
        for i, line in enumerate(lines):
            if line.startswith("User:") or "User:" in line:
                bot_response = "\\n".join(lines[i+1:]).strip()
                break
        else:
            bot_response = response_text.strip()
    bot_response = re.sub(r'###.*$', '', bot_response).strip()
    bot_response = re.sub(r'Bot:.*$', '', bot_response).strip()
    gordon_phrases = ["you donut", "bloody hell", "for god's sake", "idiot sandwich", "you muppet"]
    if not any(phrase in bot_response.lower() for phrase in gordon_phrases) and len(bot_response) > 30:
        endings = ["you bloody donut!", "is that clear, you muppet?", "now get cooking!", "it's not rocket science!", "for God's sake!"]
        if not any(bot_response.endswith(e) for e in ["!", "?"]):
            bot_response += " " + random.choice(endings)
    return bot_response

def generate_Gordon_Ramsay_style_prefix(user_input):
    if not model_exists:
        return "You call that a question, you donkey?"
    try:
        prompt = f"User: {user_input}\\n### Bot:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=80,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return process_response(response_text)
    except Exception as e:
        logger.error(f"Prefix generation failed: {e}")
        return "You better not mess this up, you muppet!"

def classify_user_intent(user_input):
    text = user_input.lower()
    if any(word in text for word in ["suggestion", "more", "improve", "better"]):
        return "tips"
    elif any(word in text for word in ["replace", "instead of", "can i use", "substitute"]):
        return "substitution"
    elif "roast" in text:
        return "roast"
    elif any(word in text for word in ["how", "cook", "make", "recipe", "i want to eat", "fried rice", "what can i eat"]):
        return "new_recipe"
    else:
        return "general"

def get_system_prompt(intent):
    base_prompt = (
        "You are Gordon Ramsay, the world-famous angry chef. "
        "You respond to cooking questions with brutal honesty, insults, sarcasm, and some helpful advice. "
        "Include phrases like 'you muppet', 'bloody hell', or 'idiot sandwich'. "
        "Despite your aggressive style, always provide detailed step-by-step instructions.\n\n"
        "**Ingredients:**\n- List each ingredient clearly\n\n"
        "**Steps:**\n1. Write each step clearly\n"
    )
    return base_prompt

def generate_response(user_input):
    style_prefix = generate_Gordon_Ramsay_style_prefix(user_input)
    intent = classify_user_intent(user_input)
    system_prompt = get_system_prompt(intent)
    full_user_input = f"{style_prefix}\\n\\n{user_input}"

    history_messages = [{"role": "system", "content": system_prompt}]

    # Add content into memory_store (memory)
    if memory_store.get("content"):
        history_messages.append({
            "role": "system",
            "content": f"( {memory_store['content'][:1500]})"
        })

    # Add history
    for msg in conversation_history:
        if msg["sender"] in ["User", "Gordon Ramsay"]:
            history_messages.append({
                "role": "user" if msg["sender"] == "User" else "assistant",
                "content": msg["text"]
            })

    history_messages.append({"role": "user", "content": full_user_input})

    payload = {
        "model": MODEL_NAME,
        "messages": history_messages,
        "temperature": 0.8,
        "top_p": 0.92,
        "max_tokens": 700
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            result = response.json()
            return process_response(result["choices"][0]["message"]["content"].strip())
        else:
            return "Groq API error, try again later!"
    except Exception as e:
        logger.error(f"Groq request failed: {e}")
        return "Something went wrong in the kitchen!"

@app.route("/", methods=["GET", "POST"])
def chat():
    global conversation_history

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        file = request.files.get("file")

        file_text = ""
        file_name = ""

        # Upload file
        if file and os.path.splitext(file.filename)[1].lower() in ExtractorFactory._registry:
            file_name = secure_filename(file.filename)
            filepath = os.path.join("uploads", file_name)
            file.save(filepath)

            # Read File
            extractor = ExtractorFactory.get_extractor(filepath)
            file_text = extractor.extract_text()

            # Save into memory
            memory_store["content"] = file_text
            memory_store["filename"] = file_name

        # User Prompt
        combined_input = ""

        if user_input:
            combined_input += user_input
        if file_text:
            combined_input += f"\n\n[Reference from uploaded file]\n{file_text}"
        elif memory_store.get("content"):
            combined_input += f"\n\n[Reference from uploaded file]\n{memory_store['content']}"

        if combined_input.strip():
            user_display_text = ""
            if file_name:
                user_display_text += f"File Uploaded: **{file_name}**\n\n"
            if user_input:
                user_display_text += user_input

        conversation_history.append({"sender": "User", "text": user_display_text.strip()})
        bot_response = generate_response(combined_input.strip())
        conversation_history.append({"sender": "Gordon Ramsay", "text": bot_response})

    return render_template("chat.html", conversation_history=conversation_history, model_exists=model_exists)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    global conversation_history
    user_input = request.json.get("user_input", "").strip() if request.is_json else ""
    if not user_input:
        return jsonify({"status": "error", "response": "No input provided!"})
    bot_response = generate_response(user_input)
    conversation_history.append({"sender": "User", "text": user_input})
    conversation_history.append({"sender": "Gordon Ramsay", "text": bot_response})
    return jsonify({"status": "success", "response": bot_response})

memory_store = {}

# upload file
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    if os.path.splitext(file.filename)[1].lower() in ExtractorFactory._registry:
        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        file.save(filepath)

        # Read pdf @ word file
        extractor = ExtractorFactory.get_extractor(filepath)
        text = extractor.extract_text()

        # save into memory
        memory_store["content"] = text
        memory_store["filename"] = filename

        return jsonify({"filename": filename})
    else:
        return "Invalid file type", 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)