
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
from docx import Document
from PyPDF2 import PdfReader

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

conversation_history = []
memory_store = {}

# Load GPT-2 Model
model_path = "gpt2"  # or your fine-tuned model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
model.eval()

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <title>Chatbot with GPT-2</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2em; }
        .chat-box { border: 1px solid #ccc; padding: 1em; margin-top: 1em; }
        .user { color: blue; }
        .bot { color: green; }
    </style>
</head>
<body>
    <h2>Chatbot with GPT-2</h2>
    <form method="post" enctype="multipart/form-data">
        <input type="text" name="user_input" required style="width: 300px;">
        <input type="file" name="file">
        <button type="submit">Send</button>
    </form>
    <div class="chat-box">
        {% for message in conversation_history %}
            <p><strong class="{{ message.sender|lower }}">{{ message.sender }}:</strong> {{ message.text }}</p>
        {% endfor %}
    </div>
</body>
</html>
"""

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""

def generate_gpt2_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=100,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        file = request.files.get("file")
        file_text = ""

        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            file_text = extract_text_from_file(filepath)
            memory_store["content"] = file_text
            conversation_history.append({"sender": "Memory", "text": f"File uploaded: {filename}"})

        combined_input = user_input
        if memory_store.get("content"):
            combined_input += f"\n[Context from file]: {memory_store['content'][:300]}..."

        bot_response = generate_gpt2_response(combined_input)
        conversation_history.append({"sender": "User", "text": user_input})
        conversation_history.append({"sender": "Bot", "text": bot_response})

    return render_template_string(HTML_TEMPLATE, conversation_history=conversation_history)

if __name__ == "__main__":
    app.run(debug=True)
