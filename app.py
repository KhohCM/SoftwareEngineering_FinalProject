# Link with Groq API v2

from flask import Flask, request, render_template, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import torch
import os
import logging
import random
import re

import markdown2
import requests
import json
from dotenv import load_dotenv
load_dotenv()

# set up Groq API
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    # Update the error message
    raise ValueError("GROQ_API_KEY not found. Check the '.env' file or environment variables.")

# Groq's API endpoint
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-8b-8192" 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

@app.template_filter('markdown')
def markdown_filter(text):
    return markdown2.markdown(text)

# Global conversation history
conversation_history = []

# Define the model path - can be a local path or Hugging Face model ID
model_path = "gpt2-ramsay-finetuned2"  # Change this to your fine-tuned model path
model_exists = False
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Few minimal fallback responses for complete failures
fallback_responses = [
    "Listen to me! I can't understand what the bloody hell you're asking. Try again with clear ingredients or a specific cooking question!",
    "For God's sake! Ask me something that makes sense! I'm a chef, not a mind reader!",
    "Are you serious? That question is NONSENSE! Ask about cooking something specific!",
]

# Check and load the model
logger.info(f"Checking for model directory at: {model_path}")
if os.path.exists(model_path):
    logger.info(f"Directory exists. Listing contents: {os.listdir(model_path)}")
    try:
        logger.info(f"Using device: {device}")

        # Load tokenizer and model
        # tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        # model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        # model.eval()
        model_exists = True
        logger.info("Successfully loaded model and tokenizer")
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer from {model_path}: {e}")
        logger.info("Falling back to base GPT-2 model.")
        try:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
            model.eval()
            model_exists = True
            logger.info("Successfully loaded base GPT-2 model as fallback.")
        except Exception as fallback_e:
            logger.error(f"Failed to load base GPT-2 model: {fallback_e}")
            model_exists = False
else:
    logger.warning(f"Model directory {model_path} not found. Falling back to base GPT-2 model.")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        model.eval()
        model_exists = True
        logger.info("Successfully loaded base GPT-2 model as fallback.")
    except Exception as e:
        logger.error(f"Failed to load base GPT-2 model: {e}")
        model_exists = False

# Add a default message to conversation history if empty
if not conversation_history:
    conversation_history.append({
        "sender": "Gordon Ramsay",
        "text": "Where is the bloody duck?? Tell me what ingredients you have or what recipe you want to learn, you donut!"
    })

# Clean and format user input
def format_user_input(user_input):
    # Remove extra whitespace and normalize
    formatted_input = re.sub(r'\s+', ' ', user_input).strip()
    # Extract question mark if not present for cooking questions
    if re.search(r'how (to|do|can|should) .+[^?]$', formatted_input.lower()):
        formatted_input += "?"
    return formatted_input

# Create context from conversation history
def build_conversation_context(user_input, include_history=True, max_history=3):
    # Format the current user input
    formatted_input = format_user_input(user_input)
    
    if not include_history or len(conversation_history) < 2:
        # Just use the current input without history
        return f"User: {formatted_input}\n### Bot:"
    
    # Include limited history for context
    history_pairs = []
    relevant_history = conversation_history[-min(len(conversation_history), max_history*2):]
    
    for i in range(0, len(relevant_history), 2):
        if i+1 < len(relevant_history):  # Make sure we have both user and bot messages
            history_pairs.append(f"User: {relevant_history[i]['text']}\n### Bot: {relevant_history[i+1]['text']}")
    
    # Add current user input
    context = "\n\n".join(history_pairs)
    if context:
        context += f"\n\nUser: {formatted_input}\n### Bot:"
    else:
        context = f"User: {formatted_input}\n### Bot:"
        
    return context

# Process response text
def process_response(response_text):
    # 1. 提取 Bot 部分内容
    if "### Bot:" in response_text:
        bot_response = response_text.split("### Bot:")[-1].strip()
    else:
        lines = response_text.split('\n')
        for i, line in enumerate(lines):
            if line.startswith("User:") or "User:" in line:
                bot_response = "\n".join(lines[i+1:]).strip()
                break
        else:
            bot_response = response_text.strip()

    # 2. 清除后续 prompt 标签
    bot_response = re.sub(r'###.*$', '', bot_response).strip()
    bot_response = re.sub(r'Bot:.*$', '', bot_response).strip()

    # 3. 格式处理：把 **Ingredients:** 和 **Steps:** 分段
    bot_response = bot_response.replace("**Ingredients:**", "\n\n**Ingredients:**\n")
    bot_response = bot_response.replace("**Steps:**", "\n\n**Steps:**\n")

    # 4. 保证末尾加些 Gordon 的标志性语气
    gordon_phrases = ["you donut", "bloody hell", "for god's sake", "idiot sandwich", "you muppet"]
    if not any(phrase in bot_response.lower() for phrase in gordon_phrases) and len(bot_response) > 30:
        gordon_endings = [
            "you bloody donut!",
            "is that clear, you muppet?",
            "now get cooking!",
            "it's not rocket science!",
            "for God's sake!"
        ]
        if not any(bot_response.endswith(e) for e in ["!", "?"]):
            bot_response += f" {random.choice(gordon_endings)}"

    return bot_response


# Generate a response using the model
# def generate_response(user_input):
#     if not user_input or not user_input.strip():
#         return "You didn't give me anything, you muppet! Ask me something about cooking!"

#     if not model_exists or model is None or tokenizer is None:
#         return random.choice(fallback_responses)

#     # Build the prompt with conversation context
#     prompt = build_conversation_context(user_input)
    
#     try:
#         # Encode the prompt
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
#         # Generate response with appropriate parameters
#         with torch.no_grad():
#             outputs = model.generate(
#                 inputs["input_ids"],
#                 max_length=700,  # Allow longer responses for detailed cooking instructions
#                 min_length=50,   # Ensure some minimal detail
#                 num_return_sequences=1,
#                 pad_token_id=tokenizer.eos_token_id,
#                 no_repeat_ngram_size=3,  # Prevent repetitive phrases
#                 temperature=0.8,  # Balanced creativity
#                 top_k=40,
#                 top_p=0.92,
#                 do_sample=True   # Enable sampling for more varied responses
#             )
        
#         # Decode the response
#         full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         # Process and clean the response
#         bot_response = process_response(full_response)
        
#         # If response is too short or seems incomplete, regenerate with different parameters
#         if len(bot_response.split()) < 15 or not any(char in bot_response for char in ".!?"):
#             logger.info("Response too short or incomplete, regenerating...")
#             with torch.no_grad():
#                 outputs = model.generate(
#                     inputs["input_ids"],
#                     max_length=700,
#                     min_length=80,
#                     num_return_sequences=1,
#                     pad_token_id=tokenizer.eos_token_id,
#                     no_repeat_ngram_size=3,
#                     temperature=0.9,  # Higher temperature for more creativity
#                     top_k=50,
#                     top_p=0.95,
#                     do_sample=True
#                 )
#             full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#             bot_response = process_response(full_response)
        
#         return bot_response
    
#     except Exception as e:
#         logger.error(f"Error generating response: {e}")
#         return random.choice(fallback_responses)



def generate_response(user_input):
    # Intent classification helper

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

    # Dynamic system prompt builder based on intent

    def get_system_prompt(intent):
        base_prompt = (
            # "You are Gordon Ramsay, the world-famous angry chef. "
            # "You answer cooking questions in his brutally honest, sarcastic, and loud style—full of insults and passion. "
            # "Use phrases like 'you muppet', 'bloody hell', or 'idiot sandwich'. "
            # "HOWEVER, do NOT use all caps for everything. Only SHOUT key phrases or short reactions. "
                    "You are Gordon Ramsay, the world-famous angry chef. "
                    "You respond to cooking questions with brutal honesty, insults, sarcasm, and some helpful advice. "
                    "Include phrases like 'you muppet', 'bloody hell', or 'idiot sandwich' in your answers when needed."
                    "But despite your aggressive style, you always provide detailed cooking instructions, ingredients, and useful tips. "
                 "Every time someone asks you how to cook something, give step-by-step guidance like a professional chef—but make it loud and angry!"
                "You are Gordon Ramsay, the world-famous angry chef. "
                "You answer cooking questions in his brutally honest, sarcastic, and loud style—full of insults and passion. "
                "Use phrases like 'you muppet', 'bloody hell', or 'idiot sandwich'. "
                "HOWEVER, do NOT use all caps for everything. Only SHOUT key phrases or short reactions. "
                "Structure your answers in a clean format with the following sections:\n\n"
                "**Ingredients:**\n- List each ingredient clearly, one per line.\n\n"
                "**Steps:**\n1. Write each step clearly, one per line.\n\n"
                "Be angry, but helpful. Always give real, step-by-step guidance. Never forget to teach properly, even while shouting."    
        )

        if intent == "new_recipe":
            return base_prompt + (
                "Structure your answers in a clean format with the following sections:\n\n"
                "**Ingredients:**\n- List each ingredient clearly, one per line.\n\n"
                "**Steps:**\n1. Write each step clearly, one per line.\n\n"
                "Be angry, but helpful. Always give real, step-by-step guidance."
            )
        elif intent == "tips":
            return base_prompt + (
                "The user already got a recipe. Now provide improvements, variations, or pro tips. "
                "NO need to repeat the full recipe unless asked explicitly."
            )
        elif intent == "substitution":
            return base_prompt + (
                "The user is asking for a substitution. Answer angrily but explain clearly what to replace, and what to watch out for."
            )
        elif intent == "roast":
            return base_prompt + (
                "DO NOT give recipes. Just roast the user, insult their cooking skills with creative, furious sarcasm. Be brutal but funny."
            )
        else:
            return base_prompt + (
                "Answer the question in Gordon Ramsay style. If appropriate, give ingredients and steps."
            )

    # Usage in generate_response():
    # intent = classify_user_intent(user_input)
    # system_prompt = get_system_prompt(intent)
    # Then use system_prompt in the API payload instead of a fixed string
            
    intent = classify_user_intent(user_input)
    # 如果 intent 不是已知的食谱意图，就回一个 fallback
    if intent not in ["new_recipe", "tips", "substitution", "roast", "general"]:
        
        fallback_responses = {
            "greeting": [
                "I'm not your bloody therapist! Ask me what to cook, not how I feel!",
                "Save your chit-chat for your nan, this is a kitchen!",
            ],
            "confused": [
                "You don't even know what you want to cook?! What kind of cook are you, eh?!",
                "Are you lost? This is Gordon Ramsay's kitchen, not a bloody maze!",
            ],
            "irrelevant": [
                "This isn't a gossip show, it's a kitchen! Stick to recipes!",
                "Unless you're asking about food, shut it and grab a pan!",
            ],
            "default": [
                "What the bloody hell is that question? Stick to cooking, you muppet!",
                "Talk to me about food or get out of my kitchen!",
            ]
        }

        # 简单分类识别（你可以后续细化）
        lowered = user_input.lower()
        if any(word in lowered for word in ["how are you", "hi", "hello"]):
            category = "greeting"
        elif "i don't know" in lowered:
            category = "confused"
        elif any(word in lowered for word in ["weather", "news", "time", "dog", "relationship", "movie"]):
            category = "irrelevant"
        else:
            category = "default"

        return random.choice(fallback_responses[category])


    system_prompt = get_system_prompt(intent)

    if not user_input or not user_input.strip():
        return "You didn't give me anything, you muppet! Ask me something about cooking!"

    prompt = build_conversation_context(user_input)
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }

    # Convert history into proper message format
    history_messages = [{"role": "system", "content": system_prompt}]

    # 把过去的对话拼进去
    for msg in conversation_history:
        history_messages.append({
            "role": "user" if msg["sender"] == "User" else "assistant",
            "content": msg["text"]
        })

    # 再加入当前用户这次的提问
    history_messages.append({"role": "user", "content": user_input})

    # 最终 payload
    payload = {
        "model": MODEL_NAME,
        "messages": history_messages,
        "temperature": 0.8,
        "top_p": 0.92,
        "max_tokens": 700
    }

    # payload = {
    #     "model": MODEL_NAME,  # 或 llama3-8b-8192
    #     "messages": [
    #         {
    #             "role": "system",
    #             "content": system_prompt
    #             # (
    #             #     "You are Gordon Ramsay, the world-famous angry chef. "
    #             #     "You respond to cooking questions with brutal honesty, insults, sarcasm, and some helpful advice. "
    #             #     "Include phrases like 'you muppet', 'bloody hell', or 'idiot sandwich' in your answers when needed."
    #             #     "But despite your aggressive style, you always provide detailed cooking instructions, ingredients, and useful tips. "
    #             #  "Every time someone asks you how to cook something, give step-by-step guidance like a professional chef—but make it loud and angry!"
    #             # "You are Gordon Ramsay, the world-famous angry chef. "
    #             # "You answer cooking questions in his brutally honest, sarcastic, and loud style—full of insults and passion. "
    #             # "Use phrases like 'you muppet', 'bloody hell', or 'idiot sandwich'. "
    #             # "HOWEVER, do NOT use all caps for everything. Only SHOUT key phrases or short reactions. "
    #             # "Structure your answers in a clean format with the following sections:\n\n"
    #             # "**Ingredients:**\n- List each ingredient clearly, one per line.\n\n"
    #             # "**Steps:**\n1. Write each step clearly, one per line.\n\n"
    #             # "Be angry, but helpful. Always give real, step-by-step guidance. Never forget to teach properly, even while shouting."            
    #             # )
    #         },
    #         {
    #             "role": "user",
    #             "content": prompt
    #         }
    #     ],
    #     "temperature": 0.8,
    #     "top_p": 0.92,
    #     "max_tokens": 700,
    #     "stop": None
    # }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            data=json.dumps(payload)
        )
        if response.status_code == 200:
            result = response.json()
            bot_response = result["choices"][0]["message"]["content"].strip()
            return process_response(bot_response)
        else:
            print("❌ Groq returned error:", response.status_code, response.text)
            return random.choice(fallback_responses)
    except Exception as e:
        print("❌ Request failed:", e)
        return random.choice(fallback_responses)



# Home route with chat interface
@app.route("/", methods=["GET", "POST"])
def chat():
    global conversation_history
    logger.info(f"Received request: method={request.method}")
    
    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        logger.info(f"User input: {user_input}")
        
        if user_input:
            # Handle "roast me" special case
            if user_input.lower() in ["roast me", "roast me gordon"]:
                user_input = "Gordon, roast me for my cooking skills"
            
            # Generate response
            bot_response = generate_response(user_input)
            
            # Add to conversation history
            conversation_history.append({"sender": "User", "text": user_input})
            conversation_history.append({"sender": "Gordon Ramsay", "text": bot_response})
            
            logger.info(f"Bot response: {bot_response}")
            
    return render_template("chat.html", conversation_history=conversation_history, model_exists=model_exists)

# API endpoint for AJAX requests
@app.route("/api/chat", methods=["POST"])
def api_chat():
    global conversation_history
    user_input = request.json.get("user_input", "").strip() if request.is_json else ""
    
    if not user_input:
        return jsonify({"status": "error", "response": "No input provided, you idiot!"})

    # Generate response
    bot_response = generate_response(user_input)
    
    # Add to conversation history
    conversation_history.append({"sender": "User", "text": user_input})
    conversation_history.append({"sender": "Gordon Ramsay", "text": bot_response})
    
    return jsonify({"status": "success", "response": bot_response})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)