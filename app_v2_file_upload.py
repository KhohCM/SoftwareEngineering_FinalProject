
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
import os
from docx import Document
from PyPDF2 import PdfReader

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

conversation_history = []
memory_store = {}

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <title>Chatbot with File Upload</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2em; }
        .chat-box { border: 1px solid #ccc; padding: 1em; margin-top: 1em; }
        .user { color: blue; }
        .bot { color: green; }
    </style>
</head>
<body>
    <h2>Chatbot with File Upload</h2>
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
            combined_input += f"\n[From file]: {memory_store['content'][:200]}..."

        bot_response = generate_simple_response(combined_input)
        conversation_history.append({"sender": "User", "text": user_input})
        conversation_history.append({"sender": "Bot", "text": bot_response})

    return render_template_string(HTML_TEMPLATE, conversation_history=conversation_history)

def generate_simple_response(user_input):
    if "ingredient" in user_input.lower():
        return "Make sure to check your ingredients list in the uploaded file."
    elif "step" in user_input.lower() or "instruction" in user_input.lower():
        return "Follow the cooking steps carefully as described in the document."
    return "Thanks for your message. I'll use your uploaded file to help better next time!"

if __name__ == "__main__":
    app.run(debug=True)
