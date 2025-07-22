
from flask import Flask, request, render_template_string

app = Flask(__name__)

conversation_history = []

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <title>Basic Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2em; }
        .chat-box { border: 1px solid #ccc; padding: 1em; margin-top: 1em; }
        .user { color: blue; }
        .bot { color: green; }
    </style>
</head>
<body>
    <h2>Simple Chatbot</h2>
    <form method="post">
        <input type="text" name="user_input" required style="width: 300px;">
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

@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        bot_response = generate_simple_response(user_input)
        conversation_history.append({"sender": "User", "text": user_input})
        conversation_history.append({"sender": "Bot", "text": bot_response})
    return render_template_string(HTML_TEMPLATE, conversation_history=conversation_history)

def generate_simple_response(user_input):
    if "recipe" in user_input.lower():
        return "Here's a basic recipe: 2 eggs, salt, and pepper. Beat and cook!"
    elif "hello" in user_input.lower():
        return "Hello there! Ask me anything about cooking."
    return "Sorry, I can only help with recipes right now."

if __name__ == "__main__":
    app.run(debug=True)
