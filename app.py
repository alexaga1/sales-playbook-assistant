"""Web demo for the Sales Playbook Assistant.

Run:  python3 app.py   ->  open http://127.0.0.1:5000
"""
from flask import Flask, jsonify, render_template, request

from assistant import KnowledgeAssistant

app = Flask(__name__)
assistant = KnowledgeAssistant()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def ask():
    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Ask a question first."}), 400
    return jsonify(assistant.generate_answer(question))


if __name__ == "__main__":
    app.run(debug=True)
