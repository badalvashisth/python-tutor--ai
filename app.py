"""
Python Tutor AI - Flask backend
--------------------------------
Serves the chat UI and proxies chat messages to the Mistral API.

The Mistral API key is loaded from the environment (never exposed to the
browser) and the running conversation is kept server-side, in the user's
Flask session, exactly following the "messages list" pattern:

    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    messages.append({"role": "user", "content": user_input})
    response = client.chat.complete(model="mistral-large-latest", messages=messages)
    messages.append({"role": "assistant", "content": assistant_reply})
"""

import os
import logging

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from mistralai.client  import Mistral

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()  # reads MISTRAL_API_KEY (and FLASK_SECRET_KEY) from a local .env file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("python-tutor-ai")

MISTRAL_MODEL = "mistral-large-latest"
MAX_USER_MESSAGE_LENGTH = 4000  # basic guardrail against runaway payloads

app = Flask(__name__)

# Needed so Flask can sign the session cookie that stores each visitor's
# conversation history. Falls back to a random key so the app still runs
# in local/dev if it isn't set, but a real deployment should set this.
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(32)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    # We don't crash on import so the page can still load and show a clear
    # error, but every /chat call will fail fast with a helpful message.
    logger.warning("MISTRAL_API_KEY is not set. /chat will return an error until it is.")

client = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None

# ---------------------------------------------------------------------------
# System instruction - keeps the assistant scoped to Python tutoring only
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """You are an expert Python programming tutor.

Your job is ONLY to answer questions related to Python programming.

You can answer questions about:
- Python syntax
- Python variables
- Data types
- Strings
- Lists
- Tuples
- Sets
- Dictionaries
- Loops
- Functions
- Classes and OOP
- Modules and packages
- Exceptions
- File handling
- APIs
- Python libraries
- Python debugging
- Python projects
- Python best practices
- Python code explanations
- Python errors
- Python interview questions

Rules:
1. Only answer Python-related questions.
2. If the user asks about a non-Python topic, politely refuse and say that you are specifically a Python tutor.
3. Do not answer general questions unrelated to Python.
4. Explain concepts clearly and step-by-step.
5. Assume the user may be a beginner.
6. Provide practical Python examples when useful.
7. When the user provides Python code, analyze the code and explain errors clearly.
8. Maintain the context of the conversation.
9. Remember previous Python-related messages within the current conversation.
10. Do not pretend to remember conversations that are not present in the provided chat history."""


def get_conversation():
    """Return this visitor's message list, creating it (with the system
    instruction as the first entry) if it doesn't exist yet."""
    if "messages" not in session:
        session["messages"] = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    return session["messages"]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the single-page chat UI."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Receive one user message, call Mistral with the full conversation
    history, store the reply, and return it to the browser."""

    if client is None:
        return jsonify({"error": "Server is missing MISTRAL_API_KEY. Set it in your .env file and restart."}), 500

    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    if len(user_message) > MAX_USER_MESSAGE_LENGTH:
        return jsonify({"error": f"Message is too long (max {MAX_USER_MESSAGE_LENGTH} characters)."}), 400

    messages = get_conversation()
    messages.append({"role": "user", "content": user_message})

    try:
        # Single call to Mistral per user message, with the full running
        # conversation (system + prior turns + this new user turn).
        response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=messages,
        )
        assistant_reply = response.choices[0].message.content

    except Exception as exc:  # noqa: BLE001 - we want to catch any SDK/network error
        # Roll back the user message we just appended so the stored history
        # stays valid (no dangling turn with no reply) and the visitor can
        # simply try again.
        messages.pop()
        session["messages"] = messages
        logger.exception("Mistral API call failed")
        return jsonify({"error": "The tutor is temporarily unavailable. Please try again in a moment."}), 502

    messages.append({"role": "assistant", "content": assistant_reply})
    session["messages"] = messages  # persist the updated history in the session

    return jsonify({"reply": assistant_reply})


@app.route("/reset", methods=["POST"])
def reset():
    """Optional helper: clear the current visitor's conversation history."""
    session.pop("messages", None)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)