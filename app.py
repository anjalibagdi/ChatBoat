from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import get_response
from dotenv import load_dotenv
import os
import uuid

# Load environment variables
load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})  # Restrict CORS to React app

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question")
    session_id = data.get("session_id", str(uuid.uuid4()))
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    try:
        answer = get_response(question, session_id)
        print("answer", answer)
        return jsonify({
            "question": question,
            "answer": answer,
            "session_id": session_id
        })
    except Exception as e:
        print("error", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "message": "API is running"}), 200

if __name__ == "__main__":
    print("Starting Flask server on http://0.0.0.0:4000")
    app.run(debug=True, host="0.0.0.0", port=4000)