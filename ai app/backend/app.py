from flask import Flask, request, jsonify
from flask_cors import CORS
from backend.query_handler import answer_query, get_available_documents

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend → backend calls

# Health check endpoint
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Contract AI Backend is running"}), 200

# Endpoint to get list of available documents
@app.route("/documents", methods=["GET"])
def list_documents():
    try:
        documents = get_available_documents()
        return jsonify({"documents": documents}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to ask a question and get an answer
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")
    top_k = int(data.get("top_k", 5))
    model = data.get("model", "gpt-4")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        result = answer_query(question, top_k=top_k, model=model)
        return jsonify({
            "answer": result["answer"],
            "chunks_used": result["chunks_used"]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


