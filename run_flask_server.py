from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer
import os
import dotenv
import hashlib

dotenv.load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")

# Initialize Flask app
app = Flask(__name__)


# Initialize Qdrant client and sentence transformer model
def init_vector_db():
    client = QdrantClient("http://localhost:6333")

    model = SentenceTransformer(MODEL_NAME)

    return client, model


client, model = init_vector_db()

secret_key = os.getenv("SECRET_KEY")
assert secret_key, "SECRET_KEY is not set in .env"


# Helper function to handle searching for similar notes
def search_similar_notes(query, threshold):
    query_vector = model.encode(query).tolist()
    results = client.search(
        collection_name=COLLECTION_NAME, query_vector=query_vector, limit=10
    )

    notes = []
    for result in results:
        if result.score < threshold:
            continue
        notes.append(
            {
                "title": result.payload["title"].strip(),
                "similarity": round(result.score, 2),
                "content": result.payload["content"][
                    :100
                ],  # First 100 characters of content
            }
        )

    return notes


# Route to query notes
@app.route("/query", methods=["POST"])
def query_notes():
    try:
        # Get the query content from the request body
        data = request.get_json()
        query = data.get("query")
        threshold = float(data.get("threshold"))

        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Search for similar notes
        similar_notes = search_similar_notes(query, threshold)

        return jsonify({"query": query, "results": similar_notes})
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


# Route to embed and add a new note (useful if you need to insert new notes into the DB)
@app.route("/add_note", methods=["POST"])
def add_note():
    try:
        # Get note data from request body
        data = request.get_json()
        title = data.get("title")
        content = data.get("content")

        if not title or not content:
            return jsonify({"error": "Title and content are required"}), 400

        # Hash note title with secret key
        hash_object = hashlib.sha256()
        hash_object.update(title.encode("utf-8") + secret_key.encode("utf-8"))
        note_id = int(hash_object.hexdigest(), 16)

        # Embed the content
        embedding = model.encode(content).tolist()

        # Store the note in the database
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=note_id,
                    payload={"title": title, "content": content},
                    vector=embedding,
                )
            ],
        )

        return jsonify({"message": "Note added successfully", "note_id": note_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run the Flask web server
    app.run(debug=True)
