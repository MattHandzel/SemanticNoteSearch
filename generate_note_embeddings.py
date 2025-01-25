from utils import *
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer
import os
import dotenv
from nlp_utils import clean_text_for_embedding_model

dotenv.load_dotenv()
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
print("COLLECTION_NAME is", COLLECTION_NAME)


def init_vector_db():
    client = QdrantClient("http://localhost:6333")
    model = SentenceTransformer(MODEL_NAME)

    print(client.get_collections().collections)
    if not any(
        [c.name == COLLECTION_NAME for c in client.get_collections().collections]
    ):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE
            ),
        )

    return client, model


client, model = init_vector_db()

import hashlib

secret_key = os.getenv("SECRET_KEY")
assert secret_key, "SECRET_KEY is not set in .env"


def embed_all_notes_into_vector_database(note_path, meta_data, content, root_directory):
    note_title = (
        ("\t" + meta_data["title"])
        if "title" in meta_data
        else ("\t" + str(meta_data["id"]) if "id" in meta_data else note_path)
    )
    # hash note_title with secret key
    hash_object = hashlib.sha256()
    hash_object.update(note_title.encode("utf-8") + secret_key.encode("utf-8"))
    note_id = int(hash_object.hexdigest()[:16], 16)

    # Check if note is already in the database
    embedded_content = model.encode(clean_text_for_embedding_model(content))
    search_result = client.search(
        collection_name=COLLECTION_NAME, query_vector=embedded_content, limit=1
    )

    if not search_result or search_result[0].score < 0.95:
        # Embed and store the note
        embedding = embedded_content.tolist()
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=note_id,
                    payload={
                        "title": note_title,
                        "content": content,
                    },  # TODO: Add note creation time/edit time to payload
                    vector=embedding,
                )
            ],
        )
        print(f"Embedded note: {note_title}")
    else:
        print(f"Note already embedded: {note_title}")

    return None, None, False


def query_notes(query):
    query_vector = model.encode(clean_text_for_embedding_model(query)).tolist()
    results = client.search(
        collection_name=COLLECTION_NAME, query_vector=query_vector, limit=10
    )

    print(f"Query: {query}")
    print("Results:")
    for result in results:
        print(f"- {result.payload['title']} (similarity: {result.score:.2f})")
        print(f"  Content: {result.payload['content'][:100]}...")
        print("-----------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_directory", type=str)
    parser.add_argument("--query", type=str, help="Query to search for similar notes")
    args = parser.parse_args()
    root_directory = args.root_directory

    if args.query:
        query_notes(args.query)
    else:
        loop_through_notes(
            root_directory,
            [embed_all_notes_into_vector_database],
            clear_bottom_matter=False,
        )
