import os

QDRANT_STORAGE_DIR = "./qdrant_storage"


def start_qdrant_server():
    os.system(
        f"docker run -p 6333:6333 -v {QDRANT_STORAGE_DIR}:/qdrant/storage qdrant/qdrant &"
    )
