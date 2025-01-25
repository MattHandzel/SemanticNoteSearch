# MAKE A BACKUP OF YOUR VAULT BEFORE RUNNING THIS SCRIPT!

### Steps

1. Create a .env file with the following variables:

```
SECRET_KEY=
COLLECTION_NAME=vault
```

2. Run `sh install.sh` to install the necessary dependencies

3. Start the qdrant server, you can change `qdrant_storage` to be where you want to store the embeddings locally:

```sh
docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```

4. Choose the directories and/or files you want to encode and the script to create the note embeddings:

```py
python generate_note_embeddings --directories directory_1 directory_2 directory_3 --files file_1 file_2 file_3
```

This will generate embeddings for those notes in

5. Now, run the python flask server (`python run_flask_server.py`), and you will be able to query it for similar notes!
