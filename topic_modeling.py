"""
I have a lot of documents in my notes that are related to other documents in my notes, but they are not linked together with wiki-links. I want to find common topics between notes and then create wiki-links. If two notes are related to each other, I also want to link them together. I don't want to care too much about the formatting, just the semantic content of the notes.

Remove formatting/cleaning data:
    - Use the functions in nlp_utils.py if applicable

I want to have a database to store information persistently. In the database, it will store the note's path, it's hash, the topics in the document, and the vector embeddings.

I want this to be an efficient system that periodically runs in the background, checks to see if notes were changed, and if the notes were changed 

Check to see if note was changed:
    Every time the notes are processed and right before the notes are written, store the note path as a key in the database and then a sha-256 hash of the document. If the note is changed then the sha-256 hash will be different and the note will be processed again.

Use LDA from gensim initially, then try out TopicBERT

For the similar topics, I could maybe have a human label the similarities (and what words come from the topic), or I could use the similarity of that vector with the embeddings of wikipedia and automatically choose the topic

The project should be high quality code, with error checking, log statements, for every function from database creation and management, to the topic modeling.

Include visualizations of the topics and print some examples of documents and the topics involved. Compute the topic similarity heatmap.

Also, add functions and code to do the following, but make them optional (i.e. flags to disable and enable the following):
    - I could do topic modeling on each portion of the document, that might give better results and I could put the links at the end of the different segments.
    - I could seed the topics with the current links, as well as wikepedia links or links from other obsidian vaults
    - Because properties like tags and links already define topics, I could use them for seeding, or I could also change them to be special tokens, and their TF-IDF scores would be high, hopefully they would semantically correspond to the correct topics
    - How to handle different languages? Maybe I can classify all of the documents based on language and only run topic modeling on the chosen languagee. I could also translate the languages into english. 
    - Add features to visualize data from the database such as clusters and graphs 
    - Maybe I could use vector embeddings to do document-document similarity using SOA language models, and then combine the results with topic modeling.

"""

import os
import hashlib
import sqlite3
from gensim import corpora
from gensim.models import LdaModel
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nlp_utils import *
from utils import *
from gensim.test.utils import datapath
import pickle
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# DATABASE SETUP
DATABASE_PATH = "notes.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformer model
LDA_MODEL_PATH = "./lda/lda_model"
LDA_DICTIONARY_PATH = "./lda/lda_dictionary"
LDA_CORPUS_PATH = "./lda/lda_corpus"
lda_model = None

# TODO: Increase n-gram frequency to be 3 (monte carlo method), or even 4


def semantically_generate_topics(note_path, meta_data, content, root_directory):
    # TODO:

    pass


def cleanse_content(content):
    # Clean and preprocess content
    return remove_words(
        remove_numbers(
            convert_to_lowercase(
                remove_whitespace(
                    remove_special_characters(
                        remove_html_tags(
                            (
                                clean_custom_patterns(
                                    remove_words((remove_markdown_formatting(content)))
                                )
                            )
                        )
                    )
                )
            )
        )
    )


def tokenize_content(content):
    tokens = tokenize_text(content)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_text(tokens)
    return tokens


def create_database():
    """Creates the SQLite database to store notes."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS Notes (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      path TEXT UNIQUE,
                      hash TEXT,
                      topics TEXT,
                      embedding BLOB)"""
    )
    conn.commit()
    conn.close()


def store_note_data(path, note_hash, topics, embedding):
    """Stores or updates the note's data in the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT OR REPLACE INTO Notes (path, hash, topics, embedding) 
                      VALUES (?, ?, ?, ?)""",
        (path, note_hash, topics, embedding),
    )
    conn.commit()
    conn.close()


def get_note_hash_from_db(path):
    """Fetches the hash of a note from the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT hash FROM Notes WHERE path = ?", (path,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


def get_all_embeddings_and_paths():
    """Fetches all embeddings and their paths from the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT path, embedding FROM Notes")
    results = cursor.fetchall()
    conn.close()
    return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in results]


def get_all_topics_for_note(note_path):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT topic FROM Notes WHERE path= ? ", (note_path,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


# HASH CALCULATION
def calculate_hash(content):
    """Calculates the SHA-256 hash of the document content."""
    return hashlib.sha256(content.encode()).hexdigest()


topic_id_words_map = {}


def run_inference_on_lda_model(lda_model, dictionary, content):
    tokens = prepare_raw_content_for_lda(content)
    bow = dictionary.doc2bow(tokens)
    topics = lda_model.get_document_topics(bow)
    return topics


# TOPIC MODELING
def lda_topic_modeling(preprocessed_docs, num_topics=50):
    """Performs LDA topic modeling on a list of preprocessed documents."""
    dictionary = corpora.Dictionary(preprocessed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]
    logger.info("Training LDA model...")

    best_model = None
    best_score = 0
    best_num_topics = -1
    for num_topics in range(20, 200, 10):
        lda_model = LdaModel(
            corpus,
            num_topics=num_topics,
            id2word=dictionary,
            alpha="auto",
            iterations=50,
            passes=10,
            eval_every=5,
            random_state=42,
        )
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=preprocessed_docs,
            dictionary=dictionary,
            coherence="c_v",
            processes=8,
        )
        score = coherence_model.get_coherence()
        if score > best_score:
            best_model = lda_model
            best_score = score
            best_num_topics = num_topics

        print(f"Num Topics: {num_topics}, Coherence Score: {score}")
    logger.info(
        f"Best number of topics is {best_num_topics}, best coherence is {best_score}"
    )

    return best_model, dictionary, corpus


# EMBEDDING GENERATION
def generate_embedding(content, model):
    """Generates a vector embedding for the given content."""
    return model.encode(content)


# LINK CREATION
def find_similar_notes(target_embedding, all_embeddings, threshold=0.8):
    """Finds similar notes by comparing embeddings using cosine similarity."""
    similar_notes = []
    for path, embedding in all_embeddings:
        similarity = cosine_similarity(target_embedding, embedding)
        if similarity >= threshold:
            similar_notes.append(path)
    return similar_notes


def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


all_documents = []
potential_topics_from_documents = set()
potential_meta_data_topics_properties = [
    "tags",
    "links",
    "categories",
    "tag",
    "link",
    "category",
]


def prepare_raw_content_for_lda(content):
    tokens = tokenize_content(cleanse_content(content))
    return tokens


def add_note_to_training_set(note_path, meta_data, content, root_directory):
    if detect_language(content) != "en":
        return None, None, False

    all_documents.append(prepare_raw_content_for_lda(content))
    for property_name in potential_meta_data_topics_properties:
        if property_name in meta_data:
            potential_topics_from_documents.update(meta_data[property_name])
    return None, None, False  # This code doesn't do anything


def add_semantic_topics_to_notes(note_path, meta_data, content, root_directory):
    # TODO: In the future, we can train the LDA model every so often and just use LDA model to extract the topics

    topics = run_inference_on_lda_model(lda_model, dictionary, content)
    print(topics)
    raise
    # return {"topics": topics}, content, False


# VISUALIZATION
def visualize_topics_heatmap(topics, lda_model, dictionary):
    """Generates a heatmap of topic-word probabilities."""
    topic_words = []
    for topic_id, topic in lda_model.show_topics(formatted=False):
        topic_words.append([word for word, prob in topic])

    sns.heatmap(topic_words, annot=True, fmt="", cmap="coolwarm")
    plt.title("Topic-Word Heatmap")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_directory", type=str)

    args = parser.parse_args()
    root_directory = args.root_directory

    NUM_TOPICS = 10

    if (
        not os.path.exists(LDA_MODEL_PATH) or not os.path.exists(LDA_DICTIONARY_PATH)
    ) or DEBUG:
        loop_through_notes(root_directory, [add_note_to_training_set])

        print("Training LDA model...")

        lda_model, dictionary, corpus = lda_topic_modeling(
            all_documents, num_topics=NUM_TOPICS
        )
        print("LDA model trained.")
        lda_model.save(LDA_MODEL_PATH)
        dictionary.save(LDA_DICTIONARY_PATH)
        # save corpus
        with open(LDA_CORPUS_PATH, "wb") as f:
            pickle.dump(corpus, f)

    else:
        lda_model = LdaModel.load(LDA_MODEL_PATH)
        dictionary = corpora.Dictionary.load(LDA_DICTIONARY_PATH)
        with open(LDA_CORPUS_PATH, "rb") as f:
            corpus = pickle.load(f)
        NUM_TOPICS = len(lda_model.get_topics())

    all_topics = lda_model.show_topics(num_topics=NUM_TOPICS, num_words=25)
    for topic_id, words in all_topics:
        topic_id_words_map[topic_id] = words

    print("Topics:")
    for topic_id, words in all_topics:
        print(f"Topic {topic_id}: {words}")

    vis = gensimvis.prepare(lda_model, corpus, dictionary)

    # Alternatively, save it to an HTML file
    pyLDAvis.save_html(vis, "lda_visualization.html")

    exit()
    loop_through_notes(root_directory, [add_semantic_topics_to_notes], max_notes=200)
