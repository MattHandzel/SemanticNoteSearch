# 1h55 for 82198 documents
import os
import hashlib
import sqlite3
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from bertopic import BERTopic

from sentence_transformers import SentenceTransformer
from nlp_utils import *
from utils import *

DATABASE_PATH = "notes.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformer model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
TOPIC_BERT_MODEL_PATH = "./topic_bert_model_all.pkl"
bert_topic_model = None
NUM_WIKI_FILES = 2_000_000

# TODO: Download all wikipedia topics and then run on all articles
# TODO: Discover a better way of converting wikipedia to mardown, so we have anly content, nothing "meta"
# TODO: Tune topic modeling
# TODO: Increase n-gram frequency to be 3 (monte carlo method), or even 4


# For demonstration, we keep the same database structure, but remove LDA-specific references
all_token_lengths = []


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
            embedding BLOB
        )"""
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


def calculate_hash(content):
    """Calculates the SHA-256 hash of the document content."""
    return hashlib.sha256(content.encode()).hexdigest()


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


def add_note_to_training_set(note_path, meta_data, content, root_directory):
    """
    Gathers documents for training. We still check language, then
    collect them for topic modeling. (Same structure as before.)
    """
    if detect_language(content) != "en":
        return None, None, False

    # Prepare content for bert_topic
    prepared_text = prepare_raw_content_for_bert_topic(content)
    all_documents.append(prepared_text)

    # Also gather potential topics from meta_data if desired
    for property_name in potential_meta_data_topics_properties:
        if property_name in meta_data:
            potential_topics_from_documents.update(meta_data[property_name])

    return None, None, False


def run_inference_on_bert_topic_model(topic_bert_model, content):
    """
    Generate topics from a trained bert_topic model given new content.
    """
    prepared_text = prepare_raw_content_for_bert_topic(content)
    topics, _ = topic_bert_model.transform([prepared_text])
    return topics


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


def generate_embedding(content, model):
    """Generates a vector embedding for the given content using SentenceTransformer."""
    return model.encode(content)


def bert_topic_topic_modeling(docs):
    """
    Performs BERTopic-based modeling (bert_topic).
    'docs' should be a list of strings (already preprocessed).
    """
    # TODO: Save each individual model
    # Initialize BERTopic

    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    from bertopic.representation import KeyBERTInspired
    from bertopic import BERTopic

    # Create your representation model
    representation_model = KeyBERTInspired()
    hdbscan_model = HDBSCAN(
        min_cluster_size=50,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words="english", min_df=2, ngram_range=(1, 2)
    )

    umap_model = UMAP(
        n_neighbors=15,
        n_components=10,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=False,
    )
    embeddings = embedding_model.encode(docs)
    topic_model = BERTopic(
        language="english",
        # Pipeline models
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        # Hyperparameters
        top_n_words=20,
        verbose=True,
    )

    # Train model
    topics, _ = topic_model.fit_transform(docs, embeddings)
    return topic_model, topics


def visualize_topics_heatmap(topic_bert_model):
    """
    Example: use BERTopic's built-in visualization or generate custom plots.
    Below is a placeholder heatmap logic. You may tailor it to your needs.
    """
    fig = topic_bert_model.visualize_heatmap()
    fig.show()


import os
from multiprocessing import Pool


def process_file(file_path):
    delete_it = False
    if file_path.endswith(".md"):
        assert os.path.exists(file_path)
        with open(file_path, "r") as f:
            content = f.read()
            length = len(content)
            if not (length < 100):
                return prepare_raw_content_for_bert_topic(content)
            else:
                # delete it
                delete_it = True
        if delete_it:
            os.remove(file_path)
            return None

    return None


import random


def add_extra_directories_to_train_data(directories):
    all_files = []
    random.seed(42)
    for directory in directories:
        l = os.listdir(directory)
        random.shuffle(l)
        all_files.extend([os.path.join(directory, z) for z in l[:NUM_WIKI_FILES]])

    with Pool() as pool:
        results = pool.map(process_file, all_files)

    all_documents.extend([result for result in results if result is not None])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_directory", type=str)
    args = parser.parse_args()
    root_directory = args.root_directory
    data_dirs = ["./wiki_md_output/"]

    create_database()

    # Loop through your notes, collecting training data
    # (Assuming something like loop_through_notes exists in utils)
    loop_through_notes(root_directory, [add_note_to_training_set])
    add_extra_directories_to_train_data(data_dirs)
    print(len(all_documents), "documents collected for training.")

    print("Training bert_topic model with BERTopic...")
    for i, doc in enumerate(all_documents):
        all_token_lengths.append(len(doc.split()))
    logger.info(f"Average token length:{np.mean(all_token_lengths)}")
    try:
        with open("all_token_lengths.pkl", "wb") as f:
            pickle.dump(all_token_lengths, f)
    except Exception as e:
        logger.error(e)

    # Train bert_topic on the collected documents
    bert_topic_model, topics = bert_topic_topic_modeling(all_documents)
    bert_topic_model.save(TOPIC_BERT_MODEL_PATH)  # Save the BERTopic model
    print("bert_topic model trained and saved.")

    # Optionally, visualize the topics in a heatmap
    visualize_topics_heatmap(bert_topic_model)

    # Print discovered topics
    # You can retrieve topic info via topic_bert_model.get_topic_info()
    topic_info = bert_topic_model.get_topic_info()
    print("Topics:")
    print(topic_info)

    # Example usage for inference on a single note's content:
    new_note_content = """
The N400 is a component of time-locked EEG signals known as
event-related potentials (ERP). It is a negative-going deflection that
peaks around 400 milliseconds post-stimulus onset, although it can
extend from 250-500 ms, and is typically maximal over centro-parietal
electrode sites. The N400 is part of the normal brain response to words
and other meaningful (or potentially meaningful) stimuli, including
visual and auditory words, sign language signs, pictures, faces,
environmental sounds, and smells.(See Kutas & Federmeier, 2009, for
review) History The N400 was first discovered by Marta Kutas and Steven
Hillyard in 1980. They conducted the first experiment looking at the
response to unexpected words in read sentences, expecting to elicit a
P300 component. The P300 had previously been shown to be elicited by
unexpected stimuli. Kutas and Hillyard therefore used sentences with
anomalous endings (i.e.I take coffee with cream and dog), expecting to
see a P300 to the unexpected sentence-final words. However, instead of
eliciting a large positivity, these anomalous endings elicited a large
negativity, relative to the sentences with expected endings (i.e. He
returned the book to the library) In the same paper, they confirmed that
the negativity was not just caused by any unexpected event at the end of
a sentence, since a semantically expected but physically unexpected word
(i.e. She put on her high-heeled SHOES) elicited a P300 instead of
negativity in the N400 window. This finding showed that the N400 is
related to semantic processing, and is not just a response to unexpected
words. Component characteristics The N400 is characterized by a distinct
pattern of electrical activity that can be observed at the scalp. As its
name indicates, this waveform peaks around 400 ms post-stimulus onset,
with negativity that can be observed in the time window ranging from
250-500 ms. This latency (delay between stimulus and response) is very
stable across tasks—little else besides age affects the timing of its
peak. The N400 is a negative component, relative to reference electrodes
placed on the mastoid processes (the bony ridge behind the ear), and
relative to a 100 ms pre-stimulus baseline. Its amplitude can range from
-5 to 5 microvolts. However, it is important to note that in studies
using the N400 as a dependent measure, the relative amplitude of the
waveform compared to another experimental condition (the "N400 effect")
is more important than its absolute amplitude. The N400 itself is not
always negative—it is just a more negative-going deflection than that
seen to other conditions. Its distribution is maximal over
centro-parietal electrode sites, and is slightly larger over the left
side of the head for visual words, although the distribution can change
slightly depending on the eliciting stimulus. Main paradigms A typical
experiment designed to study the N400 will usually involve the visual
presentation of words, either in sentence or list contexts. In a typical
visual N400 experiment, for example, subjects will be seated in front of
a computer monitor while words are presented one-by-one at a central
screen location. Stimuli must be presented centrally because eye
movements will generate large amounts of electrical noise that will mask
the relatively small N400 component. Subjects will often be given a
behavioral task (e.g., making a word/nonword decision, answering a
comprehension question, responding to a memory probe), either after each
stimulus or at longer intervals, to ensure that subjects are paying
attention. Note, however, that overt responses by the subject are not
required to elicit the N400—passively viewing stimuli will still evoke
this response. An example of an experimental task used to study the N400
is a priming paradigm. Subjects are shown a list of words in which a
prime word is either associatively related to a target word (e.g. bee
and honey), semantically related (e.g. sugar and honey) or a direct
repetition (e.g. honey and honey). The N400 amplitude seen to the target
word (honey) will be reduced upon repetition due to semantic priming.
The amount of reduction in amplitude can be used to measure the degree
of relatedness between the words. Another widely used experimental task
used to study the N400 is sentence reading. In this kind of study,
sentences are presented to subjects centrally, one word at a time, until
the sentence is completed. Alternatively, subjects could listen to a
sentence as natural auditory speech. Again, subjects may be asked to
respond to comprehension questions periodically throughout the
experiment, although this is not necessary. Experimenters can choose to
manipulate various linguistic characteristics of the sentences,
including contextual constraint or the cloze probability of the
sentence-final word (see below for a definition of cloze probability) to
observe how these changes affect the waveform's amplitude. As previously
mentioned, the N400 response is seen to all meaningful, or potentially
meaningful, stimuli. As such, a wide range of paradigms can be used to
study it. Experiments involving the presentation of spoken words,
acronyms, pictures embedded at the end of sentences, music, words
related to current context or orientation and videos of real-word
events, have all been used to study the N400, just to name a few.
Functional sensitivity Extensive research has been carried out to better
understand what kinds of experimental manipulations do and do not affect
the N400. General findings are discussed below. Factors that affect N400
amplitude The frequency of a word's usage is known to affect the
amplitude of the N400. With all else being constant, highly frequent
words elicit reduced N400s relative to infrequent words. As previously
mentioned, N400 amplitude is also reduced by repetition, such that a
word's second presentation exhibits a more positive response when
repeated in context. These findings suggest that when a word is highly
frequent or has recently appeared in context, it eases the semantic
processing thought to be indexed by the N400, reducing its amplitude.
N400 amplitude is also sensitive to a word's orthographic neighborhood
size, or how many other words differ from it by only one letter (e.g.
boot and boat). Words with large neighborhoods (that have many other
physically similar items) elicit larger N400 amplitudes than do words
with small neighborhoods. This finding also holds true for pseudowords,
or pronounceable letter strings that are not real words (e.g. flom),
which are not themselves meaningful but look like words. This has been
taken as evidence that the N400 reflects general activation in the
comprehension network, such that an item that looks like many words
(regardless of whether it itself is a word) partially activates the
representations of similar-looking words, producing a more negative
N400. The N400 is sensitive to priming: in other words, its amplitude is
reduced when a target word is preceded by a word that is semantically,
morphologically, or orthographically related to it. In a sentence
context, an important determinant of N400 amplitude elicited by a word
is its cloze probability. Cloze probability is defined as the
probability of the target word completing that particular sentence
frame. Kutas and Hillyard (1984) found that a word's N400 amplitude has
a nearly inverse linear relationship with its cloze probability. That
is, as a word becomes less expected in context, its N400 amplitude is
increased relative to more expected words. Words that are incongruent
with a context (and thus have a cloze probability of 0) elicit large
N400 amplitudes as well (although the amplitude of the N400 for
incongruent words is also modulated by the cloze probability of the
congruent word that would have been expected in its place Relatedly, the
N400 amplitude elicited by open-class words (i.e. nouns, verbs,
adjectives, and adverbs) is reduced for words appearing later in a
sentence compared to earlier words. Taken together, these findings
suggest that when the prior context builds up meaning, it makes the
processing of upcoming words that fit with that context easier, reducing
the N400 amplitude they elicit. Factors that do not affect N400
amplitude While the N400 is larger to unexpected items at the end of a
sentence, its amplitude is generally unaffected by negation that causes
the last word to be untrue and thus anomalous. For example, in the
sentence A sparrow is a building, the N400 response to building is more
negative than the N400 response to bird in the sentence A sparrow is a
bird. In this case, building has a lower cloze probability, and so it is
less expected than bird. However, if negation is added to both sentences
in the form of the word not (i.e. A sparrow is not a building and A
sparrow is not a bird), the N400 amplitude to building will still be
more negative than that seen to bird. This suggests that the N400
responds to the relationship between words in context, but is not
necessarily sensitive to the sentence's truth value. More recent
research, however, has demonstrated that the N400 can sometimes be
modulated by quantifiers or adjectives that serve negation-like
purposes, or by pragmatically licensed negation. Additionally,
grammatical violations do not elicit a large N400 response. Rather,
these types of violations show a large positivity from about 500-1000 ms
after stimulus onset, known as the P600. Factors that affect N400
latency A striking feature of the N400 is the general invariance of its
peak latency. Although many different experimental manipulations affect
the amplitude of the N400, few factors (aging and disease states and
language proficiency being rare examples) alter the time it takes for
the N400 component to reach a peak amplitude.Federmeier, K. D. and
Laszlo, S. (2009). Time for meaning: Electrophysiology provides insights
into the dynamics of representation and processing in semantic memory.
In B. H. Ross (Ed.), Psychology of Learning and Motivation, Volume 51
(pp 1-44). Burlington: Academic Press. Sources Although localization of
the neural generators of an ERP signal is difficult due to the spreading
of current from the source to the sensors, multiple techniques can be
used to provide converging evidence about possible neural sources.Haan,
H., Streb, J., Bien, S., & Ro, F. (2000). Reconstructions of the
Semantic N400 Effect : Using a Generalized Minimum Norm Model with
Different Constraints ( L1 and L2 Norm ), 192, 178–192. Using methods
such as recordings directly off the surface of the brain or from
electrodes implanted in the brain, evidence from brain damaged patients,
and magnetoencephalographic (MEG) recordings (which measure magnetic
activity at the scalp associated with the electrical signal measured by
ERPs), the left temporal lobe has been highlighted as an important
source for the N400, with additional contributions from the right
temporal lobe. More generally, however, activity in a wide network of
brain areas is elicited in the N400 time window, suggesting a highly
distributed neural source. Theories There is still much debate as to
exactly what kind of neural and comprehension processes the N400
indexes. Some researchers believe that the underlying processes
reflected in the N400 occur after a stimulus has been recognized. For
example, Brown and Hagoort (1993) believe that the N400 occurs late in
the processing stream, and reflects the integration of a word's meaning
into the preceding context (see Kutas & Federmeier, 2011, for a
discussion). However, this account has not explained why items that
themselves have no meaning (e.g. pseudowords without defined
associations) also elicit the N400. Other researchers believe that the
N400 occurs much earlier, before words are recognized, and represents
neurolinguistics, orthographic or phonological analysis. More recent
accounts posit that the N400 represents a broader range of processes
indexing access to semantic memory. According to this account, it
represents the binding of information obtained from stimulus input with
representations from short- and long-term memory (such as recent
context, and accessing a word's meaning in long term memory) that work
together to create meaning from the information available in the current
context (Federmeier & Laszlo, 2009; see Kutas & Federmeier, 2011).
Another account is that the N400 reflects prediction error or surprisal.
Word-based surprisal was a strong predictor of N400 amplitude in an ERP
corpus. In addition, connectionist models make use of prediction error
for learning and linguistic adaptation, and these models can explain
several N400/P600 results in terms of prediction error propagation for
learning. It may also be that the N400 reflects a combination of these
or other factors. Nieuwland et al. (2019) argue that the N400 is
actually made up of two sub-components, with predictability affecting
the early part of the N400 (200-500 ms after stimulus onset) and
plausibility affecting it later (350-650 ms after stimulus onset). This
suggests that the N400 reflects both access to semantic memory (which is
sensitive to prediction), and semantic integration (sensitive to
plausibility). As research in the field of electrophysiology continues
to progress, these theories will likely be refined to include a complete
account of just what the N400 represents. See also
Bereitschaftspotential C1 and P1 Contingent negative variation
Difference due to memory Early left anterior negativity Error-related
negativity Late positive component Lateralized readiness potential
Mismatch negativity N2pc N100 N170 N200 P3a P3b P200 P300 (neuroscience)
P600 Somatosensory evoked potential Visual N1 References External links
Video of the N400 and P600 visualized as animated scalp topographies
Category:Electroencephalography Category:Evoked potentials
Category:Neurolinguistics
"""
    inferred_topics = run_inference_on_bert_topic_model(
        bert_topic_model, new_note_content
    )
    print(inferred_topics)
