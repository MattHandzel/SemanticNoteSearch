from bert_topic_modeling import *
from utils import query_ollama

MODEL_NAME = "llama3.2"

system_message = """
Instructions:
1. Analyze the provided list of keywords to identify the main theme.
2. Generate a topic label of at most three words.
3. The label should be one high level topic encompassing the keywords. Ignore details.
4. There are noise in the keywords, focus on abstract concepts.
5. Present the output in the following format, delimited by triple backticks:
```
topic: <topic label>
```
"""


def load_bert_topic_model(path):
    # Load the BERT model
    model = BERTopic.load(path)
    return model


# Convert topics into a dictionary
topic_number_dictionary = {
    -1: "UNKNOWN",
}


# topic_model.get_document_info(docs) -> DataFrame
def convert_topics_to_labels(topics):
    if type(topics) != list and type(topics) != tuple and type(topics) != np.ndarray:
        return topic_number_dictionary[str(topics)]
    return [topic_number_dictionary[str(topic)] for topic in topics]


def get_topics_for_every_note(note_path, meta_data, content, root_directory):

    if detect_language(content) != "en":
        return None, None, False
    # topics = run_inference_on_bert_topic_model(bert_topic_model, content)
    # topics = convert_topics_to_labels(topics)

    bottom_matter = {}

    topic_distr, _ = bert_topic_model.approximate_distribution([content])
    topic_distr = topic_distr[0]  # only running on 1 document
    topics = np.where(topic_distr > 0.7)[0]

    for topic in topics:
        topic = convert_topics_to_labels(topic)
        if topic.lower() != "unknown":
            if "topics" not in bottom_matter:
                bottom_matter["topics"] = []
            bottom_matter["topics"].append(f"[[{topic}]]")
        print(f"Adding topic {topic} to bottom matter")

    return bottom_matter, None, False


def manually_review_topics(topics):
    for key, topic_word_distribution in topics.items():
        result = input(
            "Please give a high level topic label for the following keywords:\t"
            + ", ".join([z[0] for z in topic_word_distribution])
            + "\n"
        )
        topic_number_dictionary[str(key)] = result


def populate_topic_number_dictionary(raw_topics):
    manual_review_topics = {}
    raw_topics.pop(-1)
    for key, topic_word_distribution in raw_topics.items():
        print(f"Reviewing {key}", ", ".join([z[0] for z in topic_word_distribution]))

        try:
            result = (
                query_ollama(
                    f"""Input keywords:
[{', '.join([z[0] for z in topic_word_distribution])}]
    """,
                    system_message,
                    MODEL_NAME,
                )
                .strip()
                .lower()
                .split("```")[1]
                .strip()
                .split(":")[1]
                .strip()
            )
            print("RESULT", result)
            if len(result) > 50:
                raise Exception("Topic label too long")

            topic_number_dictionary[key] = result
        except Exception as e:
            print("expcetion", e)
            # print("Unable to label this topic, adding to manual review")
            manual_review_topics[key] = topic_word_distribution

    if len(manual_review_topics) > 0:
        manually_review_topics(manual_review_topics)


bert_topic_model = load_bert_topic_model(TOPIC_BERT_MODEL_PATH)
with open("topic_number_dictionary.json", "r") as f:
    topic_number_dictionary = json.load(f)

if __name__ == "__main__":
    dictionary = populate_topic_number_dictionary(bert_topic_model.get_topics())
    with open("topic_number_dictionary.json", "w") as f:
        json.dump(topic_number_dictionary, f)
