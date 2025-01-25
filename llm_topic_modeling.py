from utils import query_ollama
from utils import *
from PARA_linking import *


MODEL_NAME = "llama3.2:1b"
system_message = """Analyze the following note and respond with concepts related to the note.
1. Cover all concepts mentioned in the note.
2. The concepts should be abstract, broad, or extract an idea prevalent in the note. Concepts should not be narrow or specific to the note.
3. Keep the concepts simple, the concepts should be a single entity. 
4. De not respond with a phrase, or sentence. The concept should be extremely simple and straightforward.
5. Do not respond with any reasoning. Make your response as concise as possible.
6. If there are no topics then respond with "no topics".
7. Output the topics and ideas in a bulleted, unordered list, separated by a new line, and in wiki-link format.
Example output:
```
* [[concept 1]]
* [[concept 2]]
* [[concept 3]]
```
"""


def get_topics_for_every_note(note_path, meta_data, content, root_directory):

    print("Querying OLLAMA")
    print(f"{meta_data}")
    note_title = (
        ("\t" + meta_data["title"])
        if "title" in meta_data
        else ("\t" + str(meta_data["id"]) if "id" in meta_data else "")
    )
    result = (
        query_ollama("NOTE:{note_title}\n```\n" + content + "\n```", system_message)
        .replace("* ", "")
        .replace("*", "")
        .strip()
    )
    for line in result.split("\n"):
        if "[[" not in line:
            return None, None, False  # it broke somewhere
    print("RESULT")
    print(result)
    print()

    return {f"topics_from_{MODEL_NAME}": result}, None, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_directory", type=str)
    args = parser.parse_args()
    root_directory = args.root_directory

    loop_through_notes(
        root_directory,
        [link_folders_into_notes],
        clear_bottom_matter=True,
    )  #
    loop_through_notes(
        root_directory,
        [get_topics_for_every_note],
        clear_bottom_matter=False,
    )  #
