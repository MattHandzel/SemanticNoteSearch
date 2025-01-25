import os
import frontmatter
from frontmatter import Post
import logging
import argparse
import jison
import re

DEBUG = False
INT_MAX = int(2**63)
note_extensions = [".md"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

handler = logging.FileHandler("app.log")
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)

para_folders = ["projects", "areas", "resources", "archive", "zettelkasten"]
# banned_folders = []
banned_folders = ["sam-thomas-second-brain"]

project_name = "RELATED_NOTES_BOT"
bottom_matter_header = f"\n---\n\n**{project_name}**"


def get_all_notes(root_directory):
    for root, dirs, files in os.walk(root_directory, followlinks=True):
        for file in files:
            if any(file.endswith(ext) for ext in note_extensions):
                yield os.path.join(root, file)


def read_note(note_path):
    """
    Reads the note and returns the content
    """
    assert os.path.exists(note_path)
    assert os.path.isfile(note_path)
    assert note_path.endswith(".md")
    try:
        with open(note_path, "r") as f:
            file_content = f.read()
    except Exception as e:
        print(f"Error reading file {note_path}")
        return None
    return file_content


def parse_note(raw_content):
    """
    Pares the note to extract the meta data and content
    """

    # Extract the meta data
    meta_data, content = frontmatter.parse(raw_content)

    return meta_data, content


def remove_bottom_matter(content) -> str:
    if bottom_matter_header in content:
        content = content.split(bottom_matter_header)[0]
    return content


def update_bottom_matter(content, data):
    if content is None:
        return False
    content = remove_bottom_matter(content)
    content += bottom_matter_header
    content += "\n"
    for key, value in data.items():
        if type(value) == list:
            value = ", ".join(value)
        content += f"\n{key}: {value}"
    if len(data) > 0:
        content += "\n"
    return content


def loop_through_notes(
    root_directory, functions, max_notes=INT_MAX, clear_bottom_matter=False
):
    notes = get_all_notes(root_directory)
    num_notes_processed = 0
    for note in notes:
        if num_notes_processed >= max_notes:
            return

        parent_folders = note.split(root_directory)[1].split("/")[1:][:-1]
        if not any([folder in para_folders for folder in parent_folders]):
            continue
        if any([folder in banned_folders for folder in parent_folders]):
            continue

        # don't care about anything in archive
        if "archive" in parent_folders:
            continue

        raw_content = read_note(note)
        if len(raw_content) == 0:
            continue  # empyt note
        meta_data, content = parse_note(raw_content)

        bottom_matter_data = {}

        if clear_bottom_matter:
            content = remove_bottom_matter(content)

        for function in functions:
            new_bottom_matter_data, new_content, did_update_content = function(
                note, meta_data, content, root_directory
            )
            if new_bottom_matter_data:
                bottom_matter_data.update(new_bottom_matter_data)

            if did_update_content:
                content = new_content

        if bottom_matter_data and bottom_matter_data != {}:
            content = update_bottom_matter(content, bottom_matter_data)

        try:
            with open(note, "w") as f:
                if meta_data is {} or len(meta_data) == 0 or meta_data is None:
                    f.write(content)
                else:
                    f.write(frontmatter.dumps(Post(content, **meta_data)))

        except Exception as e:
            logger.error(f"Error writing to file {note}: {e}")
            # Replace with old content
            with open(note, "w") as f:
                f.write(raw_content)
        num_notes_processed += 1


def process_note(note, root_directory, functions, clear_bottom_matter=False):
    raw_content = read_note(note)

    if len(raw_content) == 0:
        return  # empyt note

    meta_data, content = parse_note(raw_content)

    bottom_matter_data = {}

    if clear_bottom_matter:
        content = remove_bottom_matter(content)

    content_was_updated = False
    for function in functions:
        new_bottom_matter_data, new_content, did_update_content = function(
            note, meta_data, content, root_directory
        )
        if new_bottom_matter_data:
            bottom_matter_data.update(new_bottom_matter_data)

        if did_update_content:
            content_was_updated = True
            content = new_content

    if bottom_matter_data and bottom_matter_data != {}:
        content = update_bottom_matter(content, bottom_matter_data)

    try:
        if not content_was_updated and bottom_matter_data == {}:
            return
        with open(note, "w") as f:
            if meta_data is {} or len(meta_data) == 0 or meta_data is None:
                f.write(content)
            else:
                f.write(frontmatter.dumps(Post(content, **meta_data)))

    except Exception as e:
        logger.error(f"Error writing to file {note}: {e}")
        # Replace with old content
        with open(note, "w") as f:
            f.write(raw_content)


def loop_through_directories(
    root_directory, functions, max_notes=INT_MAX, clear_bottom_matter=False
):
    notes = get_all_notes(root_directory)
    num_notes_processed = 0
    for note in notes:
        print("Working on note: ", note)
        if num_notes_processed >= max_notes:
            return

        process_note(note, root_directory, functions, clear_bottom_matter)
        num_notes_processed += 1


def clean_custom_patterns(text):
    # TODO: Thoroughly test these
    # Replace: email, phone, youtube link, regular link  with [email], [phone], [youtube], [link]
    clean_text = re.sub(  # email
        r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", "<EMAIL>", text
    )
    clean_text = re.sub(  # phone
        r"(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})",
        "<PHONE>",
        clean_text,
    )
    # clean_text = re.sub(  # youtube link
    #     r"(https?:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+", "", clean_text
    # )
    clean_text = re.sub(  # regular link
        r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)",
        "<URL>",
        clean_text,
    )

    return clean_text


def normalize_whitespace(text):
    lines = text.split("\n")
    normalized_lines = [" ".join(line.split()) for line in lines]
    return "\n".join(normalized_lines)


def clean_text_for_embedding_model(text):

    return clean_custom_patterns(normalize_whitespace(text))
