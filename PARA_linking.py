"""
This code will add links to your notes based off the folder directories they are in
"""

from utils import *


def link_folders_into_notes(note_path, meta_data, content, root_directory):

    parent_folders = note_path.split(root_directory)[1].split("/")[1:][:-1]
    if not any([folder in para_folders for folder in parent_folders]):
        return None, None, False

    # don't care about anything in archive
    if "archive" in parent_folders:
        return None, None, False

    if content is None:
        return None, None, False

    parent_folders = parent_folders[1:]

    # Add links to the note
    has_been_modified = False

    bottom_matter_data = {}

    for folder_name in parent_folders:
        if f"[[{folder_name}]]" not in content:
            if "PARA-linking" not in bottom_matter_data:
                bottom_matter_data["PARA-linking"] = []
            bottom_matter_data["PARA-linking"].append(f"[[{folder_name}]]")

    # check all the links in the note, wiki links should be in the format of "word-word-word"

    return bottom_matter_data, None, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_directory", type=str)

    args = parser.parse_args()
    root_directory = args.root_directory

    link_folders_into_notes(root_directory)
