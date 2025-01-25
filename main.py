"""
This code will add links to your notes based off the folder directories they are in
"""

from utils import *
from PARA_linking import link_folders_into_notes
from apply_topic_modeling_to_notes import get_topics_for_every_note


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_directory", type=str)

    args = parser.parse_args()
    root_directory = args.root_directory

    loop_through_notes(
        root_directory,
        [link_folders_into_notes, get_topics_for_every_note],
        clear_bottom_matter=True,
    )  #
