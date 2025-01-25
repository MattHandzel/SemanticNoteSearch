#!/usr/bin/env python3
# Made by chatgpt
import os
import sys
import logging
import argparse
import requests
import bz2
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator
import subprocess

try:
    import mwparserfromhell  # For parsing the Wikitext
except ImportError:
    print(
        "Please install mwparserfromhell (pip install mwparserfromhell).",
        file=sys.stderr,
    )
    sys.exit(1)

################################################################################
# Logging Configuration
################################################################################

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("wiki_dump_to_md")

################################################################################
# Utility: Download Wikipedia dump
################################################################################


def download_dump(dump_url: str, output_path: str) -> None:
    """
    Downloads the specified Wikipedia dump URL to output_path.
    Uses a streaming HTTP GET request.
    """
    logger.info(f"Starting download from {dump_url} ...")
    with requests.get(dump_url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    logger.info(f"Download complete. Saved to {output_path}")


################################################################################
# Utility: Stream through the compressed dump
################################################################################


def iterate_xml_pages(bz2_path: str) -> Iterator[str]:
    """
    Generator function that yields <page>...</page> blocks from a
    compressed .bz2 XML dump. Uses a streaming approach.
    """
    logger.info(f"Opening XML dump: {bz2_path}")
    page_buffer = []
    in_page = False

    with bz2.open(bz2_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            # detect start of <page>
            if "<page>" in line:
                in_page = True
                page_buffer = [line]
            elif "</page>" in line:
                # end of page
                page_buffer.append(line)
                yield "".join(page_buffer)
                in_page = False
                page_buffer = []
            else:
                if in_page:
                    page_buffer.append(line)


################################################################################
# Markdown Converter (uses mwparserfromhell + optional pandoc)
################################################################################


def convert_wikitext_to_md(wikitext: str) -> str:
    """
    Convert wikitext to Markdown.
    This uses mwparserfromhell for basic cleanup and pandoc for final.

    If you don't have pandoc installed, you can rely solely on mwparserfromhell
    transformations or other pure-Python approaches like 'bye-wiki' if you prefer.
    """


################################################################################
# Extract & Convert a Single Page
################################################################################


def process_page(page_xml: str, output_dir: str) -> None:
    """
    Parses a <page>...</page> block of XML, extracts the wikitext,
    converts to markdown, and saves to a file named after the <title>.
    """
    # We'll do naive stubbing: find <title>, <text>, ignoring complexities.
    # A robust approach: parse with 'xml.etree.ElementTree'. For brevity, do minimal.
    title_start = page_xml.find("<title>")
    title_end = page_xml.find("</title>", title_start)

    text_start = page_xml.find("<text", title_end)
    text_tag_end = page_xml.find(">", text_start)
    text_close = page_xml.find("</text>", text_tag_end)

    if title_start == -1 or title_end == -1 or text_start == -1 or text_close == -1:
        return  # skip if parsing fails

    title = page_xml[title_start + 7 : title_end].strip()
    # The wikitext can have markup or even be empty
    wikitext = page_xml[text_tag_end + 1 : text_close].strip()

    # Convert the page title into a filesystem-friendly name:
    # e.g. remove slashes, question marks, etc.
    safe_title = "".join(c for c in title if c.isalnum() or c in " -_.").rstrip()

    if not safe_title:
        safe_title = "untitled_page"

    # Convert wikitext to Markdown
    markdown = convert_wikitext_to_md(wikitext)

    # Save
    file_path = Path(output_dir) / f"{safe_title}.md"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(markdown)

    logger.debug(f"Saved article '{title}' -> {file_path}")


def init_multiprocessing(args_):
    """
    Helper to ensure each process has correct global references if needed.
    """
    global global_args
    global_args = args_


def process_page_wrapper(page_xml: str):
    """
    Multiprocessing wrapper that uses the global args to call process_page.
    """
    process_page(page_xml, global_args["output_dir"])


################################################################################
# Main Script
################################################################################


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert Wikipedia Dump to Markdown."
    )
    parser.add_argument(
        "--dump-url",
        default="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
        help="URL of the Wikipedia XML dump (.bz2). Default is the latest EN Wikipedia.",
    )
    parser.add_argument(
        "--output-dir",
        default="wiki_md_output",
        help="Directory to store the Markdown files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 1),
        help="Number of parallel worker processes.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dump_filename = os.path.basename(args.dump_url)
    local_dump_path = os.path.join(args.output_dir, dump_filename)

    # 1. Download the dump
    if not os.path.exists(local_dump_path):
        logger.info("Dump file not found locally. Downloading...")
        download_dump(args.dump_url, local_dump_path)
    else:
        logger.info("Wikipedia dump file already exists. Skipping download.")

    # 2. Iterate through the dump, parse pages, convert to MD
    logger.info("Beginning extraction and conversion to Markdown...")

    # Prepare multiprocessing
    pool_args = {"output_dir": args.output_dir}

    with Pool(
        processes=args.workers, initializer=init_multiprocessing, initargs=(pool_args,)
    ) as pool:
        # Use imap or similar to feed pages
        page_generator = iterate_xml_pages(local_dump_path)
        # We map each page_xml to process_page_wrapper
        for _ in pool.imap_unordered(
            process_page_wrapper, page_generator, chunksize=50
        ):
            pass

    logger.info(
        "All pages processed. Markdown files are stored in: %s", args.output_dir
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
