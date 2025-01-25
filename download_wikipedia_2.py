import os
import requests
import bz2
import xml.etree.ElementTree as ET
import wikipediaapi
from tqdm import tqdm


# Download the latest Wikipedia dump
def download_wiki_dump():
    url = (
        "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
    )
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open("wikipedia_dump.xml.bz2", "wb") as file, tqdm(
        desc="Downloading Wikipedia dump",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)


# Parse the XML dump and convert to Markdown
def convert_to_markdown():
    wiki = wikipediaapi.Wikipedia("en")

    with bz2.BZ2File("wikipedia_dump.xml.bz2", "rb") as xml_file:
        context = ET.iterparse(xml_file, events=("end",))

        for event, elem in tqdm(context, desc="Converting pages"):
            if elem.tag.endswith("page"):
                title = elem.find(".//title").text
                page = wiki.page(title)

                if page.exists():
                    content = page.text
                    markdown_content = f"# {title}\n\n{content}"

                    # Save as Markdown file
                    with open(
                        f"markdown_wiki/{title.replace('/', '_')}.md",
                        "w",
                        encoding="utf-8",
                    ) as md_file:
                        md_file.write(markdown_content)

                elem.clear()


# Main execution
if __name__ == "__main__":
    print("Starting Wikipedia download and conversion process...")

    # Create output directory
    os.makedirs("markdown_wiki", exist_ok=True)

    # Download Wikipedia dump
    download_wiki_dump()

    # Convert to Markdown
    convert_to_markdown()

    print("Process completed. Markdown files are in the 'markdown_wiki' directory.")
