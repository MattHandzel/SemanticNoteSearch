import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from spellchecker import SpellChecker
from langdetect import detect
from bs4 import BeautifulSoup

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")


def remove_html_tags(text):
    clean_text = re.sub(r"<.*?>", "", text)
    return clean_text


def remove_numbers(text):
    clean_text = re.sub(r"\d+", "", text)
    return clean_text


def remove_special_characters(text):
    clean_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return clean_text


def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens


def convert_to_lowercase(text):
    lowercased_text = text.lower()
    return lowercased_text


def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens


def stem_text(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return stemmed_tokens


def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return lemmatized_tokens


def remove_duplicates(texts):
    unique_texts = list(set(texts))
    return unique_texts


def correct_spelling(text):
    spell = SpellChecker()
    tokens = word_tokenize(text)
    corrected_tokens = [spell.correction(word) for word in tokens]
    corrected_text = " ".join(corrected_tokens)
    return corrected_text


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


# other words to remove
words_to_remove = ["paraobsidian", "paralinking"]


def remove_words(text):
    for word in words_to_remove:
        text = text.replace(word, "")
    return text


def fix_encoding(text):
    try:
        decoded_text = text.encode("utf-8").decode("utf-8")
    except UnicodeDecodeError:
        decoded_text = "Encoding Error"
    return decoded_text


def remove_whitespace(text):
    cleaned_text = " ".join(text.split())
    return cleaned_text


def detect_language(text):
    try:
        language = detect(text)
    except:
        language = "unknown"
    return language


def clean_multilingual_text(text, language_code):
    nlp = spacy.load(language_code)
    doc = nlp(text)
    cleaned_text = " ".join([token.lemma_ for token in doc])
    return cleaned_text


def clean_html_with_beautifulsoup(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    cleaned_text = soup.get_text()
    return cleaned_text


def remove_markdown_formatting(text):
    # Remove wiki links
    text = re.sub(r"\[\[.*?\]\]", "", text)
    # Remove markdown links
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    # Remove markdown images
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Remove markdown headings
    text = re.sub(r"#+", "", text)
    # Remove markdown bold and italic
    text = re.sub(r"[*_]", "", text)
    return text


def cleanse_content(content):
    """
    Clean and preprocess content using your custom pipeline.
    (Same as in the original code)
    """
    return remove_words(
        remove_numbers(
            convert_to_lowercase(
                remove_whitespace(
                    remove_special_characters(
                        remove_html_tags(
                            (
                                clean_custom_patterns(
                                    ((remove_markdown_formatting(content)))
                                )
                            )
                        )
                    )
                )
            )
        )
    )


def tokenize_content(content):
    """
    Tokenize, remove stopwords, and lemmatize.
    (Same as in the original code)
    """
    tokens = tokenize_text(content)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_text(tokens)
    return tokens


def prepare_raw_content_for_bert_topic(content):
    """
    Cleanses and tokenizes content for TopicBERT.
    In BERTopic, you can either supply raw strings or tokenized strings.
    Here we return tokenized form joined as a single string.
    """
    tokens = tokenize_content(cleanse_content(content))
    return " ".join(tokens)
