import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalize extraction artifacts while retaining punctuation and paragraphs."""
    text = unicodedata.normalize("NFKC", text or "")
    text = "".join(character for character in text if character in "\n\t" or unicodedata.category(character) != "Cc")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
