"""Shared filesystem locations for the Streamlit app and research notebook."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_ROOT / "assets" / "images"
PREPARED_DATA = PROJECT_ROOT / "data" / "processed" / "spam_ham.csv"
ENGLISH_STOPWORDS = PROJECT_ROOT / "data" / "stopwords" / "english.txt"


def raw_corpus_paths():
    """Return spam and ham directories independently of the working directory."""
    configured = os.environ.get("SPAM_DETECTOR_RAW_DATA")
    base = Path(configured).expanduser() if configured else PROJECT_ROOT / "data" / "raw"
    if not base.is_absolute():
        raise ValueError("SPAM_DETECTOR_RAW_DATA must be an absolute directory path.")
    spam, ham = base / "spam", base / "ham"
    if not spam.is_dir() or not ham.is_dir():
        raise FileNotFoundError(
            "Raw email corpus missing: populate data/raw/spam and data/raw/ham, "
            "or set SPAM_DETECTOR_RAW_DATA to a directory containing spam/ and ham/. "
            "The prepared CSV is for the article and cannot replace the raw corpus."
        )
    return spam, ham
