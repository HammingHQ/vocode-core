from loguru import logger


def ensure_punkt_installed():
    try:
        from nltk.data import find, download

        find("tokenizers/punkt")

        download("punkt")
        download("punkt_tab")
    except LookupError:
        from nltk import download

        # If not installed, download 'punkt'
        logger.info("Downloading 'punkt' tokenizer...")
        download("punkt")
        logger.info("'punkt' tokenizer downloaded successfully.")
