"""Practice words by difficulty, with targets for the chosen accent.

The word bank and the accent lexicon are static JSON shared with the web app
(web/public/lexicon/). Words were picked to have human reference recordings and
to include many that differ between accents.
"""
import json
import random
from pathlib import Path

from pronunciation.lexicon import DEFAULT_ACCENT, word_phonemes

WORDS_PATH = Path(__file__).resolve().parent / "web" / "public" / "lexicon" / "practice_words.json"
LEVELS = json.loads(WORDS_PATH.read_text())
DIFFICULTY_LEVELS = tuple(LEVELS)


def get_all_words(difficulty=None, accent=DEFAULT_ACCENT):
    levels = [difficulty] if difficulty in LEVELS else DIFFICULTY_LEVELS
    return [
        {"word": word, "phonemes": word_phonemes(word, accent), "difficulty": level}
        for level in levels
        for word in LEVELS[level]
    ]


def get_random_word(difficulty=None, accent=DEFAULT_ACCENT):
    return random.choice(get_all_words(difficulty, accent))


def find_word(word: str, accent=DEFAULT_ACCENT):
    """The practice entry for a word, or None if it is not in the word bank."""
    return next((entry for entry in get_all_words(accent=accent) if entry["word"] == word.lower()), None)
