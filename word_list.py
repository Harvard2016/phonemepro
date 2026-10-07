"""
Curated word list for pronunciation practice.
The list emphasizes common American-English vowel, consonant, rhotic,
and stress contrasts instead of just word length.
"""

import random

DIFFICULTY_LEVELS = ("easy", "medium", "hard")

PRACTICE_WORDS = [
    {"word": "ship", "phonemes": ["SH", "IH1", "P"], "difficulty": "easy"},
    {"word": "sheep", "phonemes": ["SH", "IY1", "P"], "difficulty": "easy"},
    {"word": "full", "phonemes": ["F", "UH1", "L"], "difficulty": "easy"},
    {"word": "fool", "phonemes": ["F", "UW1", "L"], "difficulty": "easy"},
    {"word": "west", "phonemes": ["W", "EH1", "S", "T"], "difficulty": "easy"},
    {"word": "vest", "phonemes": ["V", "EH1", "S", "T"], "difficulty": "easy"},
    {"word": "thin", "phonemes": ["TH", "IH1", "N"], "difficulty": "easy"},
    {"word": "then", "phonemes": ["DH", "EH1", "N"], "difficulty": "easy"},
    {"word": "bat", "phonemes": ["B", "AE1", "T"], "difficulty": "easy"},
    {"word": "bet", "phonemes": ["B", "EH1", "T"], "difficulty": "easy"},
    {"word": "bird", "phonemes": ["B", "ER1", "D"], "difficulty": "easy"},
    {"word": "word", "phonemes": ["W", "ER1", "D"], "difficulty": "easy"},
    {"word": "rice", "phonemes": ["R", "AY1", "S"], "difficulty": "easy"},
    {"word": "lice", "phonemes": ["L", "AY1", "S"], "difficulty": "easy"},

    {"word": "water", "phonemes": ["W", "AO1", "T", "ER0"], "difficulty": "medium"},
    {"word": "father", "phonemes": ["F", "AA1", "DH", "ER0"], "difficulty": "medium"},
    {"word": "mother", "phonemes": ["M", "AH1", "DH", "ER0"], "difficulty": "medium"},
    {"word": "weather", "phonemes": ["W", "EH1", "DH", "ER0"], "difficulty": "medium"},
    {"word": "measure", "phonemes": ["M", "EH1", "ZH", "ER0"], "difficulty": "medium"},
    {"word": "vision", "phonemes": ["V", "IH1", "ZH", "AH0", "N"], "difficulty": "medium"},
    {"word": "very", "phonemes": ["V", "EH1", "R", "IY0"], "difficulty": "medium"},
    {"word": "world", "phonemes": ["W", "ER1", "L", "D"], "difficulty": "medium"},
    {"word": "button", "phonemes": ["B", "AH1", "T", "AH0", "N"], "difficulty": "medium"},
    {"word": "little", "phonemes": ["L", "IH1", "T", "AH0", "L"], "difficulty": "medium"},
    {"word": "teacher", "phonemes": ["T", "IY1", "CH", "ER0"], "difficulty": "medium"},
    {"word": "student", "phonemes": ["S", "T", "UW1", "D", "AH0", "N", "T"], "difficulty": "medium"},
    {"word": "music", "phonemes": ["M", "Y", "UW1", "Z", "IH0", "K"], "difficulty": "medium"},
    {"word": "chocolate", "phonemes": ["CH", "AO1", "K", "L", "AH0", "T"], "difficulty": "medium"},

    {"word": "mirror", "phonemes": ["M", "IH1", "R", "ER0"], "difficulty": "hard"},
    {"word": "squirrel", "phonemes": ["S", "K", "W", "ER1", "AH0", "L"], "difficulty": "hard"},
    {"word": "rural", "phonemes": ["R", "UH1", "R", "AH0", "L"], "difficulty": "hard"},
    {"word": "comfortable", "phonemes": ["K", "AH1", "M", "F", "T", "ER0", "B", "AH0", "L"], "difficulty": "hard"},
    {"word": "temperature", "phonemes": ["T", "EH1", "M", "P", "ER0", "AH0", "CH", "ER0"], "difficulty": "hard"},
    {"word": "opportunity", "phonemes": ["AA2", "P", "ER0", "T", "UW1", "N", "AH0", "T", "IY0"], "difficulty": "hard"},
    {"word": "refrigerator", "phonemes": ["R", "IH0", "F", "R", "IH1", "JH", "ER0", "EY2", "T", "ER0"], "difficulty": "hard"},
    {"word": "schedule", "phonemes": ["S", "K", "EH1", "JH", "UW0", "L"], "difficulty": "hard"},
    {"word": "particularly", "phonemes": ["P", "ER0", "T", "IH1", "K", "Y", "AH0", "L", "ER0", "L", "IY0"], "difficulty": "hard"},
    {"word": "theater", "phonemes": ["TH", "IY1", "AH0", "T", "ER0"], "difficulty": "hard"},
    {"word": "jewelry", "phonemes": ["JH", "UW1", "AH0", "L", "R", "IY0"], "difficulty": "hard"},
    {"word": "camera", "phonemes": ["K", "AE1", "M", "ER0", "AH0"], "difficulty": "hard"},
]


def get_words_by_difficulty(difficulty=None):
    if difficulty in DIFFICULTY_LEVELS:
        return [word for word in PRACTICE_WORDS if word["difficulty"] == difficulty]
    return PRACTICE_WORDS


def get_random_word(difficulty=None):
    return random.choice(get_words_by_difficulty(difficulty))


def get_all_words(difficulty=None):
    return get_words_by_difficulty(difficulty)
