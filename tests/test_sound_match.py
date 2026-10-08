"""Tolerant matching of heard sounds to target sounds."""
import json
from pathlib import Path

from pronunciation.sound_match import GROUPS_PATH, same_class, sounds_attempted

CASES = json.loads((Path(__file__).parent / "fixtures" / "sound_match_cases.json").read_text())


def test_python_rule_matches_shared_fixtures():
    # web/src/engine/soundMatch.test.js asserts the same cases against the JavaScript port.
    assert len(CASES) > 80
    for case in CASES:
        assert sounds_attempted(case["target"], case["heard"]) == case["expected"], case["note"]


def test_accent_swaps_count_and_wrong_words_do_not():
    by_note = {case["note"]: case["expected"] for case in CASES}
    assert by_note["think as tink"] == by_note["this as dis"] == by_note["non-rhotic: vowel for vowel"] == 100
    assert by_note["wrong word: fish for water"] == 0
    # Vowel-for-vowel is loose: an unrelated word still lines its vowels up.
    assert by_note["wrong word: banana for water"] == 50


def test_classes():
    assert same_class("AO1", "AH0") and same_class("TH", "T") and same_class("T", "TH") and same_class("V", "W")
    assert not same_class("AE1", "T") and not same_class("K", "M") and not same_class("P", "T")


def test_groups_file_uses_known_sounds():
    data = json.loads(GROUPS_PATH.read_text())
    vocab = {p.rstrip("012") for p in json.loads(
        (GROUPS_PATH.parents[2] / "public" / "model" / "model.json").read_text())["vocab"]}
    assert set(data["vowels"]) <= vocab
    assert all(len(group) >= 2 and set(group) <= vocab - set(data["vowels"]) for group in data["groups"])
