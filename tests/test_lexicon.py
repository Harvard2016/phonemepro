import pytest

from pronunciation.lexicon import UnknownWord, accents, normalize_text, text_phonemes, word_phonemes
from scripts.build_lexicon import britfone_to_arpabet, to_australian


def test_accents_available():
    assert set(accents()) == {"ga", "rp", "au"}


def test_rhoticity_differs_between_american_and_british():
    assert word_phonemes("water", "ga") == ["W", "AO1", "T", "ER0"]
    assert word_phonemes("water", "rp") == ["W", "AO1", "T", "AH0"]
    assert word_phonemes("car", "ga") == ["K", "AA1", "R"]
    assert word_phonemes("car", "rp") == ["K", "AA1"]


def test_bath_vowel_and_yod():
    assert word_phonemes("bath", "ga") == ["B", "AE1", "TH"]
    assert word_phonemes("bath", "rp") == ["B", "AA1", "TH"]
    assert word_phonemes("new", "ga") == ["N", "UW1"]
    assert word_phonemes("new", "rp") == ["N", "Y", "UW1"]


def test_australian_follows_british_except_for_its_own_rules():
    assert word_phonemes("water", "au") == word_phonemes("water", "rp")
    assert word_phonemes("bath", "au") == ["B", "AA1", "TH"]
    assert word_phonemes("dance", "au") == ["D", "AE1", "N", "S"]
    assert word_phonemes("rabbit", "au") == ["R", "AE1", "B", "AH0", "T"]
    assert word_phonemes("music", "au") == ["M", "Y", "UW1", "Z", "IH0", "K"]


def test_default_accent_is_american():
    assert word_phonemes("water") == word_phonemes("water", "ga")


def test_text_phonemes_joins_words_and_ignores_punctuation():
    assert normalize_text("Red lorry, 42 times!") == ["red", "lorry", "times"]
    assert text_phonemes("The car.", "rp") == word_phonemes("the", "rp") + ["K", "AA1"]


def test_unknown_word_and_accent():
    with pytest.raises(UnknownWord) as error:
        text_phonemes("the zzyzx", "ga")
    assert error.value.args[0] == "zzyzx"
    with pytest.raises(ValueError):
        word_phonemes("water", "cockney")


def test_britfone_mapping_handles_stress_and_centring_diphthongs():
    assert britfone_to_arpabet("w ˈɔː t ə") == ["W", "AO1", "T", "AH0"]
    assert britfone_to_arpabet("n ˈɪə") == ["N", "IH1", "AH0"]
    assert britfone_to_arpabet("t ə m ˈɑː t ˌəʊ") == ["T", "AH0", "M", "AA1", "T", "OW2"]
    assert britfone_to_arpabet("ʃ ˈɛ dʒ uː l") == ["SH", "EH1", "JH", "UW0", "L"]


def test_australian_rules_in_isolation():
    # BATH word before a nasal keeps the short vowel; before a fricative it does not.
    assert to_australian(["P", "L", "AA1", "N", "T"], ["P", "L", "AE1", "N", "T"]) == ["P", "L", "AE1", "N", "T"]
    assert to_australian(["P", "AA1", "TH"], ["P", "AE1", "TH"]) == ["P", "AA1", "TH"]
    # START words are not BATH words: American has no AE there.
    assert to_australian(["AA1", "N", "T"], ["AA1", "R", "N", "T"]) == ["AA1", "N", "T"]
    # Weak vowel merger, blocked before velars and at the end of a word.
    assert to_australian(["B", "AA1", "K", "S", "IH0", "Z"], []) == ["B", "AA1", "K", "S", "AH0", "Z"]
    assert to_australian(["M", "Y", "UW1", "Z", "IH0", "K"], []) == ["M", "Y", "UW1", "Z", "IH0", "K"]
    assert to_australian(["M", "IH1", "N", "IH0", "AH0", "M"], []) == ["M", "IH1", "N", "IH0", "AH0", "M"]
    # "can't" keeps the long vowel although it is a BATH word before a nasal.
    assert to_australian(["K", "AA1", "N", "T"], ["K", "AE1", "N", "T"], "can't") == ["K", "AA1", "N", "T"]
    assert word_phonemes("can't", "au") == ["K", "AA1", "N", "T"]
