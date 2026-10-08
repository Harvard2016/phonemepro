"""Build the accent pronunciation lexicon shared by the API and the web app.

    python -m scripts.build_lexicon

Sources:
  General American   CMUdict (via NLTK), first listed pronunciation
  British (RP)       Britfone 3.0.1 (MIT), IPA mapped onto the model's ARPABET symbols
  Australian         derived from the British entry by two documented rules (see to_australian)

Only words present in both source dictionaries are kept, so every entry can be
practised in every accent. Output: web/public/lexicon/lexicon.json
"""
import json
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BRITFONE_PATH = ROOT_DIR / "data" / "lexicon" / "britfone.csv"
BRITFONE_URL = "https://raw.githubusercontent.com/JoseLlarena/Britfone/master/britfone.main.3.0.1.csv"
OUTPUT_PATH = ROOT_DIR / "web" / "public" / "lexicon" / "lexicon.json"

# Britfone IPA -> ARPABET. The model's symbol set is General American, so a few
# British vowels have no symbol of their own:
#   ɒ (LOT) shares AA with ɑː (PALM/BATH/START); the model cannot tell them apart.
#   ɜː (NURSE) is written ER, the same symbol as the American r-coloured vowel.
#   The centring diphthongs ɪə, ɛə, ʊə become a vowel plus schwa.
IPA_VOWELS = {
    "ə": ["AH"], "ɐ": ["AH"], "ɪ": ["IH"], "ɛ": ["EH"], "æ": ["AE"], "ɒ": ["AA"], "ʊ": ["UH"],
    "i": ["IY"], "iː": ["IY"], "uː": ["UW"], "ɔː": ["AO"], "ɑː": ["AA"], "ɜː": ["ER"],
    "eɪ": ["EY"], "aɪ": ["AY"], "əʊ": ["OW"], "aʊ": ["AW"], "ɔɪ": ["OY"],
    "ɪə": ["IH", "AH"], "ɛə": ["EH", "AH"], "ʊə": ["UH", "AH"],
}
IPA_CONSONANTS = {
    "p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "g": "G", "f": "F", "v": "V", "s": "S", "z": "Z",
    "h": "HH", "m": "M", "n": "N", "l": "L", "w": "W", "ɹ": "R", "j": "Y", "ŋ": "NG", "ʃ": "SH",
    "ʒ": "ZH", "θ": "TH", "ð": "DH", "tʃ": "CH", "dʒ": "JH",
}
STRESS = {"ˈ": "1", "ˌ": "2"}

# Before these sounds Australian English keeps a short unstressed /ɪ/ rather than schwa.
KEEP_SHORT_I_BEFORE = {"K", "G", "NG", "SH", "CH", "JH", "ZH"}
NASALS = {"N", "M"}
# BATH words that keep the long vowel in Australian English even before a nasal.
AUSTRALIAN_LONG_A = {"can't", "aunt", "aunts", "shan't"}


def britfone_to_arpabet(ipa: str) -> list[str]:
    phonemes = []
    for token in ipa.split():
        stress = STRESS.get(token[0])
        symbol = token[1:] if stress else token
        if symbol in IPA_CONSONANTS:
            phonemes.append(IPA_CONSONANTS[symbol])
            continue
        first, *rest = IPA_VOWELS[symbol]
        phonemes.append(first + (stress or "0"))
        phonemes.extend(vowel + "0" for vowel in rest)
    return phonemes


def to_australian(british: list[str], american: list[str], word: str = "") -> list[str]:
    """Approximate General Australian from the British entry.

    1. BATH words keep the short TRAP vowel before a nasal (dance, chance, plant),
       where southern British English has the long vowel. BATH words are found by
       comparing with the American entry, which has AE in the same word. A few
       words (can't, aunt) are exceptions and keep the long vowel.
    2. Weak vowel merger: unstressed short i becomes schwa (rabbit, boxes) except
       before velar and postalveolar consonants or another vowel.
    """
    american_has_trap = any(p.startswith("AE") for p in american) and word not in AUSTRALIAN_LONG_A
    result = []
    for index, phoneme in enumerate(british):
        following = british[index + 1] if index + 1 < len(british) else None
        if phoneme == "AA1" and following in NASALS and american_has_trap and index + 2 < len(british):
            result.append("AE1")
        elif (
            phoneme == "IH0" and following is not None
            and following not in KEEP_SHORT_I_BEFORE and following[-1] not in "012"
        ):
            result.append("AH0")
        else:
            result.append(phoneme)
    return result


def load_britfone() -> dict[str, list[str]]:
    if not BRITFONE_PATH.exists():
        BRITFONE_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BRITFONE_URL, BRITFONE_PATH)

    entries = {}
    for line in BRITFONE_PATH.read_text(encoding="utf-8").splitlines():
        word, ipa = line.split(", ", 1)
        word = word.split("(")[0].lower()
        # Keep the first listed variant, and skip clitics such as 'LL.
        if word not in entries and word[0].isalpha():
            entries[word] = britfone_to_arpabet(ipa)
    return entries


def main():
    from nltk.corpus import cmudict

    american = {word: pronunciations[0] for word, pronunciations in cmudict.dict().items()}
    british = load_britfone()

    words = {}
    for word in sorted(set(american) & set(british)):
        ga, rp = american[word], british[word]
        au = to_australian(rp, ga, word)
        entry = {"ga": " ".join(ga), "rp": " ".join(rp)}
        if au != rp:
            entry["au"] = " ".join(au)
        words[word] = entry

    lexicon = {
        "accents": {
            "ga": {"name": "American", "detail": "General American", "source": "CMUdict"},
            "rp": {"name": "British", "detail": "Received Pronunciation", "source": "Britfone 3.0.1"},
            "au": {"name": "Australian", "detail": "General Australian", "source": "derived from Britfone by rule"},
        },
        "words": words,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(lexicon, ensure_ascii=False, separators=(",", ":")))

    differing = sum(1 for entry in words.values() if entry["ga"] != entry["rp"])
    print(f"{len(words)} words, {differing} differ between American and British, "
          f"{sum('au' in e for e in words.values())} have a distinct Australian form")
    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
