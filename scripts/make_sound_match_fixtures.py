"""Write tolerant-match cases that the Python and JavaScript implementations must both reproduce.

    python -m scripts.make_sound_match_fixtures

Output: tests/fixtures/sound_match_cases.json
"""
import json
import random
from pathlib import Path

from pronunciation.sound_match import sounds_attempted

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sound_match_cases.json"
PHONEMES = [
    "K", "AE1", "T", "S", "IH0", "N", "ZH", "ER0", "AH0", "UW1", "R", "TH", "OY1", "AA1", "L", "IY0", "D", "W",
    "DH", "Z", "F", "V", "SH", "CH", "JH", "NG", "M", "P", "B", "G", "HH", "Y",
]

# (note, target, heard)
HAND_WRITTEN = [
    ("exact", ["W", "AO1", "T", "ER0"], ["W", "AO1", "T", "ER0"]),
    ("stress ignored", ["W", "AO1", "T", "ER0"], ["W", "AO0", "T", "ER1"]),
    ("non-rhotic: vowel for vowel", ["W", "AO1", "T", "ER0"], ["W", "AO1", "T", "AH0"]),
    ("flapped t heard as d", ["W", "AO1", "T", "ER0"], ["W", "AA1", "D", "ER0"]),
    ("think as tink", ["TH", "IH1", "NG", "K"], ["T", "IH1", "NG", "K"]),
    ("think as sink", ["TH", "IH1", "NG", "K"], ["S", "IY1", "N", "K"]),
    ("think as fink", ["TH", "IH1", "NG", "K"], ["F", "IH1", "NG", "K"]),
    ("this as dis", ["DH", "IH1", "S"], ["D", "IY1", "S"]),
    ("this as zis", ["DH", "IH1", "S"], ["Z", "IH1", "Z"]),
    ("very as wery", ["V", "EH1", "R", "IY0"], ["W", "EH1", "R", "IY0"]),
    ("right as light", ["R", "AY1", "T"], ["L", "AY1", "T"]),
    ("ship as sip", ["SH", "IH1", "P"], ["S", "IY1", "B"]),
    ("chair as share", ["CH", "EH1", "R"], ["SH", "EH1", "R"]),
    ("sing as sin", ["S", "IH1", "NG"], ["S", "IH1", "N"]),
    ("dog as dok", ["D", "AO1", "G"], ["D", "AO1", "K"]),
    ("dropped final sound", ["W", "AO1", "T", "ER0"], ["W", "AH1", "T"]),
    ("extra sounds do not count against", ["W", "AO1", "T", "ER0"], ["AH0", "W", "AO1", "T", "ER0", "F"]),
    ("wrong word: banana for water", ["W", "AO1", "T", "ER0"], ["B", "AH0", "N", "AE1", "N", "AH0"]),
    ("wrong word: fish for water", ["W", "AO1", "T", "ER0"], ["F", "IH1", "SH"]),
    ("wrong word: hello for water", ["W", "AO1", "T", "ER0"], ["HH", "AH0", "L", "OW1"]),
    ("wrong word: butter for water", ["W", "AO1", "T", "ER0"], ["B", "AH1", "T", "ER0"]),
    ("consonant for vowel", ["K", "AE1", "T"], ["K", "T", "T"]),
    ("unrelated consonants", ["K", "AE1", "T"], ["M", "AE1", "L"]),
    ("the model cannot hear ZH", ["V", "IH1", "ZH", "AH0", "N"], ["V", "IH1", "JH", "AH0", "N"]),
    ("the model cannot hear ZH, dropped", ["V", "IH1", "ZH", "AH0", "N"], ["V", "IH1", "AH0", "N"]),
    ("nothing heard", ["K", "AE1", "T"], []),
    ("no target", [], ["K"]),
    ("groups are not chained: P is not T", ["P", "AE1", "T"], ["T", "AE1", "P"]),
    ("a group works both ways", ["T", "IH1", "N"], ["TH", "IH1", "N"]),
    ("rounding of a third", ["K", "AE1", "T"], ["K", "L", "L"]),
    ("rounding of two thirds", ["K", "AE1", "T"], ["K", "AE1", "L"]),
]


def main():
    rng = random.Random(4)
    cases = [(note, target, heard) for note, target, heard in HAND_WRITTEN]
    for index in range(60):
        target = [rng.choice(PHONEMES) for _ in range(rng.randint(1, 8))]
        heard = [rng.choice(PHONEMES) if rng.random() < 0.5 else p for p in target]
        for _ in range(rng.randint(0, 2)):
            if heard and rng.random() < 0.5:
                heard.pop(rng.randrange(len(heard)))
            else:
                heard.insert(rng.randint(0, len(heard)), rng.choice(PHONEMES))
        cases.append((f"random {index}", target, heard))

    fixtures = [
        {"note": note, "target": target, "heard": heard, "expected": sounds_attempted(target, heard)}
        for note, target, heard in cases
    ]
    OUTPUT_PATH.write_text(json.dumps(fixtures, indent=1) + "\n")
    for case in fixtures[:len(HAND_WRITTEN)]:
        print(f"{case['expected']:4}%  {case['note']}")
    print(f"Wrote {len(fixtures)} cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
