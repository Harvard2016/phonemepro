"""Write scoring cases that the Python and JavaScript scorers must both reproduce.

    python -m scripts.make_scoring_fixtures

The web app scores in the browser (web/src/engine/scoring.js), so the logic
exists twice. tests/fixtures/scoring_cases.json pins both to the same answers.
"""
import json
import random
from pathlib import Path

from pronunciation.scoring import score_attempt

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "scoring_cases.json"
PHONEMES = ["K", "AE1", "T", "S", "IH0", "N", "ZH", "ER0", "AH0", "UW1", "R", "TH", "OY1", "AA1", "L", "IY0", "D", "W"]

HAND_WRITTEN = [
    (["K", "AE1", "T"], ["K", "AE1", "T"], 0.97, 0.95, None),
    (["F", "UW1", "L"], ["F", "UW0", "L"], 0.97, 0.95, None),
    (["SH", "IH1", "P"], ["S", "IY1", "P"], 0.85, 0.8, None),
    (["S", "T", "UW1", "D"], ["S", "UW1", "D"], 0.9, 0.9, 0.7),
    (["V", "IH1", "ZH", "AH0", "N"], ["V", "IH1", "JH", "AH0", "N"], 0.97, 0.95, None),
    (["V", "IH1", "ZH", "AH0", "N"], ["B", "IH1", "AH0", "N"], 0.7, 0.6, 0.55),
    (["V", "IH1", "N"], ["TH", "AO1", "T"], 0.6, 0.5, 0.95),
    (["K"], ["K", "AE", "T", "S", "IH", "N"], 0.1, 0.1, None),
    (["K", "AE1", "T"], [], 0.9, 0.9, None),
    (["T", "T", "T", "T", "K"], ["D", "D", "D", "D", "K"], 0.97, 0.95, 4.2),
    (["W", "AO1", "T", "AH0"], ["W", "AO1", "T", "ER0"], 0.88, 0.82, 0.81),
    (["B", "AA1", "TH"], ["B", "AE1", "TH"], 0.75, 0.7, -1.0),
]


def spans_for(predicted: list[str]) -> list[list[float]]:
    return [[round(0.04 * i, 3), round(0.04 * (i + 1), 3)] for i in range(len(predicted))]


def main():
    rng = random.Random(20261007)
    cases = list(HAND_WRITTEN)
    for _ in range(60):
        target = [rng.choice(PHONEMES) for _ in range(rng.randint(1, 9))]
        predicted = []
        for phoneme in target:
            roll = rng.random()
            if roll < 0.65:
                predicted.append(phoneme)
            elif roll < 0.85:
                predicted.append(rng.choice(PHONEMES))
            if rng.random() < 0.1:
                predicted.append(rng.choice(PHONEMES))
        native = round(rng.uniform(0.2, 1.0), 3) if rng.random() < 0.6 else None
        cases.append((target, predicted, round(rng.uniform(0.4, 1.0), 3), round(rng.uniform(0.3, 1.0), 3), native))

    fixtures = []
    for target, predicted, confidence, margin, native in cases:
        spans = spans_for(predicted)
        expected = score_attempt(target, predicted, confidence, margin, native_score=native, spans=spans)
        fixtures.append({
            "target": target, "predicted": predicted, "confidence": confidence, "margin": margin,
            "native_score": native, "spans": spans, "expected": expected,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(fixtures, indent=1) + "\n")
    print(f"Wrote {len(fixtures)} cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
