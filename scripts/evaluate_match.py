"""Choose the threshold for "this take was an attempt at its target".

    python -m scripts.evaluate_match

A practice take is kept only if it is recognisably the target word. The rule has
to let heavily accented speech through and keep a different word out. Three
rules are compared at several thresholds:

  exact      share of target sounds heard exactly as written
  tolerant   share heard as the same broad class (pronunciation/sound_match.py):
             any vowel for a vowel, consonants within accent-swap groups
  any_sound  share with any sound heard in their place (the old "attempted" check)

Both sides are measured from model transcriptions already cached on disk, with
no audio and no model run:

  should pass   SpeechOcean762 test utterances and words that human annotators
                scored 5 out of 10 or lower (heavily accented learners), and VCTK
                Scottish and Irish speakers scored against the American dictionary
  should fail   the same transcriptions scored against a different target of
                similar length

Practice words are short, so everything is measured twice: on whole utterances,
and on single words with 3 to 5 target sounds. Words are cut out of an utterance
by aligning the transcription to the dictionary pronunciation of the whole
utterance and taking the sounds that fall on each word. (Timings of the heard
sounds are not cached, so the cut uses the alignment, not time.)

Needs results/eval/predictions.jsonl (python -m scripts.evaluate infer) and
data/vctk/features (python -m scripts.accent_data features).
Writes Docs/model/match_threshold.json.
"""
import json
import random
from pathlib import Path

from pronunciation.lexicon import UnknownWord, normalize_text, word_phonemes
from pronunciation.phonemes import strip_stress
from pronunciation.scoring import align
from pronunciation.sound_match import sounds_attempted
from scripts.accent_data import ACCENT_SPEAKERS, FEATURES_DIR
from scripts.evaluate import DEFAULT_PARQUET, PREDICTIONS_PATH

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT_DIR / "Docs" / "model" / "match_threshold.json"
THRESHOLDS = (30, 40, 50, 60, 70, 80)
LOW_HUMAN_SCORE = 5  # out of 10
WORD_SOUNDS = (3, 4, 5)
SEED = 7


def exact_share(target: list[str], heard: list[str]) -> int:
    """The rule this replaces: percentage of target sounds heard exactly as written."""
    ref = [strip_stress(p) for p in target]
    return round(100 * align(ref, [strip_stress(p) for p in heard]).hits / max(len(ref), 1))


def any_sound_share(target: list[str], heard: list[str]) -> int:
    """The old "most target sounds attempted" check: any sound at all in the target sound's place."""
    ref = [strip_stress(p) for p in target]
    alignment = align(ref, [strip_stress(p) for p in heard])
    return round(100 * (alignment.hits + alignment.substitutions) / max(len(ref), 1))


RULES = {"exact": exact_share, "tolerant": sounds_attempted, "any_sound": any_sound_share}


def cut_words(words: list[list[str]], heard: list[str]) -> list[list[str]]:
    """The heard sounds that fall on each word, from one alignment of the whole utterance."""
    owner = [index for index, word in enumerate(words) for _ in word]
    flat = [strip_stress(p) for word in words for p in word]
    pieces = [[] for _ in words]
    current = 0
    for op in align(flat, [strip_stress(p) for p in heard]).ops:
        if op.ref_index is not None:
            current = owner[op.ref_index]
        if op.hyp_index is not None:
            pieces[current].append(heard[op.hyp_index])
    return pieces


def load_speechocean() -> list[dict]:
    """[{words: [phones], word_scores, heard, score}] for every test utterance."""
    import pyarrow.parquet as pq

    rows = pq.ParquetFile(DEFAULT_PARQUET).read(columns=["text", "accuracy", "words"]).to_pylist()
    predictions = [json.loads(line) for line in PREDICTIONS_PATH.read_text().splitlines()]
    if len(rows) != len(predictions):
        raise SystemExit("predictions.jsonl does not cover the test split; rerun scripts.evaluate infer")
    utterances = []
    for row, prediction in zip(rows, predictions):
        if row["text"] != prediction["text"]:
            raise SystemExit("predictions.jsonl is out of order with the test split")
        utterances.append({
            "text": row["text"],
            "words": [word["phones"] for word in row["words"]],
            "word_scores": [word["accuracy"] for word in row["words"]],
            "heard": prediction["predicted"],
            "score": row["accuracy"],
        })
    return utterances


def load_vctk(accent: str) -> list[dict]:
    """Native speakers of `accent`, with American dictionary targets."""
    utterances = []
    for speaker in ACCENT_SPEAKERS[accent]:
        for meta in json.loads((FEATURES_DIR / f"{speaker}.json").read_text()):
            try:
                words = [word_phonemes(word, "ga") for word in normalize_text(meta["text"])]
            except UnknownWord:
                continue
            if words:
                utterances.append({"text": meta["text"], "words": words, "heard": meta["predicted"]})
    return utterances


def as_utterances(utterances: list[dict]) -> list[tuple]:
    return [([p for word in u["words"] for p in word], u["heard"]) for u in utterances]


def as_words(utterances: list[dict], keep=lambda utterance, index: True) -> list[tuple]:
    """(target, heard) for every word of 3 to 5 sounds."""
    pairs = []
    for utterance in utterances:
        for index, (word, piece) in enumerate(zip(utterance["words"], cut_words(utterance["words"], utterance["heard"]))):
            if len(word) in WORD_SOUNDS and keep(utterance, index):
                pairs.append((word, piece))
    return pairs


def mismatched(pairs: list[tuple], rng: random.Random, slack: float) -> list[tuple]:
    """Each transcription against a different target of similar length: someone saying the wrong thing."""
    wrong = []
    for target, heard in pairs:
        own = [strip_stress(p) for p in target]
        for _ in range(50):
            other = rng.choice(pairs)[0]
            if [strip_stress(p) for p in other] != own and abs(len(other) - len(target)) <= slack * len(target):
                wrong.append((other, heard))
                break
    return wrong


def pass_rates(pairs: list[tuple]) -> dict:
    rates = {"takes": len(pairs)}
    for rule, measure in RULES.items():
        values = [measure(target, heard) for target, heard in pairs]
        rates[rule] = {str(t): round(sum(value >= t for value in values) / len(pairs), 4) for t in THRESHOLDS}
    return rates


def separation(scale: dict, rule: str) -> dict:
    """For each threshold: genuine takes kept (mean over the groups) minus wrong takes kept."""
    genuine = [group for name, group in scale.items() if name != "wrong_target"]
    return {
        str(t): round(
            sum(group[rule][str(t)] for group in genuine) / len(genuine) - scale["wrong_target"][rule][str(t)], 4)
        for t in THRESHOLDS
    }


def main():
    rng = random.Random(SEED)
    speechocean = load_speechocean()
    low = [u for u in speechocean if u["score"] <= LOW_HUMAN_SCORE]
    scottish, irish = load_vctk("scottish"), load_vctk("irish")

    utterance_sets = {
        "speechocean_low_score": as_utterances(low),
        "vctk_scottish": as_utterances(scottish),
        "vctk_irish": as_utterances(irish),
    }
    word_sets = {
        "speechocean_low_score": as_words(speechocean, lambda u, i: u["word_scores"][i] <= LOW_HUMAN_SCORE),
        "vctk_scottish": as_words(scottish),
        "vctk_irish": as_words(irish),
    }
    # Wrong targets are drawn from every speaker, good and bad, so the test is not made easy by poor speech.
    utterance_sets["wrong_target"] = mismatched(as_utterances(speechocean + scottish + irish), rng, slack=0.25)
    word_sets["wrong_target"] = mismatched(as_words(speechocean + scottish + irish), rng, slack=0.0)

    results = {
        "method": "share of target sounds matched under each rule, from cached transcriptions",
        "rules": {
            "exact": "heard exactly as written",
            "tolerant": "heard as the same broad class (web/src/engine/sound_groups.json)",
            "any_sound": "any sound heard in its place",
        },
        "low_human_score": f"{LOW_HUMAN_SCORE} out of 10 or lower",
        "not_measured": "VCTK Indian speakers: their recordings and transcriptions are not cached on disk",
        "separation": "genuine takes kept, averaged over the three groups, minus wrong targets kept",
        "scales": {},
    }
    for name, sets in (("utterance", utterance_sets), ("word_3_to_5_sounds", word_sets)):
        scale = {group: pass_rates(pairs) for group, pairs in sets.items()}
        results["scales"][name] = {"groups": scale, "separation": {rule: separation(scale, rule) for rule in RULES}}

    # Practice words are short, so the rule is chosen on single words.
    words = results["scales"]["word_3_to_5_sounds"]
    best_rule, threshold = max(
        ((rule, t) for rule in RULES for t in THRESHOLDS), key=lambda choice: words["separation"][choice[0]][str(choice[1])])
    results["chosen"] = {
        "rule": best_rule,
        "threshold": threshold,
        "chosen_on": "single words of 3 to 5 sounds, because practice words are short",
        "kept": {
            group: {
                "kept": rates[best_rule][str(threshold)],
                "dropped": round(1 - rates[best_rule][str(threshold)], 4),
                "takes": rates["takes"],
            }
            for group, rates in words["groups"].items()
        },
    }
    # Short targets carry less evidence, so the chosen rule is also broken down by target length.
    measure = RULES[best_rule]
    results["chosen"]["kept_by_target_sounds"] = {
        str(sounds): {
            group: round(
                sum(measure(target, heard) >= threshold for target, heard in pairs if len(target) == sounds)
                / max(sum(len(target) == sounds for target, _ in pairs), 1), 4)
            for group, pairs in word_sets.items()
        }
        for sounds in WORD_SOUNDS
    }
    REPORT_PATH.write_text(json.dumps(results, indent=2) + "\n")

    for name, scale in results["scales"].items():
        for rule in RULES:
            print(f"\n{name}, {rule} rule: share kept at each threshold")
            print(f"{'group':24} {'takes':>6}  " + "  ".join(f"{'>=' + str(t) + '%':>6}" for t in THRESHOLDS))
            for group, rates in scale["groups"].items():
                print(f"{group:24} {rates['takes']:6}  " + "  ".join(f"{rates[rule][str(t)] * 100:6.1f}" for t in THRESHOLDS))
            print(f"{'separation':24} {'':6}  " + "  ".join(f"{scale['separation'][rule][str(t)] * 100:6.1f}" for t in THRESHOLDS))
    print(f"\nChosen: {best_rule} match of at least {threshold}%. Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
