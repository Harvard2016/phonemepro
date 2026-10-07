"""Phoneme alignment and pronunciation scoring.

Pure functions only: nothing here touches the model, so it is cheap to test.

Scoring compares phonemes with lexical stress stripped. The fine-tuned model
learned SpeechOcean762's stress convention (for example monosyllables are
labelled ``AA0``), while practice targets use CMUdict stress (``AA1``), so a
stress digit mismatch is not evidence of a mispronunciation. Stress agreement
is still reported separately as a soft signal.
"""
import difflib
from dataclasses import dataclass, field

from .phonemes import (
    UNSCORABLE_PHONEMES,
    UNSCORABLE_TIP,
    build_clarity_coaching,
    get_tip,
    stress_of,
    strip_stress,
)

# Weights for the 0-10 score when the model predicts phonemes directly.
# On SpeechOcean762, phoneme accuracy carries nearly all of the correlation with
# human accuracy ratings; CTC certainty adds none, so it only nudges the score.
PHONEME_ACCURACY_WEIGHT = 0.90
ACOUSTIC_WEIGHT = 0.10

# CTC confidence / margin ranges mapped onto 0..1 "acoustic certainty".
# Chosen from the fine-tuned model's spread on the SpeechOcean762 test split
# (confidence 5th-95th percentile: 0.73-0.91, margin: 0.63-0.89).
CONFIDENCE_RANGE = (0.60, 0.90)
MARGIN_RANGE = (0.50, 0.90)

# When the model has a learned score head, its rating is blended with the alignment
# score. The head only hears the audio, not the target word, so on its own it would
# rate a clearly spoken wrong word highly; the cap keeps the result tied to the
# target. Weight chosen on held-out training speakers.
LEARNED_SCORE_WEIGHT = 0.6
LEARNED_SCORE_CAP_POINTS = 3.0

# Thresholds used only by the public ASR fallback, which is far more peaked.
ASR_CONFIDENCE_RANGE = (0.82, 0.96)
ASR_MARGIN_RANGE = (0.55, 0.80)


@dataclass(frozen=True)
class Op:
    kind: str  # equal | substitute | delete | insert
    ref_index: int | None
    hyp_index: int | None


@dataclass
class Alignment:
    ops: list[Op] = field(default_factory=list)
    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def align(ref: list[str], hyp: list[str]) -> Alignment:
    """Minimum edit distance alignment of two token sequences."""
    rows, cols = len(ref) + 1, len(hyp) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            diagonal = cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
            cost[i][j] = min(diagonal, cost[i - 1][j] + 1, cost[i][j - 1] + 1)

    alignment = Alignment()
    i, j = len(ref), len(hyp)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] == hyp[j - 1]:
                alignment.ops.append(Op("equal", i - 1, j - 1))
                alignment.hits += 1
            else:
                alignment.ops.append(Op("substitute", i - 1, j - 1))
                alignment.substitutions += 1
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            alignment.ops.append(Op("delete", i - 1, None))
            alignment.deletions += 1
            i -= 1
        else:
            alignment.ops.append(Op("insert", None, j - 1))
            alignment.insertions += 1
            j -= 1

    alignment.ops.reverse()
    return alignment


def phoneme_error_rate(ref: list[str], hyp: list[str]) -> float:
    return align(ref, hyp).errors / max(len(ref), 1)


def _scale(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return max(0.0, min(1.0, (value - low) / (high - low)))


def acoustic_certainty(confidence: float, margin: float, phoneme_model: bool = True) -> float:
    confidence_range = CONFIDENCE_RANGE if phoneme_model else ASR_CONFIDENCE_RANGE
    margin_range = MARGIN_RANGE if phoneme_model else ASR_MARGIN_RANGE
    return 0.7 * _scale(confidence, confidence_range) + 0.3 * _scale(margin, margin_range)


def combine_scores(alignment_score: float, learned_score: float) -> float:
    """Blend the 0-10 alignment score with the score head's 0-10 rating."""
    blended = LEARNED_SCORE_WEIGHT * learned_score + (1 - LEARNED_SCORE_WEIGHT) * alignment_score
    return max(0.0, min(10.0, blended, alignment_score + LEARNED_SCORE_CAP_POINTS))


def empty_attempt(target_phonemes: list[str]) -> dict:
    """Result for a recording in which nothing was recognized."""
    return {
        "score": 0,
        "wer": 1.0,
        "per_with_stress": 1.0,
        "metrics": {"accuracy": 0, "completeness": 0, "fluency": 0},
        "stress": {"matched": 0, "total": 0},
        "phoneme_results": [
            {"phoneme": p, "status": "missed", "heard": None, "tip": get_tip(p)} for p in target_phonemes
        ],
        "errors": [],
        "feedback": "I didn't hear anything clearly. Try again?",
        "coaching_tips": [],
    }


def score_attempt(
    target_phonemes: list[str],
    predicted_phonemes: list[str],
    confidence: float,
    margin: float,
    *,
    native_score: float | None = None,
    phoneme_model: bool = True,
    word: str | None = None,
    recognized_text: str | None = None,
    spans: list[tuple[float, float]] | None = None,
) -> dict:
    """Score one attempt and build per-phoneme feedback.

    `spans` are optional (start, end) seconds for each predicted phoneme; when given,
    every heard phoneme in the result carries the slice of audio it came from.
    """
    if not predicted_phonemes:
        return empty_attempt(target_phonemes)

    ref = [strip_stress(p) for p in target_phonemes]
    hyp = [strip_stress(p) for p in predicted_phonemes]
    alignment = align(ref, hyp)
    total_target = max(len(target_phonemes), 1)

    # Sounds the phoneme model cannot recognize are given the benefit of the doubt.
    unscored = {
        op.ref_index
        for op in alignment.ops
        if phoneme_model and op.kind in ("substitute", "delete") and ref[op.ref_index] in UNSCORABLE_PHONEMES
    }
    for op in alignment.ops:
        if op.ref_index in unscored:
            alignment.hits += 1
            if op.kind == "substitute":
                alignment.substitutions -= 1
            else:
                alignment.deletions -= 1

    per = alignment.errors / total_target
    per_with_stress = phoneme_error_rate(target_phonemes, predicted_phonemes)
    accuracy = max(0.0, 1 - per)
    certainty = acoustic_certainty(confidence, margin, phoneme_model)
    asr_heard_word = (
        not phoneme_model and word is not None and recognized_text is not None
        and recognized_text.lower() == word.lower()
    )

    if phoneme_model:
        score = 10 * (PHONEME_ACCURACY_WEIGHT * accuracy + ACOUSTIC_WEIGHT * certainty)
        if native_score is not None:
            # The regression head was trained on 0.0-1.0 targets.
            score = combine_scores(score, max(0.0, min(1.0, native_score)) * 10)
    else:
        # ASR fallback: word recovery is only one signal, so acoustic evidence weighs more.
        word_match = (
            difflib.SequenceMatcher(None, recognized_text.lower(), word.lower()).ratio()
            if word and recognized_text is not None
            else accuracy
        )
        score = 10 * (0.55 * accuracy + 0.15 * word_match + 0.30 * certainty)
        if asr_heard_word and certainty < 0.92:
            score = min(score, 8.4)
    score = round(score, 1)

    max_insertions_tolerated = max(2, total_target // 2)
    insertion_component = max(0.0, 1 - alignment.insertions / max_insertions_tolerated)
    metrics = {
        "accuracy": round(alignment.hits / total_target * 100),
        "completeness": round((alignment.hits + alignment.substitutions) / total_target * 100),
        "fluency": round((0.55 * insertion_component + 0.45 * certainty) * 100),
    }

    phoneme_results = [
        {"phoneme": p, "status": "missed", "heard": None, "tip": get_tip(p)} for p in target_phonemes
    ]
    errors = []
    stress_total = stress_matched = 0

    for op in alignment.ops:
        if op.kind == "insert":
            continue

        target = target_phonemes[op.ref_index]
        entry = phoneme_results[op.ref_index]
        if spans and op.hyp_index is not None:
            entry["start"], entry["end"] = spans[op.hyp_index]

        if op.ref_index in unscored:
            heard = predicted_phonemes[op.hyp_index] if op.hyp_index is not None else None
            entry.update(status="unscored", heard=heard, tip=UNSCORABLE_TIP)
            continue

        if op.kind == "equal":
            heard = predicted_phonemes[op.hyp_index]
            entry.update(status="correct", heard=heard, tip="")
            if stress_of(target) is not None:
                stress_total += 1
                matched = stress_of(heard) == stress_of(target)
                stress_matched += int(matched)
                entry["stress_match"] = matched
        elif op.kind == "substitute":
            heard = predicted_phonemes[op.hyp_index]
            entry.update(status="error", heard=heard, tip=f"Heard {strip_stress(heard)} instead.")
            errors.append({
                "type": "substitute",
                "index": op.ref_index,
                "expected": [target],
                "heard": [heard],
                "tip": get_tip(target),
            })
        else:
            errors.append({
                "type": "delete",
                "index": op.ref_index,
                "expected": [target],
                "heard": [],
                "tip": get_tip(target),
            })

    coaching_tips = []
    if not errors and score >= 8.5:
        feedback = "Great! The phoneme match is strong."
    elif asr_heard_word:
        feedback = f"Close. I heard '{recognized_text}', but the pronunciation is not clear enough yet for a top score."
        coaching_tips = build_clarity_coaching(target_phonemes)
    elif not errors:
        feedback = "Every sound was there. Say it a little more clearly for a top score."
        coaching_tips = build_clarity_coaching(target_phonemes)
    else:
        feedback = "Focus on the highlighted sounds."
        seen = set()
        for error in errors:
            if error["tip"] not in seen:
                coaching_tips.append(error["tip"])
                seen.add(error["tip"])
        coaching_tips = coaching_tips[:3]

    return {
        "score": score,
        "wer": round(per, 2),
        "per_with_stress": round(per_with_stress, 2),
        "metrics": metrics,
        "stress": {"matched": stress_matched, "total": stress_total},
        "phoneme_results": phoneme_results,
        "errors": errors,
        "feedback": feedback,
        "coaching_tips": coaching_tips,
    }
