from pronunciation.phonemes import get_tip, stress_of, strip_stress
from pronunciation.scoring import align, combine_scores, phoneme_error_rate, score_attempt

HIGH = {"confidence": 0.97, "margin": 0.95}


def kinds(alignment):
    return [op.kind for op in alignment.ops]


def test_strip_stress_and_stress_of():
    assert strip_stress("AE1") == "AE"
    assert strip_stress("K") == "K"
    assert stress_of("AH0") == "0"
    assert stress_of("K") is None


def test_get_tip_ignores_stress():
    assert get_tip("AE1") == get_tip("AE")
    assert "QQ" in get_tip("QQ")


def test_align_identical():
    alignment = align(["K", "AE", "T"], ["K", "AE", "T"])
    assert alignment.hits == 3
    assert alignment.errors == 0
    assert kinds(alignment) == ["equal"] * 3


def test_align_substitution_pairs_the_right_tokens():
    alignment = align(["K", "AE", "T"], ["K", "EH", "T"])
    assert kinds(alignment) == ["equal", "substitute", "equal"]
    sub = alignment.ops[1]
    assert (sub.ref_index, sub.hyp_index) == (1, 1)


def test_align_deletion_and_insertion():
    deletion = align(["S", "T", "UW", "D"], ["S", "UW", "D"])
    assert (deletion.deletions, deletion.insertions, deletion.substitutions) == (1, 0, 0)

    insertion = align(["S", "UW", "D"], ["S", "T", "UW", "D"])
    assert (insertion.deletions, insertion.insertions, insertion.substitutions) == (0, 1, 0)


def test_align_empty_sides():
    assert align([], ["K"]).insertions == 1
    assert align(["K"], []).deletions == 1
    assert align([], []).errors == 0


def test_phoneme_error_rate():
    assert phoneme_error_rate(["K", "AE", "T"], ["K", "AE", "T"]) == 0
    assert phoneme_error_rate(["K", "AE", "T", "S"], ["K", "EH", "T"]) == 0.5


def test_perfect_attempt_scores_ten():
    result = score_attempt(["K", "AE1", "T"], ["K", "AE1", "T"], **HIGH)
    assert result["score"] == 10.0
    assert result["wer"] == 0
    assert result["errors"] == []
    assert result["metrics"] == {"accuracy": 100, "completeness": 100, "fluency": 100}
    assert all(p["status"] == "correct" for p in result["phoneme_results"])


def test_stress_mismatch_is_not_a_pronunciation_error():
    # The model labels this vowel UW0 where CMUdict says UW1.
    result = score_attempt(["F", "UW1", "L"], ["F", "UW0", "L"], **HIGH)
    assert result["score"] == 10.0
    assert result["wer"] == 0
    assert result["errors"] == []
    assert result["per_with_stress"] > 0
    assert result["stress"] == {"matched": 0, "total": 1}
    assert result["phoneme_results"][1]["status"] == "correct"
    assert result["phoneme_results"][1]["stress_match"] is False


def test_substitution_reports_what_was_heard_per_phoneme():
    result = score_attempt(["SH", "IH1", "P"], ["S", "IY1", "P"], **HIGH)
    statuses = [p["status"] for p in result["phoneme_results"]]
    assert statuses == ["error", "error", "correct"]
    assert result["phoneme_results"][0]["heard"] == "S"
    assert result["phoneme_results"][1]["heard"] == "IY1"
    assert [e["heard"] for e in result["errors"]] == [["S"], ["IY1"]]
    assert result["metrics"]["accuracy"] == 33
    assert result["metrics"]["completeness"] == 100


def test_deletion_marks_phoneme_missed():
    result = score_attempt(["S", "T", "UW1", "D"], ["S", "UW1", "D"], **HIGH)
    assert result["phoneme_results"][1]["status"] == "missed"
    assert result["errors"] == [
        {"type": "delete", "index": 1, "expected": ["T"], "heard": [], "tip": get_tip("T")}
    ]
    assert result["metrics"]["completeness"] == 75


def test_more_errors_means_lower_score():
    target = ["B", "AH1", "T", "AH0", "N"]
    perfect = score_attempt(target, target, **HIGH)["score"]
    one_off = score_attempt(target, ["B", "AH1", "D", "AH0", "N"], **HIGH)["score"]
    two_off = score_attempt(target, ["P", "AH1", "D", "AH0", "N"], **HIGH)["score"]
    assert perfect > one_off > two_off


def test_low_acoustic_certainty_lowers_score():
    target = ["K", "AE1", "T"]
    confident = score_attempt(target, target, confidence=0.97, margin=0.95)["score"]
    unsure = score_attempt(target, target, confidence=0.5, margin=0.3)["score"]
    assert confident > unsure
    assert 0 <= unsure <= 10


def test_score_is_bounded_with_many_insertions():
    result = score_attempt(["K"], ["K", "AE", "T", "S", "IH", "N"], confidence=0.1, margin=0.1)
    assert 0 <= result["score"] <= 10
    assert result["wer"] == 5.0


def test_empty_prediction():
    result = score_attempt(["K", "AE1", "T"], [], **HIGH)
    assert result["score"] == 0
    assert result["wer"] == 1.0
    assert all(p["status"] == "missed" for p in result["phoneme_results"])


def test_learned_score_is_blended_with_alignment():
    target = ["K", "AE1", "T"]
    # Alignment alone gives 10.0; a learned rating of 5.0 pulls it to 0.6 * 5 + 0.4 * 10.
    assert score_attempt(target, target, native_score=0.5, **HIGH)["score"] == 7.0
    # Out-of-range head outputs are clamped.
    assert score_attempt(target, target, native_score=4.2, **HIGH)["score"] == 10.0
    assert score_attempt(target, target, native_score=-1.0, **HIGH)["score"] == 4.0


def test_learned_score_cannot_rescue_the_wrong_word():
    # Every target sound is wrong, but the audio itself is fluent, so the head rates it 9.5.
    result = score_attempt(["V", "IH1", "N"], ["TH", "AO1", "T"], native_score=0.95, confidence=0.6, margin=0.5)
    assert result["score"] == 3.0
    assert combine_scores(2.0, 9.0) == 5.0
    assert combine_scores(8.0, 9.0) == 8.6


def test_asr_fallback_caps_unclear_exact_match():
    result = score_attempt(
        ["K", "AE1", "T"], ["K", "AE1", "T"],
        confidence=0.85, margin=0.6,
        phoneme_model=False, word="cat", recognized_text="cat",
    )
    assert result["score"] <= 8.4
    assert "cat" in result["feedback"]


def test_coaching_tips_are_deduplicated_and_capped():
    target = ["T", "T", "T", "T", "K"]
    result = score_attempt(target, ["D", "D", "D", "D", "K"], **HIGH)
    assert len(result["coaching_tips"]) == 1
    assert len(result["errors"]) == 4


def test_spans_follow_alignment_not_position():
    # T is dropped, so the third target phoneme was the second one heard.
    result = score_attempt(
        ["S", "T", "UW1", "D"], ["S", "UW1", "D"],
        spans=[(0.0, 0.1), (0.1, 0.3), (0.3, 0.42)], **HIGH,
    )
    cells = result["phoneme_results"]
    assert (cells[0]["start"], cells[0]["end"]) == (0.0, 0.1)
    assert "start" not in cells[1]
    assert (cells[2]["start"], cells[2]["end"]) == (0.1, 0.3)
    assert (cells[3]["start"], cells[3]["end"]) == (0.3, 0.42)


def test_sounds_the_model_cannot_recognize_are_not_penalized():
    # The model never emits ZH; here it heard JH in its place.
    target = ["V", "IH1", "ZH", "AH0", "N"]
    result = score_attempt(target, ["V", "IH1", "JH", "AH0", "N"], **HIGH)
    assert result["score"] == 10.0
    assert result["errors"] == []
    assert result["metrics"]["accuracy"] == 100
    cell = result["phoneme_results"][2]
    assert (cell["status"], cell["heard"]) == ("unscored", "JH")

    dropped = score_attempt(target, ["V", "IH1", "AH0", "N"], **HIGH)
    assert dropped["phoneme_results"][2]["status"] == "unscored"
    assert dropped["wer"] == 0

    # Other mistakes in the same word still count.
    mixed = score_attempt(target, ["B", "IH1", "JH", "AH0", "N"], **HIGH)
    assert [e["expected"] for e in mixed["errors"]] == [["V"]]
    assert mixed["metrics"]["accuracy"] == 80


def test_asr_fallback_still_scores_every_sound():
    result = score_attempt(
        ["V", "IH1", "ZH", "AH0", "N"], ["V", "IH1", "JH", "AH0", "N"],
        phoneme_model=False, word="vision", recognized_text="vigin", **HIGH,
    )
    assert result["phoneme_results"][2]["status"] == "error"


def test_python_scorer_matches_shared_fixtures():
    import json
    from pathlib import Path

    cases = json.loads((Path(__file__).parent / "fixtures" / "scoring_cases.json").read_text())
    assert len(cases) > 50
    for case in cases:
        result = score_attempt(
            case["target"], case["predicted"], case["confidence"], case["margin"],
            native_score=case["native_score"], spans=[tuple(span) for span in case["spans"]],
        )
        # JSON has no tuples, so compare through a JSON round trip.
        assert json.loads(json.dumps(result)) == case["expected"]
