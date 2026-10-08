"""Recording cleanup: where the speech is, and how clean the take was."""
import json
from pathlib import Path

import numpy as np

from pronunciation.audio_cleanup import SAMPLE_RATE, clean_take, frame_levels, high_pass, spectral_gate
from scripts.make_cleanup_fixtures import CASES, synthesize

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "cleanup_cases.json").read_text())


def test_python_cleanup_matches_shared_fixtures():
    # web/src/engine/cleanup.test.js asserts the same cases against the JavaScript port.
    assert [case["name"] for case in FIXTURES] == list(CASES)
    for case in FIXTURES:
        assert case["sections"] == CASES[case["name"]]
        result = clean_take(synthesize(case["sections"]))
        assert {
            "found_speech": result.found_speech,
            "start": result.start,
            "end": result.end,
            "speech_seconds": result.speech_seconds,
            "snr_db": result.snr_db,
            "clipped": round(result.clipped, 6),
            "kept_samples": len(result.samples),
            "cleaned_rms": round(float(np.sqrt(np.mean(result.samples.astype(np.float64) ** 2))), 5)
            if len(result.samples) else 0.0,
        } == case["expected"], case["name"]


def test_speech_is_found_and_noise_is_not():
    by_name = {case["name"]: case["expected"] for case in FIXTURES}
    assert by_name["word_with_silence_around"]["found_speech"]
    assert 0.45 <= by_name["word_with_silence_around"]["speech_seconds"] <= 0.56
    assert not by_name["only_noise"]["found_speech"]
    assert not by_name["only_silence"]["found_speech"]
    assert not by_name["too_short_to_be_speech"]["found_speech"]
    assert by_name["clipped_loud_word"]["clipped"] > 0.05
    assert by_name["very_noisy_room"]["snr_db"] < by_name["noisy_room"]["snr_db"] < by_name["word_with_silence_around"]["snr_db"]


def test_each_step_can_be_switched_off():
    audio = synthesize(CASES["dc_offset_and_rumble"])
    untouched = clean_take(audio, filter_rumble=False, trim=False, denoise=False)
    assert np.array_equal(untouched.samples, audio)
    assert len(clean_take(audio, trim=False).samples) == len(audio)
    assert len(clean_take(audio, pad_seconds=0.0).samples) < len(clean_take(audio, pad_seconds=0.3).samples) <= len(audio)
    # The rumble filter is off by default: the measured effect on the model was nil.
    default = clean_take(audio, denoise=False)
    assert np.array_equal(default.samples, audio[default.start:default.end])
    assert not np.array_equal(clean_take(audio, denoise=False, filter_rumble=True).samples, default.samples)


def test_noise_reduction_is_on_by_default_and_quietens_the_background():
    audio = synthesize(CASES["noisy_room"])
    plain, cleaned = clean_take(audio, denoise=False), clean_take(audio)
    assert (cleaned.start, cleaned.end, cleaned.snr_db) == (plain.start, plain.end, plain.snr_db)
    lead = slice(0, 2000)  # background before the word
    assert np.sqrt(np.mean(cleaned.samples[lead] ** 2)) < 0.5 * np.sqrt(np.mean(plain.samples[lead] ** 2))


def test_filter_removes_offset_and_levels_are_in_db():
    assert abs(float(high_pass(np.full(SAMPLE_RATE, 0.5, dtype=np.float32))[-1])) < 1e-3
    assert np.isclose(frame_levels(np.full(1600, 0.1, dtype=np.float32))[0], -20, atol=1e-4)
    assert clean_take(np.zeros(0, dtype=np.float32)).found_speech is False


def test_spectral_gate_lowers_steady_noise_and_keeps_length():
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(SAMPLE_RATE) * 0.05).astype(np.float32)
    cleaned = spectral_gate(noise, noise)
    assert len(cleaned) == len(noise)
    assert np.sqrt(np.mean(cleaned ** 2)) < 0.5 * np.sqrt(np.mean(noise ** 2))
    short = noise[:100]
    assert spectral_gate(short, short) is short
