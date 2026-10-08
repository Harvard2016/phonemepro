"""Write recording-cleanup cases that the Python and JavaScript implementations must both reproduce.

    python -m scripts.make_cleanup_fixtures

Each case is a recipe for a synthetic recording (silence, tones, noise, clicks)
rather than raw samples, so the fixture stays small. Both test suites rebuild
the audio from the recipe with the same generator and compare where the speech
was found, how clean the take was, and the level of the cleaned audio.
Output: tests/fixtures/cleanup_cases.json
"""
import json
import math
from pathlib import Path

import numpy as np

from pronunciation.audio_cleanup import SAMPLE_RATE, clean_take

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "cleanup_cases.json"

# Each section: [kind, seconds, level, frequency]. "tone" stands in for speech.
CASES = {
    "word_with_silence_around": [["noise", 0.6, 0.002, 0], ["tone", 0.5, 0.3, 220], ["noise", 0.7, 0.002, 0]],
    "word_at_the_very_start": [["tone", 0.4, 0.25, 180], ["noise", 0.8, 0.003, 0]],
    "word_at_the_very_end": [["noise", 0.9, 0.003, 0], ["tone", 0.35, 0.25, 300]],
    "phrase_with_a_pause": [["noise", 0.4, 0.002, 0], ["tone", 0.4, 0.3, 200], ["noise", 0.25, 0.002, 0],
                            ["tone", 0.5, 0.2, 260], ["noise", 0.5, 0.002, 0]],
    "phrase_with_a_long_pause": [["noise", 0.3, 0.002, 0], ["tone", 0.4, 0.3, 200], ["noise", 0.6, 0.002, 0],
                                 ["tone", 0.5, 0.25, 260], ["noise", 0.3, 0.002, 0]],
    "key_click_before_word": [["noise", 0.2, 0.002, 0], ["tone", 0.03, 0.5, 2000], ["noise", 0.7, 0.002, 0],
                              ["tone", 0.5, 0.3, 220], ["noise", 0.4, 0.002, 0]],
    "cough_after_word": [["noise", 0.3, 0.002, 0], ["tone", 0.6, 0.3, 240], ["noise", 0.6, 0.002, 0],
                         ["tone", 0.08, 0.2, 700], ["noise", 0.3, 0.002, 0]],
    "noisy_room": [["noise", 0.5, 0.03, 0], ["tone+noise", 0.6, 0.3, 210], ["noise", 0.5, 0.03, 0]],
    "very_noisy_room": [["noise", 0.5, 0.1, 0], ["tone+noise", 0.6, 0.25, 210], ["noise", 0.5, 0.1, 0]],
    "quiet_speaker": [["noise", 0.5, 0.001, 0], ["tone", 0.5, 0.02, 190], ["noise", 0.5, 0.001, 0]],
    "only_noise": [["noise", 1.2, 0.02, 0]],
    "only_silence": [["silence", 1.0, 0, 0]],
    "too_short_to_be_speech": [["noise", 0.5, 0.002, 0], ["tone", 0.02, 0.4, 500], ["noise", 0.5, 0.002, 0]],
    "clipped_loud_word": [["noise", 0.3, 0.002, 0], ["tone", 0.5, 1.4, 200], ["noise", 0.3, 0.002, 0]],
    "dc_offset_and_rumble": [["rumble", 0.5, 0.2, 30], ["tone+rumble", 0.5, 0.3, 230], ["rumble", 0.5, 0.2, 30]],
    "speech_edge_to_edge": [["tone", 0.8, 0.3, 220]],
    "fraction_of_a_frame": [["tone", 0.01, 0.3, 220]],
}


def synthesize(sections) -> np.ndarray:
    """Build audio from a recipe. The noise is a fixed linear congruential sequence."""
    state = 12345
    samples = []
    position = 0
    for kind, seconds, level, frequency in sections:
        for _ in range(round(seconds * SAMPLE_RATE)):
            state = (1103515245 * state + 12345) % 2147483648
            noise = state / 1073741824 - 1
            tone = math.sin(2 * math.pi * frequency * position / SAMPLE_RATE)
            if kind == "silence":
                value = 0.0
            elif kind == "noise":
                value = level * noise
            elif kind == "tone":
                value = level * tone
            elif kind == "tone+noise":
                value = level * tone + 0.1 * level * noise
            elif kind == "rumble":
                value = 0.1 + level * math.sin(2 * math.pi * 30 * position / SAMPLE_RATE)
            else:  # tone+rumble
                value = 0.1 + 0.2 * math.sin(2 * math.pi * 30 * position / SAMPLE_RATE) + level * tone
            samples.append(max(-1.0, min(1.0, value)))
            position += 1
    return np.array(samples, dtype=np.float32)


def main():
    fixtures = []
    for name, sections in CASES.items():
        result = clean_take(synthesize(sections))
        fixtures.append({
            "name": name,
            "sections": sections,
            "expected": {
                "found_speech": result.found_speech,
                "start": result.start,
                "end": result.end,
                "speech_seconds": result.speech_seconds,
                "snr_db": result.snr_db,
                "clipped": round(result.clipped, 6),
                "kept_samples": len(result.samples),
                # Level of the cleaned audio, so the noise reduction itself is compared too.
                "cleaned_rms": round(float(np.sqrt(np.mean(result.samples.astype(np.float64) ** 2))), 5)
                if len(result.samples) else 0.0,
            },
        })
        print(f"{name:28} {fixtures[-1]['expected']}")
    OUTPUT_PATH.write_text(json.dumps(fixtures, indent=1) + "\n")
    print(f"Wrote {len(fixtures)} cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
