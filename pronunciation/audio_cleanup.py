"""Clean up a recording before it reaches the model.

A take from a laptop microphone has silence or room noise before and after the
word, and sometimes steady background noise. The model pools over everything it
is given, so stray audio changes both the transcription and the score. This
module finds where the speech is, trims to it, reduces steady noise, and reports
how clean the take was.

The defaults are the steps that helped when measured (scripts/evaluate_capture.py,
Docs/model/capture_results.json):

  trim, keeping 0.5 s each side   no cost on clean audio, fewer errors in noise.
                                  A tight 0.12 s margin made clean audio worse:
                                  the model was trained on clips with quiet edges.
  spectral noise reduction        the largest gain in noise, for under half a
                                  point of phoneme error on clean audio.
  rumble filter on the output     no measurable effect, so the audio is left
                                  alone. It is still used to locate the speech.

This is the reference implementation. The web app runs the same steps in
web/src/engine/cleanup.js; both are pinned to tests/fixtures/cleanup_cases.json.
"""
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
HIGH_PASS_HZ = 70.0
FRAME = 320  # 20 ms
HOP = 160  # 10 ms
FLOOR_DB = -80.0
# A frame counts as speech when it is this far above the noise floor and within
# this range of the loudest part of the take.
ABOVE_NOISE_DB = 6.0
BELOW_PEAK_DB = 32.0
# Below this spread between loud and quiet frames there is no edge to find: the take
# is either steady noise or, if it is loud, speech from the first sample to the last.
MIN_DYNAMIC_RANGE_DB = 6.0
LOUD_DB = -35.0
MIN_RUN_FRAMES = 4  # ignore blips shorter than 40 ms
MAX_GAP_FRAMES = 35  # a silence longer than 350 ms separates groups of speech
MIN_GROUP_FRAMES = 12  # a group shorter than 120 ms, standing apart, is a noise burst
PAD_SECONDS = 0.5  # quiet kept on each side of the speech, as in the clips the model was trained on
CLIP_LEVEL = 0.985

# Spectral gate
FFT_SIZE = 512
FFT_HOP = 128
MIN_NOISE_FRAMES = 6
OVER_SUBTRACTION = 1.5
GAIN_FLOOR = 0.12


@dataclass
class CleanTake:
    samples: np.ndarray  # cleaned audio, or the input unchanged when no speech was found
    found_speech: bool
    start: int  # sample offsets of the kept region in the input
    end: int
    speech_seconds: float
    snr_db: float
    clipped: float  # fraction of samples at full scale


def high_pass(samples: np.ndarray) -> np.ndarray:
    """First-order high-pass that removes DC offset and rumble below speech."""
    rc = 1.0 / (2 * np.pi * HIGH_PASS_HZ)
    alpha = rc / (rc + 1.0 / SAMPLE_RATE)
    out = np.empty(len(samples), dtype=np.float64)
    previous_in = previous_out = 0.0
    for index, value in enumerate(samples.astype(np.float64)):
        previous_out = alpha * (previous_out + value - previous_in)
        previous_in = value
        out[index] = previous_out
    return out.astype(np.float32)


def frame_levels(samples: np.ndarray) -> np.ndarray:
    """Energy of each 20 ms frame in dB relative to full scale."""
    count = max(0, 1 + (len(samples) - FRAME) // HOP)
    levels = np.full(count, FLOOR_DB, dtype=np.float64)
    squared = samples.astype(np.float64) ** 2
    for index in range(count):
        power = squared[index * HOP:index * HOP + FRAME].mean()
        if power > 0:
            levels[index] = max(FLOOR_DB, 10 * np.log10(power))
    return levels


def percentile(values: np.ndarray, fraction: float) -> float:
    """Nearest-rank percentile, chosen because it is trivial to reproduce exactly elsewhere."""
    ordered = np.sort(values)
    return float(ordered[int(np.floor(fraction * (len(ordered) - 1)))])


def speech_frames(levels: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Boolean mask of speech frames and the noise floor, or None if there is no clear speech."""
    if len(levels) < MIN_RUN_FRAMES:
        return None
    floor = percentile(levels, 0.1)
    peak = percentile(levels, 0.95)
    if peak - floor < MIN_DYNAMIC_RANGE_DB:
        return (np.ones(len(levels), dtype=bool), floor) if floor > LOUD_DB else None

    active = levels > max(floor + ABOVE_NOISE_DB, peak - BELOW_PEAK_DB)

    # Drop short blips (a key press, a click), then keep everything from the first
    # real run to the last, bridging pauses.
    mask = np.zeros(len(levels), dtype=bool)
    run_start = None
    for index in range(len(active) + 1):
        on = index < len(active) and active[index]
        if on and run_start is None:
            run_start = index
        elif not on and run_start is not None:
            if index - run_start >= MIN_RUN_FRAMES:
                mask[run_start:index] = True
            run_start = None
    if not mask.any():
        return None

    # Split into groups at long silences. A short burst standing apart from the rest
    # (a key press, a cough) is dropped; real speech on both sides of a pause is kept.
    groups = []
    indices = np.flatnonzero(mask)
    group_start = previous = indices[0]
    for index in indices[1:]:
        if index - previous > MAX_GAP_FRAMES:
            groups.append((group_start, previous))
            group_start = index
        previous = index
    groups.append((group_start, previous))
    sizes = [int(mask[start:end + 1].sum()) for start, end in groups]
    solid = [group for group, size in zip(groups, sizes) if size >= MIN_GROUP_FRAMES or size == max(sizes)]
    first, last = solid[0][0], solid[-1][1]

    kept = np.zeros(len(levels), dtype=bool)
    kept[first:last + 1] = True
    return kept, floor


def spectral_gate(samples: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Subtract the average spectrum of `noise` (audio known to hold no speech) from `samples`."""
    window = np.hanning(FFT_SIZE + 1)[:-1]

    def spectra(audio):
        count = 1 + (len(audio) - FFT_SIZE) // FFT_HOP
        return np.stack([np.fft.rfft(audio[i * FFT_HOP:i * FFT_HOP + FFT_SIZE] * window) for i in range(count)])

    if len(noise) < FFT_SIZE + (MIN_NOISE_FRAMES - 1) * FFT_HOP or len(samples) < FFT_SIZE:
        return samples
    profile = np.abs(spectra(noise.astype(np.float64))).mean(axis=0)

    padded = np.concatenate([np.zeros(FFT_SIZE), samples.astype(np.float64), np.zeros(FFT_SIZE)])
    frames = spectra(padded)
    magnitude = np.abs(frames)
    gain = np.clip((magnitude - OVER_SUBTRACTION * profile) / np.maximum(magnitude, 1e-12), GAIN_FLOOR, 1.0)

    out = np.zeros(len(padded))
    weight = np.zeros(len(padded))
    for index, frame in enumerate(frames * gain):
        start = index * FFT_HOP
        out[start:start + FFT_SIZE] += np.fft.irfft(frame, FFT_SIZE) * window
        weight[start:start + FFT_SIZE] += window ** 2
    out /= np.maximum(weight, 1e-8)
    return out[FFT_SIZE:FFT_SIZE + len(samples)].astype(np.float32)


def clean_take(
    samples: np.ndarray,
    denoise: bool = True,
    filter_rumble: bool = False,
    trim: bool = True,
    pad_seconds: float = PAD_SECONDS,
) -> CleanTake:
    """Find the speech, trim to it, reduce steady noise, and measure quality. Each step can be switched."""
    clipped = float(np.mean(np.abs(samples) >= CLIP_LEVEL)) if len(samples) else 0.0
    # Speech is always located on the filtered signal, where rumble cannot hide it.
    filtered = high_pass(samples)
    audio = filtered if filter_rumble else samples.astype(np.float32)
    levels = frame_levels(filtered)
    found = speech_frames(levels)
    if found is None:
        return CleanTake(audio, False, 0, len(audio), 0.0, 0.0, clipped)

    mask, floor = found
    indices = np.flatnonzero(mask)
    speech_start = int(indices[0]) * HOP
    speech_end = min(len(audio), int(indices[-1]) * HOP + FRAME)
    pad = int(pad_seconds * SAMPLE_RATE)
    start = max(0, speech_start - pad) if trim else 0
    end = min(len(audio), speech_end + pad) if trim else len(audio)

    # Signal-to-noise ratio: speech frames against the typical frame outside them (the
    # median, so a click or cough does not count as background), or the quietest
    # frames when the take is speech from edge to edge.
    speech_level = 10 * np.log10(np.mean(10 ** (levels[mask] / 10)))
    outside = levels[~mask]
    noise_level = percentile(outside, 0.5) if len(outside) >= MIN_RUN_FRAMES else floor
    snr = float(np.clip(speech_level - noise_level, 0.0, 60.0))

    kept = audio[start:end]
    if denoise:
        kept = spectral_gate(kept, np.concatenate([audio[:speech_start], audio[speech_end:]]))

    return CleanTake(kept, True, start, end, round((speech_end - speech_start) / SAMPLE_RATE, 3), round(snr, 1), clipped)
