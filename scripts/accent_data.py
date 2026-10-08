"""Fetch an accent-labelled subset of VCTK and cache model features for it.

VCTK (CC BY 4.0) has 109 speakers reading newspaper sentences, each tagged with
an accent and region. The full corpus is 11.7 GB, so this reads only the row
groups that hold the chosen speakers, straight from the Hugging Face parquet
export (sanchit-gandhi/vctk), and keeps features rather than audio:

    python -m scripts.accent_data index      # speaker/accent metadata for every row (no audio)
    python -m scripts.accent_data features   # audio for chosen speakers -> transcriptions + last-layer features
    python -m scripts.accent_data layers     # pooled statistics from several layers, for the accent head

Outputs land in data/vctk/ (gitignored).
"""
import argparse
import collections
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
VCTK_DIR = ROOT_DIR / "data" / "vctk"
INDEX_PATH = VCTK_DIR / "index.json"
FEATURES_DIR = VCTK_DIR / "features"
SAMPLES_DIR = VCTK_DIR / "samples"
LAYER_DIR = VCTK_DIR / "layers"
AUDIO_DIR = VCTK_DIR / "audio"
SOURCE_MODEL = ROOT_DIR / "results" / "multitask-phoneme-model"
HF_GLOB = "datasets/sanchit-gandhi/vctk/data/*.parquet"

UTTERANCES_PER_SPEAKER = 60
SAMPLES_PER_SPEAKER = 2
GROUPS_PER_SPEAKER = 3
# Encoder layers cached for the accent head (0 is the convolutional front end, 12 the last layer).
PROBE_LAYERS = (3, 6, 9, 12)
WINDOW_FRAMES = 50  # one second

# Speakers per accent class. VCTK labels all of England "English"; the southern
# group is the closest it has to Received Pronunciation. Australian (2 speakers),
# Welsh (1) and New Zealand (1) are too thin to train or test on.
ACCENT_SPEAKERS = {
    "english_southern": [
        "p225", "p226", "p228", "p229", "p231", "p232", "p240", "p243",
        "p250", "p254", "p257", "p258", "p268", "p273", "p274", "p276",
    ],
    "american": [
        "p294", "p297", "p299", "p300", "p301", "p305", "p306", "p308", "p310",
        "p311", "p318", "p333", "p334", "p339", "p341", "p345", "p360", "p361",
    ],
    "scottish": [
        "p234", "p237", "p241", "p246", "p247", "p249", "p252", "p255",
        "p260", "p262", "p263", "p264", "p265", "p271",
    ],
    "irish": ["p245", "p266", "p283", "p288", "p295", "p298", "p313", "p340", "p364"],
}
# Extra speakers evaluated but never trained on.
PROBE_SPEAKERS = {"australian": ["p326", "p374"]}


def test_speakers(speakers: list[str]) -> set[str]:
    """Every fourth speaker of a class is held out, so train and test never share a voice."""
    return set(sorted(speakers)[3::4])


def build_index():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    rows = []
    for path in sorted(fs.glob(HF_GLOB)):
        with fs.open(path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            for group in range(parquet.num_row_groups):
                table = parquet.read_row_group(group, columns=["speaker_id", "text", "accent", "region", "gender"])
                for position, row in enumerate(table.to_pylist()):
                    rows.append({**row, "file": path, "row_group": group, "row": position})
        print(f"{path.rsplit('/', 1)[-1]}: {len(rows)} rows", flush=True)

    VCTK_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(rows))
    print(f"Wrote {INDEX_PATH}")


def extract_features():
    import librosa
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfFileSystem
    from transformers import Wav2Vec2Processor

    from pronunciation.model import SAMPLE_RATE, ctc_signals
    from src.multitask_model import Wav2Vec2ForPronunciationAssessment

    index = json.loads(INDEX_PATH.read_text())
    accent_of = {
        speaker: accent
        for accent, speakers in {**ACCENT_SPEAKERS, **PROBE_SPEAKERS}.items()
        for speaker in speakers
    }

    # Row groups to read, per parquet file. A speaker's rows are contiguous and each
    # utterance appears twice (two microphones), so the first few groups are enough.
    speaker_groups = collections.defaultdict(list)
    for row in index:
        key = (row["file"], row["row_group"])
        if row["speaker_id"] in accent_of and key not in speaker_groups[row["speaker_id"]]:
            speaker_groups[row["speaker_id"]].append(key)
    groups = collections.defaultdict(set)
    for keys in speaker_groups.values():
        for path, group in sorted(keys)[:GROUPS_PER_SPEAKER]:
            groups[path].add(group)

    processor = Wav2Vec2Processor.from_pretrained(str(SOURCE_MODEL))
    model = Wav2Vec2ForPronunciationAssessment.from_pretrained(str(SOURCE_MODEL)).eval()
    tokenizer = processor.tokenizer
    blank_id = tokenizer.pad_token_id
    special_ids = set(tokenizer.all_special_ids)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    taken = collections.Counter()
    buffers = collections.defaultdict(lambda: {"meta": [], "pooled": [], "frames": []})
    fs = HfFileSystem()
    started = time.time()

    def flush(speaker: str):
        buffer = buffers.pop(speaker)
        lengths = [len(frames) for frames in buffer["frames"]]
        np.savez(
            FEATURES_DIR / f"{speaker}.npz",
            pooled=np.stack(buffer["pooled"]).astype(np.float32),
            frames=np.concatenate(buffer["frames"]).astype(np.float16),
            lengths=np.array(lengths),
        )
        (FEATURES_DIR / f"{speaker}.json").write_text(json.dumps(buffer["meta"]))
        print(f"{speaker} ({accent_of[speaker]}): {len(lengths)} utterances, {time.time() - started:.0f}s", flush=True)

    for path in sorted(groups):
        with fs.open(path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            for group in sorted(groups[path]):
                table = parquet.read_row_group(group, columns=["speaker_id", "audio", "file", "text"])
                for row in table.to_pylist():
                    speaker = row["speaker_id"]
                    if speaker not in accent_of or taken[speaker] >= UTTERANCES_PER_SPEAKER:
                        continue
                    if "_mic1" not in row["file"] or (FEATURES_DIR / f"{speaker}.npz").exists():
                        continue

                    speech, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                    if speech.ndim > 1:
                        speech = speech.mean(axis=1)
                    speech = librosa.resample(speech, orig_sr=sr, target_sr=SAMPLE_RATE)
                    speech, _ = librosa.effects.trim(speech, top_db=35)
                    if len(speech) < SAMPLE_RATE // 2:
                        continue
                    speech = librosa.util.normalize(speech)

                    inputs = processor(speech, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
                    with torch.no_grad():
                        hidden = model.wav2vec2(inputs).last_hidden_state
                        logits = model.lm_head(hidden)
                        score = model.score_head(hidden.mean(dim=1)).item()

                    pred_ids = torch.argmax(logits, dim=-1)
                    confidence, margin = ctc_signals(logits, pred_ids, blank_id)
                    predicted, previous = [], None
                    for token_id in pred_ids[0].tolist():
                        if token_id != previous and token_id != blank_id and token_id not in special_ids:
                            predicted.append(tokenizer.convert_ids_to_tokens(token_id))
                        previous = token_id

                    buffer = buffers[speaker]
                    buffer["pooled"].append(hidden.mean(dim=1)[0].numpy())
                    buffer["frames"].append(hidden[0].numpy())
                    buffer["meta"].append({
                        "speaker": speaker,
                        "accent": accent_of[speaker],
                        "text": row["text"],
                        "predicted": predicted,
                        "confidence": confidence,
                        "margin": margin,
                        "learned_score": score,
                        "seconds": round(len(speech) / SAMPLE_RATE, 2),
                    })

                    if taken[speaker] < SAMPLES_PER_SPEAKER:
                        sf.write(SAMPLES_DIR / f"{speaker}_{taken[speaker]}.wav", speech, SAMPLE_RATE, subtype="PCM_16")
                    taken[speaker] += 1
                    if taken[speaker] == UTTERANCES_PER_SPEAKER:
                        flush(speaker)

    for speaker in list(buffers):
        flush(speaker)
    print(f"Done: {sum(taken.values())} utterances from {len(taken)} speakers")


def pooled_stats(hidden_states, frame_slice: slice) -> np.ndarray:
    """Mean and standard deviation over time for each probed layer: [layers, 2, hidden]."""
    stats = []
    for layer in PROBE_LAYERS:
        frames = hidden_states[layer][0, frame_slice]
        stats.append(torch.stack([frames.mean(dim=0), frames.std(dim=0)]).numpy())
    return np.stack(stats)


def extract_layer_stats():
    """Second pass for the accent head: pooled statistics from several encoder layers.

    The last layer is tuned to phonemes and has shed most accent cues, so the accent
    head reads an earlier layer. Which one is decided on held-out speakers, so this
    caches a few. Each utterance gets stats for the whole clip and for three
    one-second windows (start, middle, end). The 16 kHz audio is kept as FLAC so
    later experiments do not need to download it again.
    """
    import librosa
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfFileSystem
    from transformers import Wav2Vec2Processor

    from pronunciation.model import SAMPLE_RATE
    from src.multitask_model import Wav2Vec2ForPronunciationAssessment

    index = json.loads(INDEX_PATH.read_text())
    accent_of = {
        speaker: accent
        for accent, speakers in {**ACCENT_SPEAKERS, **PROBE_SPEAKERS}.items()
        for speaker in speakers
    }
    speaker_groups = collections.defaultdict(list)
    for row in index:
        key = (row["file"], row["row_group"])
        if row["speaker_id"] in accent_of and key not in speaker_groups[row["speaker_id"]]:
            speaker_groups[row["speaker_id"]].append(key)
    groups = collections.defaultdict(set)
    for keys in speaker_groups.values():
        for path, group in sorted(keys)[:GROUPS_PER_SPEAKER]:
            groups[path].add(group)

    processor = Wav2Vec2Processor.from_pretrained(str(SOURCE_MODEL))
    model = Wav2Vec2ForPronunciationAssessment.from_pretrained(str(SOURCE_MODEL)).eval()

    LAYER_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    taken = collections.Counter()
    buffers = collections.defaultdict(list)
    fs = HfFileSystem()
    started = time.time()

    def flush(speaker: str):
        np.save(LAYER_DIR / f"{speaker}.npy", np.stack(buffers.pop(speaker)).astype(np.float16))
        print(f"{speaker} ({accent_of[speaker]}): {taken[speaker]} utterances, {time.time() - started:.0f}s", flush=True)

    for path in sorted(groups):
        with fs.open(path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            for group in sorted(groups[path]):
                table = parquet.read_row_group(group, columns=["speaker_id", "audio", "file"])
                for row in table.to_pylist():
                    speaker = row["speaker_id"]
                    if speaker not in accent_of or taken[speaker] >= UTTERANCES_PER_SPEAKER:
                        continue
                    if "_mic1" not in row["file"] or (LAYER_DIR / f"{speaker}.npy").exists():
                        continue

                    # Same preprocessing as the first pass, so utterance order lines up with its metadata.
                    speech, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                    if speech.ndim > 1:
                        speech = speech.mean(axis=1)
                    speech = librosa.resample(speech, orig_sr=sr, target_sr=SAMPLE_RATE)
                    speech, _ = librosa.effects.trim(speech, top_db=35)
                    if len(speech) < SAMPLE_RATE // 2:
                        continue
                    speech = librosa.util.normalize(speech)
                    sf.write(AUDIO_DIR / f"{speaker}_{taken[speaker]:02d}.flac", speech, SAMPLE_RATE)

                    inputs = processor(speech, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
                    with torch.no_grad():
                        hidden_states = model.wav2vec2(inputs, output_hidden_states=True).hidden_states

                    frames = hidden_states[0].shape[1]
                    last_start = max(frames - WINDOW_FRAMES, 0)
                    slices = [slice(None)] + [
                        slice(start, start + WINDOW_FRAMES) for start in (0, last_start // 2, last_start)
                    ]
                    buffers[speaker].append(np.stack([pooled_stats(hidden_states, s) for s in slices]))
                    taken[speaker] += 1
                    if taken[speaker] == UTTERANCES_PER_SPEAKER:
                        flush(speaker)

    for speaker in list(buffers):
        flush(speaker)
    print(f"Done: {sum(taken.values())} utterances from {len(taken)} speakers")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["index", "features", "layers"])
    args = parser.parse_args()
    {"index": build_index, "features": extract_features, "layers": extract_layer_stats}[args.stage]()


if __name__ == "__main__":
    main()
