"""Export the model for the browser.

    python -m scripts.export_onnx

Wraps the encoder with its three heads (phoneme CTC, learned score, accent) and
the fixed accent projection (results/accent_projection.npz), exports it to ONNX, quantizes the weights to 8 bits, checks that the quantized
model still transcribes like the original, and writes everything the web app
needs into web/public/model/:

    phonemepro.onnx.partNN   the model, split so no file exceeds hosting limits
    model.json               vocabulary, accent labels, chunk list
    report.json              training log, evaluation results (the Model page)

Needs results/multitask-phoneme-model, results/accent_head.pt and
results/accent_projection.npz (python -m scripts.learn_regions fit-projection).
"""
import hashlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pronunciation.model_info import DATASET_INFO, summarize_training
from pronunciation.phonemes import strip_stress
from pronunciation.scoring import align
from scripts.evaluate import DEFAULT_PARQUET, iter_utterances
from scripts.train_accent_head import HEAD_PATH, AccentHead

ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCE_MODEL = ROOT_DIR / "results" / "multitask-phoneme-model"
WORK_DIR = ROOT_DIR / "results" / "onnx"
OUTPUT_DIR = ROOT_DIR / "web" / "public" / "model"
EVAL_RESULTS = ROOT_DIR / "Docs" / "model" / "eval_results.json"
PROJECTION_PATH = ROOT_DIR / "results" / "accent_projection.npz"
CHUNK_BYTES = 40 * 1024 * 1024
PARITY_UTTERANCES = 300


class BrowserModel(nn.Module):
    """input_values -> (logits, score, accent_logits, accent_features)."""

    def __init__(self, model, accent_head, accent_layer: int, projection: dict):
        super().__init__()
        self.encoder = model.wav2vec2
        self.lm_head = model.lm_head
        self.score_head = model.score_head
        self.accent_head = accent_head
        self.accent_layer = accent_layer
        # Fixed projection of the same layer statistics to the short summary that region
        # learning uses (scripts/learn_regions.py): standardize, then project.
        self.register_buffer("feature_mean", torch.tensor(projection["mean"], dtype=torch.float32))
        self.register_buffer("feature_scale", torch.tensor(projection["scale"], dtype=torch.float32))
        self.register_buffer("feature_directions", torch.tensor(projection["directions"], dtype=torch.float32))

    def forward(self, input_values):
        output = self.encoder(input_values, output_hidden_states=True)
        hidden = output.last_hidden_state
        # The accent head reads an earlier layer: its mean and spread over time.
        earlier = output.hidden_states[self.accent_layer][0]
        accent_stats = torch.cat([earlier.mean(dim=0), earlier.std(dim=0)]).unsqueeze(0)
        features = ((accent_stats - self.feature_mean) / self.feature_scale) @ self.feature_directions
        return (
            self.lm_head(hidden), self.score_head(hidden.mean(dim=1)).squeeze(-1),
            self.accent_head(accent_stats), features,
        )


def collapse(frame_ids, blank_id: int, special_ids: set, vocab: list[str]) -> list[str]:
    phonemes, previous = [], None
    for token_id in frame_ids:
        if token_id != previous and token_id != blank_id and token_id not in special_ids:
            phonemes.append(vocab[token_id])
        previous = token_id
    return phonemes


def parity(wrapper, quantized_path: Path, processor, vocab, blank_id, special_ids) -> dict:
    """Phoneme error rate of the original and the quantized model on the same test utterances."""
    import librosa
    import onnxruntime
    import soundfile as sf

    session = onnxruntime.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])
    errors = {"full": 0, "quantized": 0}
    phones = 0
    disagreements = 0
    score_gap = []
    feature_gap = []

    for index, row in enumerate(iter_utterances(DEFAULT_PARQUET)):
        if index >= PARITY_UTTERANCES:
            break
        speech, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if sr != 16000:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        inputs = processor(librosa.util.normalize(speech), sampling_rate=16000, return_tensors="pt").input_values

        with torch.no_grad():
            logits, score, _, features = wrapper(inputs)
        q_logits, q_score, _, q_features = session.run(None, {"input_values": inputs.numpy()})

        reference = [strip_stress(p) for word in row["words"] for p in word["phones"]]
        full = [strip_stress(p) for p in collapse(logits[0].argmax(-1).tolist(), blank_id, special_ids, vocab)]
        quant = [strip_stress(p) for p in collapse(q_logits[0].argmax(-1).tolist(), blank_id, special_ids, vocab)]
        errors["full"] += align(reference, full).errors
        errors["quantized"] += align(reference, quant).errors
        phones += len(reference)
        disagreements += int(full != quant)
        score_gap.append(abs(float(score[0]) - float(q_score[0])))
        feature_gap.append(float(np.abs(features[0].numpy() - q_features[0]).mean()))

    return {
        "parity_utterances": PARITY_UTTERANCES,
        "per_full": round(errors["full"] / phones, 4),
        "per_quantized": round(errors["quantized"] / phones, 4),
        "transcripts_changed": disagreements,
        "mean_score_shift_points": round(float(np.mean(score_gap)) * 10, 3),
        # Features are standardized (spread near 1), so this reads as a fraction of that spread.
        "mean_feature_shift": round(float(np.mean(feature_gap)), 4),
    }


def main():
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import Wav2Vec2Processor

    from src.multitask_model import Wav2Vec2ForPronunciationAssessment

    model = Wav2Vec2ForPronunciationAssessment.from_pretrained(str(SOURCE_MODEL)).eval()
    processor = Wav2Vec2Processor.from_pretrained(str(SOURCE_MODEL))
    saved = torch.load(HEAD_PATH)
    accent_head = AccentHead(2 * model.config.hidden_size, len(saved["classes"]))
    accent_head.load_state_dict(saved["state_dict"])
    projection = np.load(PROJECTION_PATH)
    if int(projection["layer"]) != saved["layer"]:
        raise SystemExit(f"{PROJECTION_PATH} was fitted on layer {int(projection['layer'])}, "
                         f"but the accent head reads layer {saved['layer']}. Refit the projection.")
    wrapper = BrowserModel(model, accent_head.eval(), saved["layer"], projection).eval()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    full_path, quantized_path = WORK_DIR / "phonemepro.fp32.onnx", WORK_DIR / "phonemepro.onnx"
    torch.onnx.export(
        wrapper, (torch.randn(1, 16000),), str(full_path),
        input_names=["input_values"], output_names=["logits", "score", "accent_logits", "accent_features"],
        dynamic_axes={"input_values": {1: "samples"}, "logits": {1: "frames"}},
        opset_version=17, dynamo=False,
    )
    # Only the transformer's matrix multiplications are quantized, per channel. Quantizing
    # the convolutional front end as well saves another 27 MB but raised the phoneme
    # error rate from 23% to 31% on the parity set; this setting is within 0.1 points.
    quantize_dynamic(
        str(full_path), str(quantized_path),
        op_types_to_quantize=["MatMul", "Gemm"], weight_type=QuantType.QInt8, per_channel=True,
    )
    full_path.unlink()

    tokenizer = processor.tokenizer
    vocab = [tokenizer.convert_ids_to_tokens(i) for i in range(len(tokenizer))]
    blank_id = tokenizer.pad_token_id
    special_ids = sorted(set(tokenizer.all_special_ids) | {blank_id})
    checks = parity(wrapper, quantized_path, processor, vocab, blank_id, set(special_ids))
    print(json.dumps(checks, indent=2))

    # Split into chunks and write the manifest.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    data = quantized_path.read_bytes()
    chunks = []
    for number, start in enumerate(range(0, len(data), CHUNK_BYTES)):
        name = f"phonemepro.onnx.part{number:02d}"
        (OUTPUT_DIR / name).write_bytes(data[start:start + CHUNK_BYTES])
        chunks.append(name)

    meta = {
        "version": hashlib.sha256(data).hexdigest()[:12],
        "bytes": len(data),
        "chunks": chunks,
        "sample_rate": 16000,
        "vocab": vocab,
        "blankId": blank_id,
        "specialIds": special_ids,
        "accents": [{"id": c, "name": saved["class_names"][c]} for c in saved["classes"]],
        "features": int(projection["directions"].shape[1]),
    }
    (OUTPUT_DIR / "model.json").write_text(json.dumps(meta) + "\n")

    state = json.loads((SOURCE_MODEL / "trainer_state.json").read_text())
    report = {
        "log_history": [
            {key: entry[key] for key in ("step", "loss", "eval_loss") if key in entry}
            for entry in state.get("log_history", []) if "loss" in entry or "eval_loss" in entry
        ],
        "config": {
            key: getattr(model.config, key)
            for key in ("model_type", "hidden_size", "num_attention_heads", "num_hidden_layers", "vocab_size")
        },
        "training": summarize_training(state),
        "runtime": {"mode": "multitask", "bytes": len(data), **checks},
        "dataset": DATASET_INFO,
        "evaluation": json.loads(EVAL_RESULTS.read_text()),
    }
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report) + "\n")
    print(f"Wrote {len(chunks)} chunks ({len(data) / 1e6:.1f} MB), model.json and report.json to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
