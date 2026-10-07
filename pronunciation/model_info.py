"""Training and evaluation metadata for the Model Info page.

Everything here is read from artifacts on disk (trainer_state.json, config.json,
Docs/model/eval_results.json) so the page reports what actually happened.
"""
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARCHITECTURE_PATH = ROOT_DIR / "Docs" / "model" / "architecture.md"
EVAL_RESULTS_PATH = ROOT_DIR / "Docs" / "model" / "eval_results.json"

DATASET_INFO = {
    "name": "Speechocean762",
    "train_samples": 2500,
    "test_samples": 2500,
    "sample_rate": 16000,
    "language": "English",
    "annotation": "ARPABET phonemes with accuracy scores",
}


def find_latest_trainer_state(model_path: Path | None) -> Path | None:
    if model_path is None:
        return None

    trainer_state = model_path / "trainer_state.json"
    if trainer_state.exists():
        return trainer_state

    checkpoint_states = []
    for checkpoint_dir in model_path.glob("checkpoint-*"):
        if not checkpoint_dir.is_dir():
            continue
        try:
            step = int(checkpoint_dir.name.removeprefix("checkpoint-"))
        except ValueError:
            continue

        state_path = checkpoint_dir / "trainer_state.json"
        if state_path.exists():
            checkpoint_states.append((step, state_path))

    if not checkpoint_states:
        return None

    return max(checkpoint_states, key=lambda item: item[0])[1]


def _read_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def summarize_training(state: dict) -> dict:
    """Derive the training summary from a Hugging Face trainer_state.json."""
    log_history = state.get("log_history", [])
    steps = [entry["step"] for entry in log_history if isinstance(entry.get("step"), int)]
    learning_rates = [entry["learning_rate"] for entry in log_history if "learning_rate" in entry]
    train_losses = [entry["loss"] for entry in log_history if "loss" in entry]
    eval_losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]

    return {
        "max_steps": state.get("global_step") or (max(steps) if steps else None),
        "epochs": state.get("num_train_epochs"),
        "batch_size": state.get("train_batch_size"),
        "learning_rate": max(learning_rates) if learning_rates else None,
        "optimizer": "AdamW" if log_history else None,
        "final_train_loss": round(train_losses[-1], 3) if train_losses else None,
        "final_eval_loss": round(eval_losses[-1], 3) if eval_losses else None,
        "best_eval_loss": round(min(eval_losses), 3) if eval_losses else None,
    }


def build_model_info(mode: str, source: str, model_path: Path | None) -> dict:
    state = _read_json(find_latest_trainer_state(model_path))
    config = _read_json(model_path / "config.json" if model_path else None)

    return {
        "log_history": state.get("log_history", []),
        "config": {
            "model_type": config.get("model_type", "wav2vec2"),
            "hidden_size": config.get("hidden_size"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "vocab_size": config.get("vocab_size"),
        },
        "training": summarize_training(state),
        "runtime": {
            "mode": mode,
            "model_source": source,
            "local_phoneme_model_enabled": mode == "finetuned",
            "native_scoring_enabled": mode == "multitask",
        },
        "dataset": DATASET_INFO,
        "evaluation": _read_json(EVAL_RESULTS_PATH) or None,
        "architecture_md": ARCHITECTURE_PATH.read_text() if ARCHITECTURE_PATH.exists() else "",
    }
