import json

from pronunciation.model_info import build_model_info, find_latest_trainer_state, summarize_training

STATE = {
    "global_step": 4710,
    "num_train_epochs": 30,
    "train_batch_size": 8,
    "log_history": [
        {"step": 50, "loss": 62.5, "learning_rate": 1.5e-05},
        {"step": 100, "loss": 20.0, "learning_rate": 3e-05},
        {"step": 200, "eval_loss": 3.86},
        {"step": 4600, "eval_loss": 1.01},
        {"step": 4700, "loss": 1.5554, "learning_rate": 1e-07},
    ],
}


def test_summarize_training_reads_real_values():
    summary = summarize_training(STATE)
    assert summary["max_steps"] == 4710
    assert summary["epochs"] == 30
    assert summary["batch_size"] == 8
    assert summary["learning_rate"] == 3e-05
    assert summary["final_train_loss"] == 1.555
    assert summary["best_eval_loss"] == 1.01


def test_summarize_training_without_logs_claims_nothing():
    summary = summarize_training({})
    assert all(value is None for value in summary.values())


def test_find_latest_trainer_state_prefers_root_then_newest_checkpoint(tmp_path):
    assert find_latest_trainer_state(None) is None
    assert find_latest_trainer_state(tmp_path) is None

    for step in (200, 4600, 1000):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text("{}")
    (tmp_path / "checkpoint-final").mkdir()
    assert find_latest_trainer_state(tmp_path) == tmp_path / "checkpoint-4600" / "trainer_state.json"

    (tmp_path / "trainer_state.json").write_text("{}")
    assert find_latest_trainer_state(tmp_path) == tmp_path / "trainer_state.json"


def test_build_model_info_from_artifacts(tmp_path):
    (tmp_path / "trainer_state.json").write_text(json.dumps(STATE))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "wav2vec2", "vocab_size": 71, "hidden_size": 768}))

    info = build_model_info("finetuned", "results/finetuned-phoneme-model", tmp_path)
    assert info["config"]["vocab_size"] == 71
    assert info["training"]["max_steps"] == 4710
    assert len(info["log_history"]) == 5
    assert info["runtime"] == {
        "mode": "finetuned",
        "model_source": "results/finetuned-phoneme-model",
        "local_phoneme_model_enabled": True,
        "native_scoring_enabled": False,
    }


def test_build_model_info_fallback_has_no_training_claims():
    info = build_model_info("asr-fallback", "facebook/wav2vec2-base-960h", None)
    assert info["log_history"] == []
    assert info["training"]["max_steps"] is None
    assert info["runtime"]["local_phoneme_model_enabled"] is False
