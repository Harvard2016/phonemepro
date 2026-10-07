import csv
import io
import json

import db


def post_attempt(client, wav_bytes, word="ship", phonemes=None):
    return client.post(
        "/api/analyze",
        files={"audio": ("recording.wav", wav_bytes, "audio/wav")},
        data={"word": word, "phonemes": json.dumps(phonemes or [])},
    )


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok", "model_loaded": True}


def test_words_filtered_by_level(client):
    easy = client.get("/api/words", params={"level": "easy"}).json()
    assert easy and all(word["difficulty"] == "easy" for word in easy)
    assert len(client.get("/api/words").json()) > len(easy)
    assert [level["id"] for level in client.get("/api/levels").json()] == ["easy", "medium", "hard"]


def test_analyze_perfect_attempt_is_saved(client, recognizer, wav_bytes):
    recognizer.phonemes = ["SH", "IH1", "P"]
    response = post_attempt(client, wav_bytes)
    assert response.status_code == 200
    body = response.json()
    assert body["word"] == "ship"
    assert body["score"] == 10.0
    assert body["predicted_phonemes"] == ["SH", "IH1", "P"]

    history = client.get("/api/history").json()
    assert [item["word"] for item in history] == ["ship"]
    assert history[0]["source"] == "real"
    assert client.get("/api/history/today-count").json() == {"count": 1}


def test_analyze_uses_server_phonemes_for_known_words(client, recognizer, wav_bytes):
    recognizer.phonemes = ["SH", "IH1", "P"]
    body = post_attempt(client, wav_bytes, word="Ship", phonemes=["Z", "Z"]).json()
    assert body["target_phonemes"] == ["SH", "IH1", "P"]


def test_analyze_custom_word_uses_client_phonemes(client, recognizer, wav_bytes):
    recognizer.phonemes = ["K", "AE1", "T"]
    body = post_attempt(client, wav_bytes, word="cat", phonemes=["K", "AE1", "T"]).json()
    assert body["target_phonemes"] == ["K", "AE1", "T"]
    assert body["score"] == 10.0


def test_analyze_rejects_bad_input(client, wav_bytes):
    assert post_attempt(client, b"").status_code == 400
    assert post_attempt(client, b"not audio at all").status_code == 400
    assert post_attempt(client, wav_bytes[:44 + 3200]).status_code == 400  # 0.1 s
    bad_phonemes = client.post(
        "/api/analyze",
        files={"audio": ("recording.wav", wav_bytes, "audio/wav")},
        data={"word": "zzyzx", "phonemes": "not json"},
    )
    assert bad_phonemes.status_code == 422
    assert client.get("/api/history").json() == []


def test_analyze_nothing_recognized(client, recognizer, wav_bytes):
    recognizer.phonemes = []
    body = post_attempt(client, wav_bytes).json()
    assert body["score"] == 0
    assert body["predicted_phonemes"] == []


def test_clear_history_by_source(client, recognizer, wav_bytes):
    recognizer.phonemes = ["SH", "IH1", "P"]
    post_attempt(client, wav_bytes)
    db.save_attempt("thin", ["TH", "IH1", "N"], ["TH", "IH1", "N"], 9.0, 0.0, [], source="sample")

    client.delete("/api/history", params={"source": "real"})
    assert [item["source"] for item in client.get("/api/history").json()] == ["sample"]

    client.delete("/api/history")
    assert client.get("/api/history").json() == []


def test_export_csv_quotes_fields(client):
    db.save_attempt('say "hi", ok', ["S", "EY1"], ["S", "EY1"], 9.5, 0.0, [])
    response = client.get("/api/history/export")
    assert response.headers["content-type"].startswith("text/csv")
    rows = list(csv.reader(io.StringIO(response.text)))
    assert rows[0][0] == "word"
    assert rows[1][0] == 'say "hi", ok'
    assert rows[1][3] == "S EY1"


def test_insights_rank_weak_phonemes_and_recommend_words(client, recognizer, wav_bytes):
    # "thin" twice with TH replaced by S, then "ship" twice cleanly.
    recognizer.phonemes = ["S", "IH1", "N"]
    post_attempt(client, wav_bytes, word="thin")
    post_attempt(client, wav_bytes, word="thin")
    recognizer.phonemes = ["SH", "IH1", "P"]
    post_attempt(client, wav_bytes, word="ship")
    post_attempt(client, wav_bytes, word="ship")

    insights = client.get("/api/insights").json()
    assert insights["attempts"] == 4
    assert [item["phoneme"] for item in insights["weak_phonemes"]] == ["TH"]
    assert insights["weak_phonemes"][0] == {"phoneme": "TH", "attempts": 2, "errors": 2, "error_rate": 1.0}

    ih = next(item for item in insights["phonemes"] if item["phoneme"] == "IH")
    assert ih == {"phoneme": "IH", "attempts": 4, "errors": 0, "error_rate": 0.0}

    recommended = insights["recommended_words"]
    assert recommended and all("TH" in word["focus"] for word in recommended)
    assert all(any(p.rstrip("012") == "TH" for p in word["phonemes"]) for word in recommended)


def test_insights_ignore_sample_rows(client):
    db.save_attempt("thin", ["TH", "IH1", "N"], ["S", "IH1", "N"], 5.0, 0.33,
                    [{"type": "substitute", "expected": ["TH"], "heard": ["S"], "tip": ""}], source="sample")
    insights = client.get("/api/insights").json()
    assert insights["attempts"] == 0
    assert insights["weak_phonemes"] == []


def test_model_info_shape(client):
    info = client.get("/api/model-info").json()
    assert info["runtime"]["mode"] == "finetuned"
    assert info["runtime"]["local_phoneme_model_enabled"] is True
    assert info["dataset"]["name"] == "Speechocean762"
    assert set(info["training"]) >= {"max_steps", "learning_rate", "batch_size", "epochs"}


def test_analyze_returns_audio_span_for_heard_phonemes(client, recognizer, wav_bytes):
    recognizer.phonemes = ["SH", "P"]
    recognizer.spans = [(0.1, 0.24), (0.24, 0.36)]
    results = post_attempt(client, wav_bytes).json()["phoneme_results"]
    assert (results[0]["start"], results[0]["end"]) == (0.1, 0.24)
    assert "start" not in results[1]  # IH1 was missed, so there is nothing to play
    assert (results[2]["start"], results[2]["end"]) == (0.24, 0.36)


def test_phonemes_for_known_word_uses_curated_entry(client):
    body = client.get("/api/phonemes", params={"text": "  Ship! "}).json()
    assert body == {"word": "ship", "phonemes": ["SH", "IH1", "P"], "difficulty": "easy"}


def test_phonemes_for_custom_phrase(client):
    body = client.get("/api/phonemes", params={"text": "Go on, 42!"}).json()
    assert body == {"word": "go on", "phonemes": ["G", "O", "O", "N"], "difficulty": "custom"}


def test_phonemes_rejects_unusable_text(client):
    assert client.get("/api/phonemes", params={"text": "123 !!"}).status_code == 422
    assert client.get("/api/phonemes", params={"text": "a " * 9}).status_code == 422
    assert client.get("/api/phonemes", params={"text": "x" * 61}).status_code == 422
    assert client.get("/api/phonemes").status_code == 422
