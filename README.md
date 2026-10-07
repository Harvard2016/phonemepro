# PhonemePro

Pronunciation practice with phoneme-level feedback. Say a word, and a fine-tuned wav2vec 2.0 model transcribes what you actually said into ARPABET phonemes, aligns it against the dictionary pronunciation, and marks each sound as correct, substituted, or missed.

- **Model:** `facebook/wav2vec2-base` fine-tuned with a CTC head on [SpeechOcean762](https://www.openslr.org/101/) (2,500 training utterances from English learners), plus a small score head trained against human accuracy ratings.
- **Backend:** FastAPI. Alignment, scoring, per-user weak-sound tracking, SQLite history.
- **Frontend:** React + Vite. Live waveform while recording, IPA transcription, proofreader-style error marks, and a timeline of your take where each sound can be replayed on its own.
- **Practice anything:** a built-in word list at three levels, or type your own word or short phrase.

## Results

Measured on the SpeechOcean762 test split: 2,500 utterances, 125 speakers, 47,369 phones, 161 minutes of audio. Reproduce with `python -m scripts.evaluate infer && python -m scripts.evaluate report`.

| Metric | Value |
|---|---|
| Phoneme error rate (stress ignored) | **19.3%** |
| Phoneme error rate (with lexical stress) | 21.2% |
| App score vs. human accuracy rating (Pearson r, per utterance) | **0.69** with the learned score head, 0.50 from alignment alone |
| Mean absolute error of the app score | 0.82 points on the 0 to 10 scale |
| Mispronunciation detection recall / precision | 75% / 30% |
| False alarm rate on well-pronounced phones | 13% |

How to read these:

- The reference is the dictionary pronunciation and the speakers are learners, so some of the error rate is real mispronunciation, not model error.
- A phone counts as mispronounced when human annotators scored it below 1.5 on the dataset's 0 to 2 scale. The model flags many more phones than annotators do, so a flagged sound is a prompt to listen again rather than a verdict.
- The test split was also used to log eval loss during training. No checkpoint selection or tuning was done against it, but it is not a fully untouched hold-out.
- `ZH` and `OY` occur about 20 times each in the test set and the model never predicts either. Both are rare in the training data. The app marks those two sounds "not scored" instead of counting them as errors.

Full results, including the per-phoneme breakdown and the most common confusions, are in [`Docs/model/eval_results.json`](Docs/model/eval_results.json) and on the app's Model page.

### The learned score head

Alignment tells you which sounds were wrong, but it is a weak predictor of how a human would rate the whole utterance (r = 0.50). `scripts/train_score_head.py` keeps the fine-tuned encoder frozen, caches one mean-pooled embedding per utterance, and fits a two-layer regression head on the human accuracy ratings of the training split, with every fifth training speaker held out for early stopping. It trains in seconds on a laptop and reaches r = 0.65 on the test split on its own.

The head only hears audio, not the target word, so a fluent rendition of the wrong word would rate highly. The app score is therefore a blend (60% learned, 40% alignment, weight chosen on the held-out training speakers) capped at three points above the alignment score. That blend reaches r = 0.69.

```bash
python -m scripts.train_score_head features   # about 10 minutes on CPU
python -m scripts.train_score_head train      # writes results/multitask-phoneme-model
```

### Why scoring ignores lexical stress

SpeechOcean762 labels stress differently from CMUdict (monosyllables are labelled `AA0` where CMUdict has `AA1`), and the practice words use CMUdict targets. Comparing stress digits therefore penalised learners for a labelling convention. Scoring now aligns stress-stripped phonemes and reports stress agreement separately.

## Quick start

Requires Python 3.12 and Node 20+.

```bash
# Backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_nltk.py
uvicorn api:app --reload          # http://localhost:8000

# Frontend, in a second terminal
cd web
npm install
npm run dev                       # http://localhost:5173
```

### Model weights

Weights are not stored in git. The backend picks the first source that exists:

1. `results/multitask-phoneme-model` (CTC plus the learned score head, built by `scripts/train_score_head.py`)
2. `results/finetuned-phoneme-model` (the model evaluated above)
3. The Hugging Face Hub repo named by the `PHONEME_MODEL_ID` environment variable
4. `facebook/wav2vec2-base-960h`, a general speech recognizer. Scores in this mode are rough, and the app labels it "Fallback model".

Download the fine-tuned model from Google Drive: [Finetuned Phoneme Model.zip](https://drive.google.com/file/d/1utPIaTqvlH2NufQiIz2xrYJ_GknTjcJv/view?usp=sharing) (2.4 GB with training checkpoints). Unzip it at the repository root so that `results/finetuned-phoneme-model/model.safetensors` exists. Only the top-level files are needed for inference; copy `checkpoint-4710/trainer_state.json` next to them to get the training curve on the Model page.

### Docker

```bash
docker compose up --build
```

The API is served on port 8000 and the web app on port 5173. `./results` is mounted read-only into the API container.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/analyze` | Score a recording. Form fields: `audio` (WAV or WebM), `word`, optional `phonemes` (JSON list, for words outside the built-in list). |
| `GET` | `/api/words?level=easy` | Practice words with ARPABET targets. |
| `GET` | `/api/phonemes?text=red lorry` | Dictionary phonemes for a typed word or phrase (up to 8 words). |
| `GET` | `/api/insights` | Per-phoneme error rates from your attempts and words that train the weakest sounds. |
| `GET` | `/api/history`, `/api/history/export` | Attempts as JSON or CSV. |
| `DELETE` | `/api/history?source=real` | Clear attempts. |
| `GET` | `/api/model-info` | Training log, run configuration, evaluation results. |
| `GET` | `/api/health` | Liveness check. |

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "audio=@recording.wav" -F "word=ship"
```

```json
{
  "word": "ship",
  "target_phonemes": ["SH", "IH1", "P"],
  "predicted_phonemes": ["S", "IH1", "P"],
  "score": 6.8,
  "wer": 0.33,
  "metrics": {"accuracy": 67, "completeness": 100, "fluency": 91},
  "phoneme_results": [
    {"phoneme": "SH", "status": "error", "heard": "S", "tip": "Heard S instead.", "start": 0.1, "end": 0.24},
    {"phoneme": "IH1", "status": "correct", "heard": "IH1", "tip": "", "stress_match": true, "start": 0.24, "end": 0.4},
    {"phoneme": "P", "status": "correct", "heard": "P", "tip": "", "start": 0.4, "end": 0.52}
  ],
  "feedback": "Focus on the highlighted sounds."
}
```

`start` and `end` are approximate seconds into the recording, taken from where the CTC head emitted each phoneme. The numbers above are illustrative.

## Project layout

```
api.py                  FastAPI routes
pronunciation/
  model.py              model selection, loading, CTC decoding
  scoring.py            alignment, scoring, per-phoneme feedback (pure functions)
  phonemes.py           ARPABET helpers and articulation tips
  model_info.py         training and evaluation metadata read from artifacts
db.py                   SQLite practice history
word_list.py            practice words by difficulty
scripts/evaluate.py     test-split evaluation
scripts/train_score_head.py   score head on the frozen encoder
src/, train_*.py        training code
tests/                  pytest suite (no model weights needed)
web/                    React frontend
Docs/                   product requirements, architecture notes, evaluation results
```

## Development

```bash
pip install -r requirements-dev.txt
pytest              # backend tests, run against a fake recognizer
ruff check .

cd web
npm run lint
npm run build
```

CI runs the same checks on every pull request, builds both Docker images, and boots the API container.

## Training

`train_finetune.py` expects SpeechOcean762 extracted at `data/speechocean762` and writes to `results/finetuned-phoneme-model`. Install `requirements-train.txt` first.

```bash
python train_finetune.py
```

The published model ran for 30 epochs (4,710 steps) at batch size 8 with gradient accumulation 2 and a peak learning rate of 3e-5.
