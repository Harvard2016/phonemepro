# PhonemePro

Pronunciation practice that teaches accents, not just words. Pick American, British or Australian English, hear a real person say the word, say it yourself, and see which sounds matched that accent's pronunciation.

Everything runs in your browser. A fine-tuned wav2vec 2.0 model transcribes your recording into phonemes on your own device, so there is no server, no account, and your voice is never uploaded.

- **Three accent targets.** "water" is `W AO T ER` in American and `W AO T AH` in British; "dance" has the short *a* in American and Australian and the long one in British. 6,326 of the 14,541 dictionary words differ between American and British.
- **Human reference voices.** 181 recordings by volunteers on Wikimedia Commons cover the 80 practice words (American 80, British 57, Australian 44). Where there is no recording yet, the app says so and falls back to the browser's synthetic voice.
- **Feedback per sound.** Substituted and missed sounds are marked, with the slice of your recording for each one playable on its own, and accent-specific coaching ("British English has no r here").
- **A guess at your accent.** An accent head says whether a take sounds closest to Southern English, American or Scottish.
- **Your own text.** Any word or phrase of up to eight words from the dictionary.
- **History that stays with you.** Attempts are stored in the browser (IndexedDB), survive refreshes and closed tabs, and can be saved to a file and imported elsewhere.

## Credits

PhonemePro started as a team project by **Arshpreet Singh Sandhu, Prakhar Verma, Abhisar Anand and Harshit Sharma**: the original FastAPI backend, the wav2vec 2.0 fine-tuning on SpeechOcean762, and the first practice interface. Harshit continued it from there: the evaluation, the learned score head, the accent dictionaries and accent head, the in-browser model, and this interface.

## Results

### Phoneme model

Measured on the SpeechOcean762 test split: 2,500 utterances, 125 speakers, 47,369 phones, 161 minutes of audio.

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
- `ZH` and `OY` occur about 20 times each in the test set and the model never predicts either. The app marks those two sounds "not scored" instead of counting them as errors.

### Accents

Measured on native speakers from VCTK that no part of the model trained on.

**Does the phoneme model hear accent?** Its transcription of each group is compared with the American dictionary and the British one. A lower error rate against a group's own dictionary means the model hears the difference.

| Speakers (held out) | vs. American dictionary | vs. British dictionary |
|---|---|---|
| Southern English (4) | 23.6% | **22.6%** |
| American (4) | **20.9%** | 22.7% |
| Scottish (3) | **23.3%** | 24.5% |
| Irish (2) | **24.0%** | 24.6% |
| Australian (2) | **24.9%** | 25.2% |

The direction is right for the accents that have their own dictionary, and Scottish and Irish, which pronounce their r's, sit closer to American. The gaps are about one to two points, so this is a modest effect. The two Australian speakers do not match the British dictionary better, which means the rule-derived Australian targets below are not validated by this data.

**Accent head.** A small classifier on an earlier encoder layer guesses which of three accents a recording is closest to. Chance is 33%.

| | Accuracy on held-out speakers |
|---|---|
| One-second clips | **70%** |
| Whole sentences | **79%** |
| Speakers named correctly by majority vote | 10 of 11 |

Limits worth knowing:

- Eleven test speakers is a small sample, and VCTK is adults reading sentences in a studio. A learner saying one word into a laptop microphone is harder, so the app presents the guess as a hint.
- The last encoder layer, tuned to phonemes, scored 38% on a four-way version of this task; layer 6 was chosen on separate development speakers.
- Irish was tried as a fourth class and failed outright (3 of 120 held-out clips) with five training speakers. Australian has only two speakers in VCTK. Neither is guessed. The data has no Cockney or Southern American speakers in useful numbers.

**Fine-tuning the phoneme output layer on accent data** (`scripts/finetune_accent_ctc.py`) was tested and not applied. Teaching it southern English speech with British labels trades against learner speech:

| Setting | Southern English error | American error | Learner error (SpeechOcean762) |
|---|---|---|---|
| Before | 23.5% | 20.8% | 21.0% |
| Gentle | 22.7% | 20.6% | 21.0% |
| Medium | 21.8% | 20.2% | 22.2% |
| Strong | 20.1% | 18.6% | 22.6% |

The gentle setting is safe but barely moves anything; the others cost learners more than a point. Training on both kinds of speech together is the next thing to try.

### Browser model

The model is exported to ONNX with the transformer's weights quantized to 8 bits: 123 MB, downloaded once and cached. On 300 test utterances it scores 23.43% phoneme error against 23.37% at full precision. Quantizing the convolutional front end too would save 27 MB but raised the error rate to 31%. A take is analysed in about half a second on a laptop.

## The accent dictionaries

| Accent | Source |
|---|---|
| General American | CMUdict |
| British (Received Pronunciation) | [Britfone](https://github.com/JoseLlarena/Britfone) 3.0.1, IPA mapped onto the model's symbols |
| General Australian | Derived from the British entry by two rules |

The model's symbols are American (ARPABET), which limits what can be taught:

- It can mark differences in **which sounds a word has**: the dropped *r* (car, water), the long *a* of bath and dance, the *y* in new and tune, and word-level differences such as schedule, tomato, either and lieutenant.
- It cannot mark differences in **how one sound is coloured**. British *hot* and *father* share a symbol, so the British short *o* is not taught.
- The Australian entries follow the British ones except that the short *a* is kept before a nasal (dance, chance, plant, but not can't) and unstressed short *i* becomes schwa (rabbit, boxes). These rules come from published descriptions of Australian English, not from data, and the result above does not confirm them.

Scoring compares phonemes with lexical stress stripped, because SpeechOcean762 and CMUdict label stress differently.

## Run it

Requires Node 20+.

```bash
cd web
npm install
npm run dev        # http://localhost:5173
```

The model, dictionaries and reference recordings are in `web/public/`, so there is nothing else to download.

### Deploy

The app is a static site. On Vercel, import the repository, set the root directory to `web`, and deploy; `web/vercel.json` adds the single-page routing and the headers that let the model use several threads. Any static host works.

### History without a server

History lives in the visitor's browser. It survives refreshes, closed tabs and restarts. It does not follow them to another device or survive clearing site data, so the History page has "Save a backup" and "Import a backup". Syncing across devices would need accounts and a database, which this project deliberately avoids.

## Rebuild the model

Requires Python 3.12 and ffmpeg. Data and weights are not in git except the exported browser model.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt -r requirements-dev.txt

python train_finetune.py                        # phoneme model (GPU recommended)
python -m scripts.evaluate infer                # test-split evaluation
python -m scripts.evaluate report
python -m scripts.train_score_head features     # learned score head
python -m scripts.train_score_head train
python -m scripts.build_lexicon                 # accent dictionaries
python -m scripts.fetch_reference_audio         # human recordings from Wikimedia Commons
python -m scripts.accent_data index             # VCTK accent subset
python -m scripts.accent_data features
python -m scripts.accent_data layers
python -m scripts.train_accent_head             # accent head and the accent tables above
python -m scripts.export_onnx                   # browser model into web/public/model/
```

The PyTorch weights for the phoneme model, score head and accent head are attached to the GitHub release. The score head and accent head keep the encoder frozen and train in seconds on a laptop from cached features.

## Layout

```
web/                    the app (React + Vite)
  src/engine/           in-browser model, CTC decoding, scoring, dictionaries, history
  public/model/         exported model, vocabulary, evaluation report
  public/lexicon/       accent dictionaries and the practice word bank
  public/audio/         human reference recordings and their credits
pronunciation/          Python reference: scoring, dictionaries, model loading
scripts/                evaluation, heads, dictionaries, reference audio, export
tests/                  pytest suite and the fixtures shared with the JavaScript tests
api.py, db.py           optional FastAPI server with the same scoring
src/, train_*.py        original training code
Docs/                   requirements, architecture notes, evaluation results
```

Scoring exists twice, in Python (`pronunciation/scoring.py`) and JavaScript (`web/src/engine/scoring.js`). Both are tested against the same 72 cases in `tests/fixtures/scoring_cases.json`.

## Development

```bash
pytest && ruff check .               # Python
cd web && npm test && npm run lint   # JavaScript
```

CI runs both, builds the site and the Docker images, and boots the optional API container.

### Optional API server

`api.py` serves the same analysis over HTTP for anyone who would rather run the model server-side. The web app does not use it.

```bash
pip install -r requirements.txt && python download_nltk.py
uvicorn api:app
curl -X POST http://localhost:8000/api/analyze -F "audio=@recording.wav" -F "word=water" -F "accent=rp"
```

## Data, voices and licences

Code: MIT, see [LICENSE](LICENSE).

| Source | Used for | Licence |
|---|---|---|
| [SpeechOcean762](https://www.openslr.org/101/) | Phoneme model and score head | CC BY 4.0 |
| [VCTK 0.92](https://datashare.ed.ac.uk/handle/10283/3443) | Accent head and accent evaluation | CC BY 4.0 |
| [wav2vec 2.0 base](https://huggingface.co/facebook/wav2vec2-base) | Pretrained encoder | Apache 2.0 |
| CMUdict | American pronunciations | BSD-style |
| [Britfone](https://github.com/JoseLlarena/Britfone) | British pronunciations | MIT |
| Wikimedia Commons | Reference recordings | CC BY, CC BY-SA, CC0 or public domain, per file |

Each recording's speaker, licence and source page is listed in `web/public/audio/credits.json` and on the app's Credits page. The recordings were converted to MP3, trimmed and loudness-normalized; those under a ShareAlike licence remain under it.
