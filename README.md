# PhonemePro

Pronunciation practice that teaches accents, not just words. Pick American, British or Australian English, hear a real person say the word, say it yourself, and see which sounds matched that accent's pronunciation.

**Try it: https://phonemepro.vercel.app**

Everything runs in your browser. A fine-tuned wav2vec 2.0 model transcribes your recording into phonemes on your own device, so there is no server, no account, and your voice is never uploaded.

- **Three accent targets.** "water" is `W AO T ER` in American and `W AO T AH` in British; "dance" has the short *a* in American and Australian and the long one in British. 6,326 of the 14,541 dictionary words differ between American and British.
- **Human reference voices.** 181 recordings by volunteers on Wikimedia Commons cover the 80 practice words (American 80, British 57, Australian 44). Where there is no recording yet, and for any text you type yourself, the app says so and uses its own synthetic voice for that accent, which runs in the browser and sounds the same on every device (see [Data, voices and licences](#data-voices-and-licences)).
- **Feedback per sound.** Substituted and missed sounds are marked, with the slice of your recording for each one playable on its own, and accent-specific coaching ("British English has no r here").
- **A guess at your accent.** An accent head says whether a take sounds closest to Southern English, American or Scottish.
- **Your own text.** Any word or phrase of up to eight words from the dictionary.
- **History that stays with you.** Attempts are stored in the browser (IndexedDB), survive refreshes and closed tabs, and can be saved to a file and imported elsewhere.
- **Takes cleaned before scoring.** The microphone stays open between takes so the first sound is not lost, and each take is trimmed and noise-reduced on the device. If there is no speech or too much noise, the app asks for another take instead of scoring it.
- **Optional, off by default: help it learn.** A visitor can say where they are from and agree to keep a 256-number summary of each clear take, labelled by whose accent it is. It is not audio, it stays in the browser, and nothing uploads it. See [What the app keeps](#what-the-app-keeps).

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

It has four outputs: phoneme logits, the learned score, the accent guess, and a 256-number accent summary (see [Learning regions from totals](#learning-regions-from-totals)). Quantization moves the summary by 0.046 on average, where each number has a spread of about 1.

### Recording cleanup

What a messy recording does to the model, and which cleanup steps help. 120 SpeechOcean762 test utterances were degraded the way a laptop take is degraded, then scored raw and after each step (`scripts/evaluate_capture.py`, `Docs/model/capture_results.json`). Phoneme error rate, lower is better:

| Recording | Raw | Rumble filter | Trim, 0.12 s margin | Trim, 0.5 s margin | Noise reduction | Trim 0.5 s + noise reduction |
|---|---|---|---|---|---|---|
| Clean | 19.6% | 19.8% | 21.5% | 19.8% | 19.7% | 20.0% |
| 0.7 s of quiet room either side | 23.2% | 23.3% | 24.2% | 23.6% | 21.8% | **21.1%** |
| Noise 25 dB below the speech | 24.3% | 24.2% | 25.1% | 24.0% | 22.8% | **21.9%** |
| Noise 15 dB below | 31.4% | 31.2% | 31.6% | 29.0% | 26.5% | **25.8%** |
| Noise 5 dB below | 59.5% | 58.7% | 59.2% | 54.2% | 40.5% | **40.1%** |
| First 100 ms missing | 20.1% | 20.1% | 21.5% | 19.9% | 19.8% | 19.8% |

What the app does as a result:

- **Trims with half a second of quiet each side, then reduces steady noise.** This is the last column. It is the best or within a point of the best on every degraded recording and costs 0.4 points on clean audio.
- **Does not trim tightly.** A 0.12 s margin made clean audio worse (19.6% to 21.5%) and moved the learned score by 1.3 points on the 0 to 10 scale, against 0.06 points for the 0.5 s margin. The training clips have quiet at both ends and the model expects it.
- **Does not filter rumble.** A 70 Hz high-pass changed nothing measurable. It is still used to find where the speech is.
- **Asks for a retake rather than scoring a bad take.** Even after cleanup the model gets four sounds in ten wrong when noise is 5 dB below the speech. A take with no speech found, or a measured signal-to-noise ratio under 10 dB, is not scored.

Limits of this measurement:

- The noise is synthetic white noise, and the 120 utterances are sentences from learners, not single words. Real rooms and real microphones will differ. The browser's own noise suppression also runs before this cleanup, which the experiment does not model.
- The cleanup steps were chosen on the same 120 utterances they are reported on.
- An earlier run on 200 utterances with a different set of variants is kept in `Docs/model/capture_results_first_run.json`. Its clean error rate (22.9%) is not comparable with the 19.6% here because the utterance sets differ. It is the only run that tested a second person talking in the background, 12 dB quieter: 84.9% phoneme error raw, 77.5% after tight trimming. Nothing tried removes a second voice. The app does not yet detect one, so such a take is scored and scored badly.

### On a real microphone

One run of the label check (below, [Checking labels by voice](#checking-labels-by-voice)) on a laptop microphone in a quiet room is saved in `Docs/model/check_run_2026-10-08.txt`. It is one speaker and one room, so it shows what can happen, not how often:

- The model dropped the first vowel of "water" (heard `W T AH0`) and doubled the T in "better" (heard `B EH1 T T AH0`).
- Clear speech scored about 7 out of 10 (6.9 to 7.3 on the four clean practice takes).
- A silent take was taken for 4.07 s of speech and scored 4.1. The silence check above was added because of it.
- With music or a fan on, the measured signal-to-noise ratio was still 33.6 dB, because the browser suppresses noise before the app sees the audio. The app's noise check will rarely catch a noisy room.
- The first sound of "fish" was heard, so the start of a take is not being clipped.

### Is a take the target word?

A practice take is kept for learning only if it is recognisably the target word. The rule has to let a strong accent through and keep a different word out, and practice words are short, so there is little to go on. Three rules were measured on model transcriptions already on disk (`scripts/evaluate_match.py`, `Docs/model/match_threshold.json`): SpeechOcean762 words that annotators scored 5 out of 10 or lower, VCTK Scottish and Irish speakers against the American dictionary, and the same transcriptions scored against a different word of the same length. Share of takes kept, by the share of target sounds required.

Sounds heard **exactly as written**:

| Single words, 3 to 5 sounds | Takes | ≥30% | ≥40% | ≥50% | ≥60% | ≥70% |
|---|---|---|---|---|---|---|
| Learners rated 5/10 or lower | 852 | 75.6% | 56.8% | 52.8% | 42.8% | 21.6% |
| Scottish speakers | 2,879 | 97.9% | 90.3% | 89.1% | 85.8% | 62.7% |
| Irish speakers | 1,848 | 97.6% | 89.3% | 87.5% | 83.8% | 61.2% |
| A different word | 13,147 | 14.8% | 3.8% | 2.9% | 2.0% | 0.2% |

Sounds heard as the **same broad class** (any vowel for a vowel, consonants within accent-swap groups such as th/t/s/f):

| Single words, 3 to 5 sounds | Takes | ≥30% | ≥40% | ≥50% | ≥60% | ≥70% |
|---|---|---|---|---|---|---|
| Learners rated 5/10 or lower | 852 | 86.7% | 77.1% | 75.0% | 67.8% | 46.4% |
| Scottish speakers | 2,879 | 98.8% | 93.6% | 93.1% | 91.6% | 79.3% |
| Irish speakers | 1,848 | 98.4% | 93.0% | 92.5% | 90.9% | 77.2% |
| A different word | 13,147 | 59.2% | 29.4% | 26.0% | 19.8% | 4.1% |

What the app does as a result:

- **Keeps a practice take when at least 30% of its target sounds are heard as written.** This keeps 76% of low-rated learner words and 98% of Scottish and Irish words, and lets through 15% of wrong words. So 24% of genuine, heavily accented learner words are still dropped, and about 2% of native Scottish and Irish ones.
- **Does not use the class-based match to decide.** It was expected to be kinder to accents, and it is, but it lets wrong words through much faster: at 60% it keeps 68% of low-rated learner words and 20% of wrong words, where the exact rule at 30% keeps more of the first and fewer of the second. Any vowel standing for any vowel is too loose on a four-sound word. The class-based share is still recorded with each take (`sounds_same_class`) so it can be judged again on real contributions.
- **Dropped two earlier checks.** Requiring 50% as written kept only 53% of low-rated learner words. Requiring "70% of sounds attempted", where any sound in the right place counted, kept 88% of wrong words and so rejected almost nothing it was meant to.
- **Never compares a normal-voice take with a dictionary.** There is no right way to sound in your own voice. Such a take only has to be between half and twice as long, in sounds heard, as the sentence shown.

Limits of this measurement:

- VCTK's three Indian speakers are not cached on disk, so Indian English was not measured.
- Words were cut out of sentences by aligning the transcription to the dictionary, not by time, so sounds near word boundaries can land on the wrong word. This makes genuine words look worse than a single recorded word would.
- A low human score does not always mean a strong accent; some of those words were simply misread.
- Wrong words that share sounds with the target ("butter" for "water") pass. With 3-sound targets 21% of wrong words pass, against 4% for 4-sound and 8% for 5-sound targets.
- On whole utterances the rules are much easier to tell apart: at 30% as written, 95% of low-rated utterances are kept and 4% of wrong ones.

### Silence, length and confidence

Three more checks, measured on speech already on disk (`scripts/evaluate_take_checks.py`, `Docs/model/take_checks.json`). "Single words" are the 181 human reference recordings run through the model as the app would capture them; they are the closest thing on disk to a practice take, and all are native speakers.

**Silence.** The speech finder works on loudness, and on a real microphone a quiet room was taken for four seconds of speech. So the model has the last word: fewer than 2 sounds, or fewer than 1 sound per second of "speech", is treated as silence, with no score, a retake notice and nothing stored. Sounds per second on real speech:

| | Takes | Slowest 1% | Median | Called silence |
|---|---|---|---|---|
| SpeechOcean762, all | 2,500 | 2.3 | 6.8 | 0.1% |
| SpeechOcean762, rated 5/10 or lower | 257 | 1.1 | 4.1 | 1.2% |
| VCTK Scottish | 840 | 5.7 | 11.0 | 0.0% |
| VCTK Irish | 540 | 5.4 | 11.1 | 0.0% |
| Single words | 181 | 2.3 | 7.7 | 0.5% |

Most real speech is well clear of 1 sound per second. The slowest low-rated learners are not: their slowest 1% sits at 1.1, and 1.2% of their utterances would be called silence.

**Length.** A practice take is kept only if the sounds heard number between 0.5 and 1.5 times the target's; more than that usually means background noise was transcribed around the word, and the learner is asked to try again. Share of genuine takes this drops:

| | Takes | Too few sounds | Extra sounds |
|---|---|---|---|
| Single words | 181 | 0.5% | 0.5% |
| SpeechOcean762 words, rated 5/10 or lower | 852 | 5.2% | 1.2% |
| VCTK Scottish words | 2,879 | 3.2% | 0.0% |
| VCTK Irish words | 1,848 | 4.3% | 0.0% |

The words cut from sentences overstate "too few", because sounds near a word boundary can be assigned to its neighbour. This check is cheap for genuine takes but is not a strong noise filter: with synthetic noise 10 dB below the speech it caught extra sounds in only 0.5% of single words, because noise mostly makes the model drop sounds rather than add them. The real take that prompted it (six sounds heard for the four of "water") sits exactly on the limit and would still be kept.

**Confidence.** The cutoff was 0.6. Share of takes below each cutoff:

| | Takes | < 0.5 | < 0.55 | < 0.6 |
|---|---|---|---|---|
| SpeechOcean762 utterances, rated 5/10 or lower | 257 | 1.6% | 1.6% | 1.6% |
| VCTK Scottish utterances | 840 | 0.0% | 0.0% | 0.0% |
| VCTK Irish utterances | 540 | 0.0% | 0.0% | 0.0% |
| Single words, quiet room | 181 | 1.1% | 2.8% | 11.6% |
| Single words, noise 10 dB below | 181 | 7.7% | 15.5% | 30.4% |
| Single words, noise 5 dB below | 181 | 16.0% | 24.9% | 36.5% |

Sentences are far more confident than single words, and it is single words the app records. At 0.6 the check dropped 11.6% of clean single words from native speakers, so the cutoff is now 0.5. Confidence is a weak filter either way: it does not depend on the target, so it cannot tell a wrong word from the right one, and most noisy takes clear it. There are no accented single-word recordings on disk, so the effect on accented single words is not measured.

### Learning regions from totals

Can the app learn what a region sounds like without keeping anyone's recordings, or even a row per person? Each take is reduced to 256 numbers: the mean and spread of encoder layer 6 over the take, standardized and projected onto fixed directions fitted on VCTK training speakers. For each region only three things are kept: a count, the sum of the vectors, and the sum of their outer products. A vector is added to the totals and dropped. Those totals are exactly what a linear discriminant needs.

Measured on the same held-out VCTK speakers as the accent head (`python -m scripts.learn_regions simulate`, the `region_totals` section of `Docs/model/eval_results.json`):

| | From totals alone | Trained accent head |
|---|---|---|
| Whole sentences | 78.0% | 78.6% |
| One-second clips | 66.4% | 70.3% |

Storage is fixed at 514 KB per region however many people contribute. This has only been tested with three accents and native speakers reading sentences. It has not been tested on contributions from the app, because sharing is not switched on and there are none.

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

The live site is https://phonemepro.vercel.app, deployed from `main` by Vercel with the root directory set to `web`. The app is a static site. On Vercel, import the repository, set the root directory to `web`, and deploy; `web/vercel.json` adds the single-page routing and the headers that let the model use several threads. Any static host works.

### What the app keeps

Nothing a visitor records leaves their device. There is no upload endpoint.

- **Recordings** are held in memory for playback and dropped when the visitor moves on.
- **Practice history** (words, scores, sounds heard) is in IndexedDB.
- **A profile**, if the visitor fills it in on the first-visit screen: country, a broad region for the few countries that offer one, and the "Help it learn" switch, which is off unless they turn it on. It is in `localStorage` and can be reopened from "Your profile" in the footer to change or withdraw.
- **Contributions**, only while the switch is on and a country is given, and only for takes that pass every check: speech found, signal-to-noise ratio of at least 20 dB, under 0.5% of samples at full scale, model confidence of at least 0.5, about the right number of sounds (0.5 to 1.5 times the target's for a practice take, 0.5 to 2 times for a normal-voice sentence), and, for a practice take, at least 30% of the target sounds heard as written (see [Is a take the target word?](#is-a-take-the-target-word) and [Silence, length and confidence](#silence-length-and-confidence)). At most 60 are kept per browser. Each is:

| Field | Meaning |
|---|---|
| `id`, `created` | Random id and time of the take |
| `contributor` | Random id for the browser. Replaced when everything is deleted |
| `country`, `region` | From the profile. `region` is a short code from a fixed list or empty, never typed text |
| `target_accent` | `ga`, `rp` or `au`: the accent being practised. Empty for a normal-voice take |
| `natural_voice` | True for the five "say these in your normal voice" sentences on the profile screen |
| `matches_home_accent` | True when the accent practised is the standard one of the country (US and `ga`, GB and `rp`, AU and `au`). A label only |
| `word`, `score` | What was said and how it scored |
| `features` | The 256 numbers. Never audio |
| `quality` | Seconds of speech, signal-to-noise ratio, clipping, model confidence, number of sounds heard, and for practice takes the share heard as written and the share in the same sound class |
| `model_version` | Summaries from different models cannot be mixed |

The labels matter because a person imitating an accent is not a sample of their home accent. `scripts/learn_regions.py ingest` routes each take accordingly, and the Your data page shows the same routing for every stored take:

| Take | Joins |
|---|---|
| Normal voice (`natural_voice`) | Native totals: the country's, and the region's when one was given (`GB` and `GB-SCT`) |
| Any practice take | Attempt totals, per country and target accent (`GB>rp`, `GB>ga`, `US>ga`) |

Only normal-voice takes are native. Someone from Scotland practising standard British, or an American practising General American, is aiming at a standard, so the take is an attempt like any other. `matches_home_accent` is stored as a label and plays no part in routing. To give one person more native data, the profile screen offers five short sentences to read in a normal voice, all optional.

Regions are a dropdown of broad areas, offered only where accent varies a lot within the country. A small town plus a voice summary could point at one person, so there is nowhere to type one, and ingest refuses any value not in this list:

| Country | Regions |
|---|---|
| United States | Northeast, South, Midwest, West |
| United Kingdom | England (South), England (North), Scotland, Wales, Northern Ireland |
| Australia | Eastern states and Tasmania, South Australia, Western Australia and the Northern Territory |
| Canada | Atlantic provinces, Quebec, Ontario, Prairies, British Columbia and the North |
| Ireland | East (Leinster), South (Munster), West and north (Connacht, Ulster) |
| India | North, South, East and Northeast, West and Central |

Every other country has no region field, and "Rather not say" is always an option. The lists outside the US and UK are a first judgment, not drawn from data. A region can be learned once enough people from it record the normal-voice sentences. Attempts are kept per country, never per region.

The **Your data** page (`/data`) lists every stored take with all of its labels, shows why each recent take was or was not kept, exports to a JSON file and deletes everything. To see how an export would be used:

```bash
python -m scripts.learn_regions ingest phonemepro-contributions-2026-10-08.json --dry-run
```

This checks every take and prints a per-country summary of what would be added and what would be refused. Without `--dry-run` the valid takes are folded into `results/region_totals.npz` and `results/attempt_totals.npz`, so leave the flag on for test exports. The 256 numbers cannot be played back but may still be characteristic of a voice, so they are treated as personal data. The notice at `/privacy` is interim and has not had legal review.

### Running a pilot

The app has no upload, so the first real contributions come by hand. Send people https://phonemepro.vercel.app/pilot (it is not linked from the site). It walks them through setting a profile, reading the five normal-voice sentences, practising a few words, exporting from Your data, and sending the file back to you.

Keep the files you receive in a private folder outside git, such as `data/contributions/`, and after each new file:

```bash
python -m scripts.learn_regions ingest data/contributions --rebuild   # totals from exactly these files
python -m scripts.learn_regions reference                             # once: native reference accents from VCTK
python -m scripts.learn_regions report
```

- `--rebuild` replaces the totals with what the files in the folder hold. People are then counted correctly across files, and if someone asks to withdraw, deleting their file and rebuilding removes them completely. Without it, totals only ever grow and a take cannot be taken back out.
- `report` lists normal-voice takes by country and region, says which places are ready (200 takes from 8 people, defaults that have not been tuned) and which of those can be told apart. For each "country attempting accent" group it gives the average distance to native speakers of American and British, taken from VCTK, and whether the group sits closest to its target.

What this can and cannot show yet:

- The distance is computed from the totals alone, so it describes a group, not a take. Nothing is shown to learners in the app yet.
- With a handful of takes the averages mostly reflect which words were said and who said them.
- The reference accents come from VCTK sentences run through the full-precision model, while contributions are single words and sentences run through the quantized browser model. That mismatch has not been measured.
- There is no Australian reference: VCTK has two Australian speakers.
- Sending a file is the consent. The privacy notice describes the pilot but has had no legal review.

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
python -m scripts.learn_regions fit-projection  # the 256-number accent summary
python -m scripts.learn_regions simulate        # region learning from totals, on VCTK
python -m scripts.evaluate_capture              # recording cleanup table
python -m scripts.make_cleanup_fixtures         # after changing pronunciation/audio_cleanup.py
python -m scripts.evaluate_match                # rule for "is this take the target word"
python -m scripts.evaluate_take_checks          # silence, length and confidence checks
python -m scripts.make_sound_match_fixtures     # after changing web/src/engine/sound_groups.json
python -m scripts.export_onnx                   # browser model into web/public/model/
```

The PyTorch weights for the phoneme model, score head and accent head are attached to the GitHub release. The score head and accent head keep the encoder frozen and train in seconds on a laptop from cached features.

## Layout

```
web/                    the app (React + Vite)
  src/engine/           in-browser model, CTC decoding, scoring, recording cleanup, dictionaries,
                        history, profile and contributions
  src/audio/            microphone session and capture worklet
  src/check/            the development-only label check: step table and pass/fail logic
  public/model/         exported model, vocabulary, evaluation report
  public/lexicon/       accent dictionaries and the practice word bank
  public/audio/         human reference recordings and their credits
pronunciation/          Python reference: scoring, recording cleanup, region totals, dictionaries, model loading
scripts/                evaluation, heads, dictionaries, reference audio, export
tests/                  pytest suite and the fixtures shared with the JavaScript tests
api.py, db.py           optional FastAPI server with the same scoring
src/, train_*.py        original training code
Docs/                   requirements, architecture notes, evaluation results
```

Scoring exists twice, in Python (`pronunciation/scoring.py`) and JavaScript (`web/src/engine/scoring.js`). Both are tested against the same 72 cases in `tests/fixtures/scoring_cases.json`. Recording cleanup is paired the same way (`pronunciation/audio_cleanup.py`, `web/src/engine/cleanup.js`, 17 cases in `tests/fixtures/cleanup_cases.json`), the sound classes (`pronunciation/sound_match.py`, `web/src/engine/soundMatch.js`) by 91 cases in `tests/fixtures/sound_match_cases.json`, and the contribution format and its routing by `tests/fixtures/contribution_export.json`.

## Development

```bash
pytest && ruff check .               # Python
cd web && npm test && npm run lint   # JavaScript
```

CI runs both, builds the site and the Docker images, and boots the optional API container.

### Checking labels by voice

With the dev server running, http://localhost:5173/check walks through eleven steps. Each sets up its own made-up profile, says what to say, runs the take through the same cleanup, model, scoring and keeping rules as the app, and shows pass or fail with expected against actual for every label. It ends with a summary that can be copied as plain text.

| Step | Say | Expected |
|---|---|---|
| 1 | Profile GB, target American: "water" | Kept, attempt, joins `GB>ga` |
| 2 | Profile GB, target British: "water" | Kept, attempt, joins `GB>rp` |
| 3 | Profile US, South, target American: "better" | Kept, attempt, joins `US>ga`, not `US` or `US-S` |
| 4 | Profile GB, Scotland: a sentence in a normal voice | Kept, native, joins `GB` and `GB-SCT` |
| 5 | Nothing | Not kept, with a retake notice, as no speech or as extra sounds from the room |
| 6 | "banana" when the target is "water" | Not kept: not recognisably the target word |
| 7 | "water" in a deliberately strong accent | Kept, attempt, joins `GB>ga` |
| 8 | "fish" | First sound heard is F, so the start was not clipped |
| 9 | "water" over music or a fan | Reported only: measured signal-to-noise ratio, whether it was kept, and whether a retake was asked |
| 10 | Sharing off: "water" | Scored, nothing stored |
| 11 | Nothing | Everything kept this session is valid for ingest and routed as steps 1 to 4 predicted |

The page is a sandbox. Takes go to a store in memory, so the real profile, contributions and practice history are never read or written, and everything is gone on leaving the page. It exists only in development: the route and its code are left out of `npm run build`. The expected values are one table in `web/src/check/steps.js`, and the pass or fail logic is unit-tested without a microphone in `web/src/check/check.test.js`.

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
| [Piper voices](https://huggingface.co/rhasspy/piper-voices) | Synthetic voices: Joe (American), Cori (British), VCTK speaker p326 (Australian) | CC0, public domain and CC BY 4.0 |
| [eSpeak NG](https://github.com/espeak-ng/espeak-ng), through [phonemizer](https://www.npmjs.com/package/phonemizer) | Turns typed text into the sounds the synthetic voices read | GPL 3.0 |

The synthetic voices are fetched with `python -m scripts.fetch_voices` into `web/public/voices`. They run in the browser with the same ONNX runtime as the phoneme model, and a voice is downloaded the first time it is needed. The American voice was chosen because it was trained on American phoneme strings: several American Piper voices were trained on British ones and say "tomato" the British way. The deployed site includes eSpeak NG, which is GPL 3.0; the code in this repository stays MIT.

Each recording's speaker, licence and source page is listed in `web/public/audio/credits.json` and on the app's Credits page. The recordings were converted to MP3, trimmed and loudness-normalized; those under a ShareAlike licence remain under it.
