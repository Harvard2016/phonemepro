// Whether the model heard speech at all.
//
// The speech finder in cleanup.js works on loudness, and a real room is not quiet: breath,
// a chair, or the browser's gain control lifting the background can look like several
// seconds of "speech". The model is the better judge. If it heard almost nothing in what
// cleanup called speech, the take is treated as silence: no score, a retake notice, and
// nothing stored. How close real speech gets to these limits is measured by
// scripts/evaluate_take_checks.py (Docs/model/take_checks.json).
export const MIN_SOUNDS = 2
export const MIN_SOUNDS_PER_SECOND = 1

// `phonemes` are the sounds the model heard; `speechSeconds` is what cleanup measured.
export function heardNoSpeech(phonemes, speechSeconds) {
  if (phonemes.length < MIN_SOUNDS) return true
  return speechSeconds > 0 && phonemes.length / speechSeconds < MIN_SOUNDS_PER_SECOND
}
