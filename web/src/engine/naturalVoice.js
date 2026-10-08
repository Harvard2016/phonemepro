// Sentences a visitor can read in their normal voice. These are the only takes that
// count as how a place sounds, so there are several: each one is optional, and each
// gives more of the speaker's own accent. They are ordinary sentences built from
// words that differ between accents (weather, bath, car, water).
import { textEntry } from './lexicon'
import { scoreAttempt } from './scoring'

export const SENTENCES = [
  'the weather is very cold today',
  'my mother walks in the garden',
  'i would like a glass of water',
  'we had a bath after the dance',
  'i park the car near the water',
]

// How much of the sentence was attempted and matched, against whichever dictionary fits the
// speaker best. Nobody is being scored here: the dictionaries only tell whether this sentence
// was read at all. Returns { soundsAttempted, soundsMatched }, each 0..100.
export function readingOf(lexicon, sentence, heard) {
  const metrics = Object.keys(lexicon.accents).map((accent) =>
    scoreAttempt(textEntry(lexicon, sentence, accent).phonemes, heard.phonemes, heard.confidence, heard.margin).metrics)
  return {
    soundsAttempted: Math.max(...metrics.map((m) => m.completeness)),
    soundsMatched: Math.max(...metrics.map((m) => m.accuracy)),
  }
}
