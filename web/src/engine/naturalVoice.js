// Sentences a visitor can read in their normal voice. These are the only takes that
// count as how a place sounds, so there are several: each one is optional, and each
// gives more of the speaker's own accent. They are ordinary sentences built from
// words that differ between accents (weather, bath, car, water).
import { textEntry } from './lexicon'

export const SENTENCES = [
  'the weather is very cold today',
  'my mother walks in the garden',
  'i would like a glass of water',
  'we had a bath after the dance',
  'i park the car near the water',
]

// The fewest and most sounds the sentence has across the accent dictionaries. A reading is
// only checked for being about this long: it is never compared with a dictionary, because
// a strong accent is exactly what these takes are for.
export function targetSounds(lexicon, sentence) {
  const counts = Object.keys(lexicon.accents).map((accent) => textEntry(lexicon, sentence, accent).phonemes.length)
  return [Math.min(...counts), Math.max(...counts)]
}
