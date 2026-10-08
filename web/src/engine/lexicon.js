// Accent dictionaries and the practice word bank, loaded once from static JSON.
// The same lexicon.json is read by the Python package (pronunciation/lexicon.py).

export const DEFAULT_ACCENT = 'ga'
export const MAX_TEXT_WORDS = 8
export const MAX_TEXT_CHARS = 60

let loading = null

export function loadLexicon() {
  if (!loading) {
    loading = Promise.all([
      fetch('/lexicon/lexicon.json').then((r) => r.json()),
      fetch('/lexicon/practice_words.json').then((r) => r.json()),
      fetch('/audio/credits.json').then((r) => (r.ok ? r.json() : {})).catch(() => ({})),
    ]).then(([lexicon, levels, credits]) => ({ ...lexicon, levels, credits }))
    loading.catch(() => { loading = null })
  }
  return loading
}

export class UnknownWordError extends Error {
  constructor(word) {
    super(`"${word}" is not in the dictionary yet. Try another word.`)
    this.word = word
  }
}

export const normalizeText = (text) =>
  text.toLowerCase().replace(/[^a-z' ]+/g, ' ').split(' ').filter(Boolean)

// Australian entries are stored only where they differ from the British entry they derive from.
export function wordPhonemes(lexicon, word, accent) {
  const entry = lexicon.words[word]
  if (!entry) throw new UnknownWordError(word)
  return (entry[accent] ?? entry[accent === 'au' ? 'rp' : 'ga']).split(' ')
}

export function textEntry(lexicon, text, accent) {
  const words = normalizeText(text)
  if (words.length === 0) throw new Error('Type a word or short phrase using letters.')
  if (words.length > MAX_TEXT_WORDS || words.join(' ').length > MAX_TEXT_CHARS) {
    throw new Error(`Keep it to ${MAX_TEXT_WORDS} words and ${MAX_TEXT_CHARS} characters.`)
  }
  return {
    word: words.join(' '),
    phonemes: words.flatMap((word) => wordPhonemes(lexicon, word, accent)),
    difficulty: 'custom',
  }
}

// Practice words for one accent, each with its target phonemes and whether a human recording exists.
export function practiceWords(lexicon, accent) {
  return Object.entries(lexicon.levels).flatMap(([difficulty, words]) =>
    words.map((word) => ({
      word,
      difficulty,
      phonemes: wordPhonemes(lexicon, word, accent),
      // True when this accent says the word differently from General American.
      distinct: accent !== 'ga' && wordPhonemes(lexicon, word, accent).join(' ') !== lexicon.words[word].ga,
      recording: lexicon.credits[accent]?.[word] ?? null,
    })))
}

export const recordingUrl = (accent, word) => `/audio/${accent}/${encodeURIComponent(word)}.mp3`
