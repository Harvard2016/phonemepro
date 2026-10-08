import { describe, expect, it } from 'vitest'
import { withAccentCoaching } from './coaching'
import { decodeCtc, normalizeAudio, softmax } from './decode'
import { countToday, exportHistory, toCsv } from './history'
import { buildInsights } from './insights'
import { normalizeText, practiceWords, textEntry, UnknownWordError, wordPhonemes } from './lexicon'
import { scoreAttempt } from './scoring'
import { toIpa, wordToIpa } from '../lib/phonemes'

const lexicon = {
  accents: { ga: {}, rp: {}, au: {} },
  words: {
    water: { ga: 'W AO1 T ER0', rp: 'W AO1 T AH0' },
    dance: { ga: 'D AE1 N S', rp: 'D AA1 N S', au: 'D AE1 N S' },
    the: { ga: 'DH AH0', rp: 'DH AH0' },
    new: { ga: 'N UW1', rp: 'N Y UW1' },
  },
  levels: { easy: ['new'], medium: ['water', 'dance'] },
  credits: { rp: { water: { author: 'A Speaker', license: 'CC BY 3.0', page: 'https://example.org' } } },
}

describe('lexicon', () => {
  it('looks up each accent, with Australian falling back to British', () => {
    expect(wordPhonemes(lexicon, 'water', 'ga')).toEqual(['W', 'AO1', 'T', 'ER0'])
    expect(wordPhonemes(lexicon, 'water', 'au')).toEqual(['W', 'AO1', 'T', 'AH0'])
    expect(wordPhonemes(lexicon, 'dance', 'au')).toEqual(['D', 'AE1', 'N', 'S'])
    expect(() => wordPhonemes(lexicon, 'zzyzx', 'ga')).toThrow(UnknownWordError)
  })

  it('builds a phrase entry and enforces limits', () => {
    expect(normalizeText('The Water, 42!')).toEqual(['the', 'water'])
    expect(textEntry(lexicon, 'The water', 'rp')).toEqual({
      word: 'the water', phonemes: ['DH', 'AH0', 'W', 'AO1', 'T', 'AH0'], difficulty: 'custom',
    })
    expect(() => textEntry(lexicon, '!!', 'ga')).toThrow()
    expect(() => textEntry(lexicon, 'the '.repeat(9), 'ga')).toThrow(/8 words/)
  })

  it('marks practice words that differ from American and attaches recordings', () => {
    const british = practiceWords(lexicon, 'rp')
    const water = british.find((w) => w.word === 'water')
    expect(water).toMatchObject({ difficulty: 'medium', distinct: true })
    expect(water.recording.author).toBe('A Speaker')
    expect(practiceWords(lexicon, 'ga').every((w) => !w.distinct && !w.recording)).toBe(true)
  })
})

describe('decodeCtc', () => {
  const meta = { vocab: ['K', 'AE1', 'T', '[PAD]', '</s>'], blankId: 3, specialIds: [3, 4] }
  const frame = (winner, strength = 8) => meta.vocab.map((_, i) => (i === winner ? strength : 0))

  it('collapses repeats and blanks, and times each phoneme', () => {
    const frames = [3, 0, 0, 3, 1, 1, 1, 3, 2, 3, 3].map((id) => frame(id))
    const decoded = decodeCtc(Float32Array.from(frames.flat()), frames.length, meta)
    expect(decoded.phonemes).toEqual(['K', 'AE1', 'T'])
    expect(decoded.spans).toEqual([[0.02, 0.08], [0.08, 0.16], [0.16, 0.22]])
    expect(decoded.confidence).toBeGreaterThan(0.99)
    expect(decoded.margin).toBeGreaterThan(0.99)
  })

  it('keeps a phoneme repeated across a blank and skips special tokens', () => {
    const frames = [0, 3, 0, 4].map((id) => frame(id))
    expect(decodeCtc(Float32Array.from(frames.flat()), 4, meta).phonemes).toEqual(['K', 'K'])
  })

  it('reports low confidence for a flat distribution and nothing for silence', () => {
    const unsure = [frame(0, 0.1)]
    expect(decodeCtc(Float32Array.from(unsure.flat()), 1, meta).confidence).toBeLessThan(0.3)
    const silent = [frame(3), frame(3)]
    expect(decodeCtc(Float32Array.from(silent.flat()), 2, meta)).toMatchObject({ phonemes: [], confidence: 0 })
  })

  it('normalizes audio to zero mean and unit variance', () => {
    const out = normalizeAudio(Float32Array.from([1, 2, 3, 4]))
    const mean = out.reduce((a, b) => a + b, 0) / out.length
    const variance = out.reduce((a, b) => a + b * b, 0) / out.length
    expect(mean).toBeCloseTo(0, 5)
    expect(variance).toBeCloseTo(1, 4)
    expect(softmax([0, 0]).map((v) => Number(v.toFixed(2)))).toEqual([0.5, 0.5])
  })
})

describe('accent coaching', () => {
  const coach = (accent, target, heard) =>
    withAccentCoaching(scoreAttempt(target, heard, 0.9, 0.9), accent, target).coaching_tips[0]

  it('explains the missing r for British and the kept r for American', () => {
    expect(coach('rp', ['W', 'AO1', 'T', 'AH0'], ['W', 'AO1', 'T', 'ER0'])).toMatch(/British English has no r/)
    expect(coach('ga', ['W', 'AO1', 'T', 'ER0'], ['W', 'AO1', 'T', 'AH0'])).toMatch(/American English keeps the r/)
    expect(coach('ga', ['K', 'AA1', 'R'], ['K', 'AA1'])).toMatch(/pronounces this r/)
  })

  it('explains the BATH vowel and the yod', () => {
    expect(coach('rp', ['B', 'AA1', 'TH'], ['B', 'AE1', 'TH'])).toMatch(/long "ah"/)
    expect(coach('ga', ['B', 'AE1', 'TH'], ['B', 'AA1', 'TH'])).toMatch(/short "a"/)
    expect(coach('rp', ['N', 'Y', 'UW1'], ['N', 'UW1'])).toMatch(/"nyoo"/)
  })

  it('leaves ordinary mistakes to the generic tips', () => {
    const target = ['SH', 'IH1', 'P']
    const plain = scoreAttempt(target, ['S', 'IH1', 'P'], 0.9, 0.9)
    expect(withAccentCoaching(plain, 'rp', target)).toBe(plain)
  })
})

describe('insights and history helpers', () => {
  const attempts = [
    { word: 'water', accent: 'ga', score: 5, wer: 0.25, timestamp: '2026-10-07T10:00:00.000Z', target_phonemes: ['W', 'AO1', 'T', 'ER0'], predicted_phonemes: ['W', 'AO1', 'T', 'AH0'], errors: [{ expected: ['ER0'], heard: ['AH0'] }] },
    { word: 'water', accent: 'ga', score: 9, wer: 0, timestamp: '2026-10-07T11:00:00.000Z', target_phonemes: ['W', 'AO1', 'T', 'ER0'], predicted_phonemes: ['W', 'AO1', 'T', 'ER0'], errors: [] },
  ]

  it('ranks weak sounds and recommends words containing them', () => {
    const insights = buildInsights(attempts, practiceWords(lexicon, 'ga'))
    expect(insights.weak_phonemes).toEqual([{ phoneme: 'ER', attempts: 2, errors: 1, error_rate: 0.5 }])
    expect(insights.recommended_words.map((w) => w.word)).toEqual(['water'])
    expect(insights.recommended_words[0].focus).toEqual(['ER'])
  })

  it('counts today, exports and quotes CSV', () => {
    expect(countToday(attempts, new Date('2026-10-07T12:00:00.000Z'))).toBeGreaterThanOrEqual(1)
    expect(countToday(attempts, new Date('2027-01-01T12:00:00.000Z'))).toBe(0)
    expect(JSON.parse(exportHistory(attempts))).toMatchObject({ app: 'phonemepro', version: 1 })
    const csv = toCsv([{ ...attempts[0], word: 'say "hi"' }]).split('\n')
    expect(csv[0]).toBe('"word","accent","score","wer","target_phonemes","predicted_phonemes","errors","timestamp"')
    expect(csv[1].startsWith('"say ""hi""","ga","5"')).toBe(true)
  })
})

describe('IPA display', () => {
  it('writes each accent in its own convention', () => {
    expect(wordToIpa(['W', 'AO1', 'T', 'ER0'], 'ga')).toBe('ˈwɔtɚ')
    expect(wordToIpa(['W', 'AO1', 'T', 'AH0'], 'rp')).toBe('ˈwɔːtə')
    expect(wordToIpa(['K', 'AA1'], 'rp')).toBe('kɑː')
  })

  it('keeps the r-colour when a learner says an American ending in British mode', () => {
    expect(toIpa('ER0', 'rp')).toBe('ɚ')
    expect(toIpa('AH0', 'rp')).toBe('ə')
  })
})
