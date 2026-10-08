// Per-phoneme accuracy from practice history, and words that train the weakest sounds.
// Mirrors build_insights in api.py.
import { stripStress } from './scoring'

const MIN_ATTEMPTS = 2
const round2 = (value) => Math.round(value * 100) / 100

export function buildInsights(attempts, words) {
  const stats = new Map()
  const bump = (phoneme, field) => {
    const key = stripStress(phoneme)
    const entry = stats.get(key) ?? { attempts: 0, errors: 0 }
    entry[field] += 1
    stats.set(key, entry)
  }

  for (const attempt of attempts) {
    attempt.target_phonemes.forEach((phoneme) => bump(phoneme, 'attempts'))
    for (const error of attempt.errors) (error.expected ?? []).forEach((phoneme) => bump(phoneme, 'errors'))
  }

  const phonemes = [...stats.entries()]
    .filter(([, s]) => s.attempts >= MIN_ATTEMPTS)
    .map(([phoneme, s]) => ({
      phoneme, attempts: s.attempts, errors: s.errors, error_rate: round2(Math.min(1, s.errors / s.attempts)),
    }))
    .sort((a, b) => b.error_rate - a.error_rate || b.attempts - a.attempts)

  const weak = phonemes.filter((item) => item.error_rate > 0).slice(0, 5)
  const weights = new Map(weak.map((item) => [item.phoneme, item.error_rate]))

  const recommended = words
    .map((entry) => {
      const focus = [...new Set(entry.phonemes.map(stripStress))].filter((p) => weights.has(p))
      return { entry: { ...entry, focus }, weight: focus.reduce((sum, p) => sum + weights.get(p), 0) }
    })
    .filter((item) => item.entry.focus.length > 0)
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 6)
    .map((item) => item.entry)

  return { attempts: attempts.length, phonemes, weak_phonemes: weak, recommended_words: recommended }
}
