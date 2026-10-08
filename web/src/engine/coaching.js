// Accent-specific coaching. The generic tips explain how to make a sound; these
// explain the difference between accents when a learner uses the other accent's form.
import { stripStress } from './scoring'

const NON_RHOTIC = new Set(['rp', 'au'])
const NAMES = { ga: 'American', rp: 'British', au: 'Australian' }

function tipFor(error, accent, target) {
  const expected = stripStress(error.expected[0])
  const heard = error.heard[0] ? stripStress(error.heard[0]) : null
  const name = NAMES[accent] ?? 'this accent'
  const nonRhotic = NON_RHOTIC.has(accent)

  if (nonRhotic && expected === 'AH' && heard === 'ER') {
    return `${name} English has no r here. Let the ending relax into a plain "uh".`
  }
  if (!nonRhotic && expected === 'ER' && (heard === 'AH' || heard === null)) {
    return `${name} English keeps the r. Curl your tongue back as the vowel finishes.`
  }
  if (!nonRhotic && expected === 'R' && heard === null) {
    return `${name} English pronounces this r. Do not let it disappear after the vowel.`
  }
  if (nonRhotic && expected === 'AA' && heard === 'AE') {
    return `${name} English uses the long "ah" of "father" in this word, not the short "a" of "cat".`
  }
  if (expected === 'AE' && heard === 'AA') {
    return `${name} English uses the short "a" of "cat" in this word, not the long "ah" of "father".`
  }
  if (expected === 'Y' && heard === null && stripStress(target[error.index + 1] ?? '') === 'UW') {
    return `${name} English slips a "y" in before this vowel: "nyoo", not "noo".`
  }
  return null
}

// Returns the result with accent tips placed ahead of the generic ones (at most three in total).
export function withAccentCoaching(result, accent, targetPhonemes) {
  const accentTips = []
  for (const error of result.errors) {
    const tip = tipFor(error, accent, targetPhonemes)
    if (tip && !accentTips.includes(tip)) accentTips.push(tip)
  }
  if (accentTips.length === 0) return result
  return { ...result, coaching_tips: [...accentTips, ...result.coaching_tips].slice(0, 3) }
}
