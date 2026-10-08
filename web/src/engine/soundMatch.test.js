import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { sameClass, soundsSameClass } from './soundMatch'

const cases = JSON.parse(
  readFileSync(new URL('../../../tests/fixtures/sound_match_cases.json', import.meta.url), 'utf8'),
)

describe('soundsSameClass', () => {
  it('has fixtures to check against', () => {
    expect(cases.length).toBeGreaterThan(80)
  })

  // The same cases are asserted against pronunciation/sound_match.py in tests/test_sound_match.py.
  it.each(cases.map((c) => [c.note, c]))('matches the Python rule on: %s', (_, c) => {
    expect(soundsSameClass(c.target, c.heard)).toBe(c.expected)
  })
})

describe('sameClass', () => {
  it('accepts any vowel for a vowel and common accent swaps', () => {
    expect(sameClass('AO1', 'AH0')).toBe(true)
    expect(sameClass('TH', 'T')).toBe(true)
    expect(sameClass('T', 'TH')).toBe(true)
    expect(sameClass('DH', 'Z')).toBe(true)
    expect(sameClass('V', 'W')).toBe(true)
  })

  it('refuses a consonant for a vowel and unrelated consonants', () => {
    expect(sameClass('AE1', 'T')).toBe(false)
    expect(sameClass('K', 'M')).toBe(false)
    // T is grouped with TH and with D, but that does not make P and T alike.
    expect(sameClass('P', 'T')).toBe(false)
  })
})
