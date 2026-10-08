import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { align, combineScores, roundHalfEven, scoreAttempt } from './scoring'

const cases = JSON.parse(
  readFileSync(new URL('../../../tests/fixtures/scoring_cases.json', import.meta.url), 'utf8'),
)

describe('scoreAttempt', () => {
  it('has fixtures to check against', () => {
    expect(cases.length).toBeGreaterThan(50)
  })

  // The same cases are asserted against the Python scorer in tests/test_scoring.py.
  it.each(cases.map((c, index) => [index, c]))('matches the Python scorer on case %i', (_, c) => {
    const result = scoreAttempt(c.target, c.predicted, c.confidence, c.margin, {
      nativeScore: c.native_score,
      spans: c.spans,
    })
    expect(result).toEqual(c.expected)
  })
})

describe('helpers', () => {
  it('rounds ties to even like Python', () => {
    expect(roundHalfEven(62.5)).toBe(62)
    expect(roundHalfEven(63.5)).toBe(64)
    expect(roundHalfEven(0.25, 1)).toBe(0.2)
    expect(roundHalfEven(6.84, 1)).toBe(6.8)
  })

  it('aligns a deletion', () => {
    const alignment = align(['S', 'T', 'UW', 'D'], ['S', 'UW', 'D'])
    expect(alignment.ops.map((op) => op.kind)).toEqual(['equal', 'delete', 'equal', 'equal'])
  })

  it('caps the learned score above the alignment score', () => {
    expect(combineScores(2, 9)).toBe(5)
    expect(combineScores(8, 9)).toBeCloseTo(8.6)
  })
})
