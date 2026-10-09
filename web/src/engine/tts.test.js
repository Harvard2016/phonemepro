import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { phonemeIds } from './tts'

const fixture = JSON.parse(readFileSync(new URL('../../../tests/fixtures/voice_ids.json', import.meta.url)))
const voices = JSON.parse(readFileSync(new URL('../../public/voices/voices.json', import.meta.url)))

describe('phonemeIds', () => {
  it('matches the ids Piper itself builds for the same sounds', () => {
    expect(phonemeIds(fixture.phonemes, voices.ga.phonemeIds)).toEqual(fixture.ids)
  })

  it('skips symbols the voice does not know', () => {
    const map = { '^': [1], $: [2], _: [0], a: [5] }
    expect(phonemeIds('a?a', map)).toEqual([1, 0, 5, 0, 5, 0, 2])
  })
})

describe('voices.json', () => {
  it('has a voice for each accent with the fields the engine reads', () => {
    for (const accent of ['ga', 'rp', 'au']) {
      const voice = voices[accent]
      expect(voice.chunks.length).toBeGreaterThan(0)
      expect(voice.scales).toHaveLength(3)
      expect(voice.sampleRate).toBe(22050)
      expect(voice.phonemeIds['^']).toBeDefined()
    }
    expect(voices.au.speakerId).toBeTypeOf('number')
    expect(voices.ga.espeak).toBe('en-us')
  })
})
