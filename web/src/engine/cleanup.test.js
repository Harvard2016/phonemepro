import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { cleanTake, frameLevels, highPass, SAMPLE_RATE, spectralGate, speechFrames } from './cleanup'

const cases = JSON.parse(
  readFileSync(new URL('../../../tests/fixtures/cleanup_cases.json', import.meta.url), 'utf8'),
)

// Rebuild a recording from its recipe. Mirrors `synthesize` in scripts/make_cleanup_fixtures.py,
// including the fixed linear congruential noise, so both languages clean identical audio.
function synthesize(sections) {
  let state = 12345n
  const samples = []
  let position = 0
  for (const [kind, seconds, level, frequency] of sections) {
    for (let i = 0; i < Math.round(seconds * SAMPLE_RATE); i += 1) {
      state = (1103515245n * state + 12345n) % 2147483648n
      const noise = Number(state) / 1073741824 - 1
      const tone = Math.sin((2 * Math.PI * frequency * position) / SAMPLE_RATE)
      const rumble = Math.sin((2 * Math.PI * 30 * position) / SAMPLE_RATE)
      let value
      if (kind === 'silence') value = 0
      else if (kind === 'noise') value = level * noise
      else if (kind === 'tone') value = level * tone
      else if (kind === 'tone+noise') value = level * tone + 0.1 * level * noise
      else if (kind === 'rumble') value = 0.1 + level * rumble
      else value = 0.1 + 0.2 * rumble + level * tone // tone+rumble
      samples.push(Math.max(-1, Math.min(1, value)))
      position += 1
    }
  }
  return Float32Array.from(samples)
}

describe('cleanTake', () => {
  it('has fixtures to check against', () => {
    expect(cases.length).toBeGreaterThan(15)
  })

  // The same cases are produced by pronunciation/audio_cleanup.py (scripts/make_cleanup_fixtures.py).
  it.each(cases.map((c) => [c.name, c]))('matches the Python cleanup on %s', (_, c) => {
    const result = cleanTake(synthesize(c.sections))
    // The cleaned audio itself: both languages run the same noise reduction, to float rounding.
    const { cleaned_rms: cleanedRms, ...expected } = c.expected
    const power = result.samples.reduce((sum, value) => sum + value * value, 0) / Math.max(result.samples.length, 1)
    expect(Math.sqrt(power)).toBeCloseTo(cleanedRms, 4)
    expect({
      found_speech: result.foundSpeech,
      start: result.start,
      end: result.end,
      speech_seconds: result.speechSeconds,
      snr_db: result.snrDb,
      clipped: Math.round(result.clipped * 1e6) / 1e6,
      kept_samples: result.samples.length,
    }).toEqual(expected)
  })

  it('returns the kept region of the input', () => {
    const audio = synthesize(cases.find((c) => c.name === 'word_with_silence_around').sections)
    const result = cleanTake(audio)
    expect(result.samples.length).toBe(result.end - result.start)
    expect(result.samples.length).toBeLessThanOrEqual(audio.length)
    expect(result.samples.every(Number.isFinite)).toBe(true)
  })

  it('copes with nothing at all', () => {
    expect(cleanTake(new Float32Array(0))).toMatchObject({ foundSpeech: false, speechSeconds: 0, snrDb: 0, clipped: 0 })
  })
})

describe('helpers', () => {
  it('removes a constant offset', () => {
    const filtered = highPass(new Float32Array(SAMPLE_RATE).fill(0.5))
    expect(Math.abs(filtered[filtered.length - 1])).toBeLessThan(1e-3)
  })

  it('measures frame energy in dB', () => {
    const levels = frameLevels(new Float32Array(1600).fill(0.1))
    expect(levels.length).toBe(9)
    expect(levels[0]).toBeCloseTo(-20, 5)
    expect(frameLevels(new Float32Array(1600))[0]).toBe(-80)
  })

  it('reduces steady noise, keeps the length, and leaves short audio alone', () => {
    const noise = synthesize([['noise', 1, 0.05, 0]])
    const cleaned = spectralGate(noise, noise)
    const rms = (audio) => Math.sqrt(audio.reduce((sum, value) => sum + value * value, 0) / audio.length)
    expect(cleaned.length).toBe(noise.length)
    expect(rms(cleaned)).toBeLessThan(0.5 * rms(noise))
    const short = noise.slice(0, 100)
    expect(spectralGate(short, short)).toBe(short)
    // With no noise to learn from there is nothing to subtract.
    const tone = synthesize([['tone', 0.5, 0.3, 220]])
    const untouched = spectralGate(tone, new Float32Array(4000))
    expect(Math.max(...tone.map((value, i) => Math.abs(value - untouched[i])))).toBeLessThan(1e-5)
  })

  it('finds no speech in too few frames', () => {
    expect(speechFrames(new Float64Array(3).fill(-10))).toBeNull()
  })
})
