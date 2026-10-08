import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { createMemoryStore } from '../engine/contributions'
import { textEntry } from '../engine/lexicon'
import { retakeOutcome, settleTake } from './run'
import { checkExport, describeSetup, evaluateStep, formatResults, refusalOf, STEPS } from './steps'

// No microphone and no model: each step is fed what the model would have heard.
const lexicon = JSON.parse(readFileSync(new URL('../../public/lexicon/lexicon.json', import.meta.url), 'utf8'))
const fixture = JSON.parse(readFileSync(new URL('../../../tests/fixtures/contribution_export.json', import.meta.url), 'utf8'))

const MODEL = 'check-model'
const CONTRIBUTOR = '3f2b8c1e-5a47-4d09-9b61-7e0c2a4d8f13'
const quality = { foundSpeech: true, speechSeconds: 0.6, snrDb: 38.2, clipped: 0 }
const step = (id) => STEPS.find((candidate) => candidate.id === id)
const saying = (text, accent = 'ga') => ({
  phonemes: textEntry(lexicon, text, accent).phonemes, confidence: 0.9, margin: 0.8, nativeScore: 0.8,
  features: Array.from({ length: 256 }, (_, i) => Math.cos(i)), featureCount: 256, modelVersion: MODEL,
})
const run = (id, heard, store = createMemoryStore(), changes = {}) =>
  settleTake({ step: step(id), lexicon, heard, quality, store, contributor: CONTRIBUTOR, ...changes })
const spoken = (id) => saying(step(id).word, step(id).accent ?? 'rp')

describe('the step table', () => {
  it('has eleven steps in order, each with an instruction', () => {
    expect(STEPS.map((s) => s.id)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    expect(STEPS.every((s) => s.title && s.say && s.how)).toBe(true)
    expect(STEPS.filter((s) => s.kind === 'take').every((s) => Object.keys(s.expect).length > 0)).toBe(true)
  })

  it('uses only words the dictionary knows', () => {
    for (const s of STEPS.filter((candidate) => candidate.word)) {
      expect(() => textEntry(lexicon, s.word, s.accent ?? 'ga')).not.toThrow()
    }
    expect(() => textEntry(lexicon, 'banana', 'ga')).not.toThrow()
  })

  it('describes each setup in words', () => {
    expect(describeSetup(step(3), { ga: 'American' })).toBe('Profile US · South · sharing on · target American')
    expect(describeSetup(step(4))).toBe('Profile GB · Scotland · sharing on · normal voice')
    expect(describeSetup(step(10), { ga: 'American' })).toBe('Profile GB · sharing off · target American')
    expect(describeSetup(step(11))).toMatch(/earlier steps/)
  })
})

describe('steps 1 to 4: labels and routing', () => {
  it.each([1, 2, 3, 4])('passes step %i when the word is said as asked', async (id) => {
    const outcome = await run(id, spoken(id))
    const evaluation = evaluateStep(step(id), outcome)
    expect(evaluation.rows.filter((row) => !row.pass)).toEqual([])
    expect(evaluation.status).toBe('pass')
    expect(outcome.joins).toEqual(step(id).expect.joins)
  })

  it('expects every practice take to be an attempt and only the normal voice to be native', () => {
    expect(step(1).expect).toMatchObject({ kind: 'attempt', joins: ['GB>ga'], matches_home_accent: false })
    expect(step(2).expect).toMatchObject({ kind: 'attempt', joins: ['GB>rp'], matches_home_accent: true })
    expect(step(3).expect).toMatchObject({ kind: 'attempt', joins: ['US>ga'], region: 'S' })
    expect(step(4).expect).toMatchObject({ kind: 'native', joins: ['GB', 'GB-SCT'], natural_voice: true })
  })

  it('fails a step, row by row, when a label comes out differently', async () => {
    const outcome = await run(3, spoken(3))
    // What the earlier routing did: count a home-standard practice take as native to country and region.
    const evaluation = evaluateStep(step(3), { ...outcome, kind: 'native', joins: ['US', 'US-S'] })
    expect(evaluation.status).toBe('fail')
    expect(evaluation.rows.filter((row) => !row.pass)).toEqual([
      { label: 'counts as', expected: 'attempt', actual: 'native', pass: false },
      { label: 'joins totals', expected: 'US>ga', actual: 'US, US-S', pass: false },
    ])
  })

  it('fails a kept-take step when the take was not kept', async () => {
    const outcome = await run(1, { ...spoken(1), confidence: 0.2 })
    expect(outcome).toMatchObject({ kept: false, failed_checks: ['confidence'], joins: null })
    expect(evaluateStep(step(1), outcome).status).toBe('fail')
  })
})

describe('steps 5 to 10', () => {
  it('step 5 passes only when silence is refused with a retake notice', async () => {
    const refused = retakeOutcome({ reason: 'silent', quality: { foundSpeech: false, snrDb: 0 } })
    expect(refused).toMatchObject({ kept: false, reason: 'no speech', retake_notice: true, scored: false })
    expect(evaluateStep(step(5), refused).status).toBe('pass')
    expect(evaluateStep(step(5), retakeOutcome({ reason: 'noisy', quality })).status).toBe('fail')
  })

  it('step 5 also passes when the room is transcribed as a run of sounds and refused as noise', async () => {
    // From a real microphone: 13 sounds heard in 3.45 s of "silence", at 17.9 dB.
    const room = { foundSpeech: true, speechSeconds: 3.45, snrDb: 17.9, clipped: 0 }
    const sounds = ['F', 'T', 'AH0', 'S', 'IH0', 'K', 'T', 'AH0', 'N', 'S', 'F', 'T', 'AH0']
    const store = createMemoryStore()
    const outcome = await run(5, { ...saying('water'), phonemes: sounds, confidence: 0.6 }, store, { quality: room })
    expect(outcome).toMatchObject({ kept: false, reason: 'extra sounds, likely background noise', retake_notice: true })
    expect(store.all()).toEqual([])
    const evaluation = evaluateStep(step(5), outcome)
    expect(evaluation.status).toBe('pass')
    expect(evaluation.rows[1]).toEqual({
      label: 'reason not kept', expected: 'no speech or extra sounds, likely background noise',
      actual: 'extra sounds, likely background noise', pass: true,
    })
    // Kept, or refused for some other reason, or with no notice: still a failure.
    expect(evaluateStep(step(5), { ...outcome, reason: 'too few sounds heard' }).status).toBe('fail')
    expect(evaluateStep(step(5), { ...outcome, retake_notice: false }).status).toBe('fail')
    expect(evaluateStep(step(5), { ...outcome, kept: true }).status).toBe('fail')
    expect(evaluateStep(step(5), await run(5, spoken(5))).status).toBe('fail')
  })

  it('step 5 passes when a quiet room is taken for speech but the model hears next to nothing', async () => {
    // From a real microphone: cleanup found 4.07 s of "speech" at 31 dB and the model heard F T.
    const room = { foundSpeech: true, speechSeconds: 4.07, snrDb: 31.3, clipped: 0 }
    const store = createMemoryStore()
    const outcome = await run(5, { ...saying('water'), phonemes: ['F', 'T'], confidence: 0.67 }, store, { quality: room })
    expect(outcome).toMatchObject({ scored: false, score: null, kept: false, reason: 'no speech', retake_notice: true })
    expect(outcome.detail).toMatch(/model heard F T in 4.07 s/)
    expect(store.all()).toEqual([])
    expect(evaluateStep(step(5), outcome).status).toBe('pass')
  })

  it('step 6 passes when the wrong word is not kept, and fails when it is', async () => {
    const wrong = await run(6, saying('banana'))
    expect(wrong).toMatchObject({ kept: false, scored: true })
    expect(wrong.failed_checks).toContain('matched')
    expect(wrong.reason).toMatch(/recognisably the target word/)
    expect(evaluateStep(step(6), wrong).status).toBe('pass')
    expect(evaluateStep(step(6), await run(6, saying('water'))).status).toBe('fail')
  })

  it('step 7 keeps a strong accent that step 6 would not confuse with another word', async () => {
    const heavy = (phonemes) => ({ ...saying('water'), phonemes })
    // Non-rhotic, flapped, a different vowel: half the sounds are still heard as written.
    for (const phonemes of [['W', 'AA1', 'T', 'AH0'], ['V', 'AO1', 'T', 'AH0'], ['W', 'AH1', 'T'], ['W', 'AO1', 'D', 'AH0']]) {
      const outcome = await run(7, heavy(phonemes))
      expect(outcome).toMatchObject({ kept: true, kind: 'attempt', joins: ['GB>ga'] })
      expect(evaluateStep(step(7), outcome).status).toBe('pass')
    }
    // The measured limit: with only one sound in four as written, the take is dropped.
    const tooFar = await run(7, heavy(['W', 'AA1', 'D', 'AH0']))
    expect(tooFar).toMatchObject({ kept: false, failed_checks: ['matched'] })
    expect(evaluateStep(step(7), tooFar).status).toBe('fail')
    // And the same rule still refuses a different word.
    expect((await run(7, saying('banana'))).kept).toBe(false)
  })

  it('a normal-voice reading in a strong accent is kept without any dictionary match', async () => {
    const unlike = { ...saying(step(4).word, 'rp'), phonemes: Array.from({ length: 18 }, (_, i) => (i % 2 ? 'AH0' : 'K')) }
    const outcome = await run(4, unlike)
    expect(outcome).toMatchObject({ kept: true, natural_voice: true, joins: ['GB', 'GB-SCT'] })
    const tooShort = await run(4, { ...unlike, phonemes: ['DH', 'AH0', 'W', 'EH1', 'DH', 'ER0'] })
    expect(tooShort).toMatchObject({ kept: false, failed_checks: ['length'], reason: 'too few sounds heard' })
    expect(outcome.detail).toMatch(/18 sounds heard, 10 to 40 allowed for a target of 20/)
  })

  it('step 8 passes only when the first sound heard is F', async () => {
    expect(evaluateStep(step(8), await run(8, saying('fish'))).status).toBe('pass')
    const clipped = { ...saying('fish'), phonemes: ['IH1', 'SH'] }
    const evaluation = evaluateStep(step(8), await run(8, clipped))
    expect(evaluation.status).toBe('fail')
    expect(evaluation.rows).toEqual([{ label: 'first sound heard', expected: 'F', actual: 'IH', pass: false }])
  })

  it('step 9 reports and never judges', async () => {
    const kept = evaluateStep(step(9), await run(9, spoken(9)))
    expect(kept.status).toBe('info')
    expect(kept.rows.map((row) => [row.label, row.actual])).toEqual([
      ['signal-to-noise (dB)', '38.2'], ['kept', 'yes'], ['reason not kept', 'none'], ['retake notice shown', 'no'],
    ])
    // Noise the browser hid from the SNR check, transcribed as extra sounds in front of the word.
    const extra = await run(9, { ...saying('water'), phonemes: ['S', 'IY0', 'K', 'W', 'AH0', 'T', 'AH0'] })
    expect(extra).toMatchObject({ scored: true, kept: false, reason: 'extra sounds, likely background noise', retake_notice: true })
    expect(evaluateStep(step(9), extra).rows.map((row) => row.actual))
      .toEqual(['38.2', 'no', 'extra sounds, likely background noise', 'yes'])
    const refused = evaluateStep(step(9), retakeOutcome({ reason: 'noisy', quality: { ...quality, snrDb: 6.5 } }))
    expect(refused.status).toBe('info')
    expect(refused.rows.map((row) => row.actual)).toEqual(['6.5', 'no', 'too noisy to score', 'yes'])
  })

  it('step 10 passes when the take is scored and nothing is stored', async () => {
    const store = createMemoryStore()
    const outcome = await run(10, spoken(10), store)
    expect(outcome).toMatchObject({ scored: true, kept: false, stored_added: 0, reason: '"Help it learn" is off' })
    expect(outcome.score).toBeGreaterThan(8)
    expect(store.all()).toEqual([])
    expect(evaluateStep(step(10), outcome).status).toBe('pass')
    expect(evaluateStep(step(10), { ...outcome, kept: true, stored_added: 1, reason: null }).status).toBe('fail')
  })
})

describe('the sandbox store', () => {
  it('keeps takes in memory only and can forget one', async () => {
    // There is no IndexedDB, localStorage or sessionStorage in this test environment: reaching for one would throw.
    expect(globalThis.indexedDB).toBeUndefined()
    const store = createMemoryStore()
    const first = await run(1, spoken(1), store)
    await run(2, spoken(2), store)
    expect(store.all().map((take) => take.target_accent)).toEqual(['ga', 'rp'])
    store.remove(first.take_id)
    expect(store.all().map((take) => take.target_accent)).toEqual(['rp'])
    expect(createMemoryStore().all()).toEqual([])
  })
})

describe('the last step: the export', () => {
  const session = async () => {
    const store = createMemoryStore()
    const kept = {}
    for (const id of [1, 2, 3, 4]) kept[id] = (await run(id, spoken(id), store)).take_id
    return { store, kept }
  }

  it('passes when everything kept is valid and routed as predicted', async () => {
    const { store, kept } = await session()
    const result = checkExport(store.all(), kept, MODEL)
    expect(result.rows.filter((row) => !row.pass)).toEqual([])
    expect(result.status).toBe('pass')
    expect(result.rows.slice(4).map((row) => [row.label, row.actual])).toEqual([
      ['step 1 joins', 'GB>ga'], ['step 2 joins', 'GB>rp'], ['step 3 joins', 'US>ga'], ['step 4 joins', 'GB, GB-SCT'],
    ])
  })

  it('marks steps that kept nothing as not judged, and is skipped when none did', async () => {
    const { store, kept } = await session()
    store.remove(kept[2])
    const partial = checkExport(store.all(), { ...kept, 2: null }, MODEL)
    expect(partial.status).toBe('pass')
    expect(partial.rows.find((row) => row.label === 'step 2 joins')).toMatchObject({ actual: 'not kept in this session', pass: null })
    expect(checkExport([], {}, MODEL).status).toBe('skipped')
  })

  it('fails when a take would be refused by ingest or routed elsewhere', async () => {
    const { store, kept } = await session()
    const takes = store.all()
    expect(checkExport(takes.map((take) => ({ ...take, region: take.region && 'Leeds' })), kept, MODEL).status).toBe('fail')
    expect(checkExport(takes, kept, 'another-model').rows[2]).toMatchObject({ pass: false })
    expect(checkExport(takes.map((take) => ({ ...take, wav: 'UklGR' })), kept, MODEL).rows[3]).toMatchObject({ actual: 'wav, wav, wav, wav', pass: false })
    // A practice take relabelled as normal voice would be routed as native.
    const relabelled = takes.map((take) => (take.id === kept[1] ? { ...take, natural_voice: true, target_accent: null } : take))
    const result = checkExport(relabelled, kept, MODEL)
    expect(result.status).toBe('fail')
    expect(result.rows.find((row) => row.label === 'step 1 joins')).toMatchObject({ expected: 'GB>ga', actual: 'GB', pass: false })
  })

  it('refuses the same takes the ingest script refuses', () => {
    const [good] = fixture.export.takes
    const model = good.model_version
    expect(fixture.export.takes.map((take) => refusalOf(take, model))).toEqual(fixture.export.takes.map(() => null))
    const refused = (changes) => refusalOf({ ...fixture.export.takes[3], ...changes }, model)
    expect(refused({ id: 'not-a-uuid' })).toBe('bad take id')
    expect(refused({ contributor: null })).toBe('bad contributor id')
    expect(refused({ country: 'India' })).toBe('bad country')
    expect(refused({ region: 'Leeds' })).toBe('bad region')
    expect(refused({ region: 'SCT' })).toBe('bad region')
    expect(refused({ natural_voice: 'yes' })).toBe('bad labels')
    expect(refused({ target_accent: 'scouse' })).toBe('bad target accent')
    expect(refused({ natural_voice: true })).toBe('bad target accent')
    expect(refused({ matches_home_accent: true })).toBe('labels disagree')
    expect(refused({ model_version: 'older' })).toBe('made with a different model version')
    expect(refused({ features: [1, 2, 3] })).toBe('wrong feature length')
    expect(refused({ features: [null, ...good.features.slice(1)] })).toBe('non-finite feature')
    expect(refused({ features: [61, ...good.features.slice(1)] })).toBe('feature out of range')
    expect(refusalOf('a string', model)).toBe('not a take')
  })
})

describe('formatResults', () => {
  it('gives plain text with every step, its rows, and a tally', async () => {
    const store = createMemoryStore()
    const results = {}
    for (const id of [1, 3]) {
      const outcome = await run(id, spoken(id), store)
      results[id] = { outcome, evaluation: evaluateStep(step(id), outcome) }
    }
    const clipped = await run(8, { ...saying('fish'), phonemes: ['IH1', 'SH'] }, store)
    results[8] = { outcome: clipped, evaluation: evaluateStep(step(8), clipped) }
    const noisy = retakeOutcome({ reason: 'noisy', quality: { ...quality, snrDb: 6.5 } })
    results[9] = { outcome: noisy, evaluation: evaluateStep(step(9), noisy) }

    const text = formatResults(results, { modelVersion: MODEL, when: new Date('2026-10-08T12:00:00Z') })
    const lines = text.split('\n')
    expect(lines[0]).toBe('PhonemePro label check · 2026-10-08T12:00:00.000Z · model check-model')
    expect(lines).toContain(' 1. PASS    From Britain, practising American')
    expect(lines).toContain('      ✓ joins totals: expected GB>ga, got GB>ga')
    expect(lines).toContain(' 2. SKIPPED From Britain, practising British')
    expect(lines).toContain(' 8. FAIL    A soft first sound')
    expect(lines).toContain('      ✕ first sound heard: expected F, got IH')
    expect(lines).toContain(' 9. INFO    A noisy room')
    expect(lines).toContain('        signal-to-noise (dB): expected not judged, got 6.5')
    expect(lines.some((line) => line.includes('heard B EH1 T ER0 · score'))).toBe(true)
    expect(lines.at(-1)).toBe('Passed 2, failed 1, informational 1, skipped 7.')
  })
})
