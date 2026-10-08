import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  allowedSounds, buildContribution, checkEligibility, checkViability, considerTake, createMemoryStore, exportContributions,
  EXTRA_SOUNDS, HOME_ACCENT, MAX_STORED, routeOf, TOO_FEW_SOUNDS, VIABILITY,
} from './contributions'
import { textEntry } from './lexicon'
import { SENTENCES, targetSounds } from './naturalVoice'
import { normalizeProfile } from './profile'
import { heardNoSpeech, MIN_SOUNDS, MIN_SOUNDS_PER_SECOND } from './takeChecks'
import { countries, isCountryCode } from '../lib/countries'
import { isRegion, regionName, REGIONS, regionsOf } from '../lib/regions'

// The same file is ingested by scripts/learn_regions.py in tests/test_learn_regions.py.
const fixture = JSON.parse(
  readFileSync(new URL('../../../tests/fixtures/contribution_export.json', import.meta.url), 'utf8'),
)

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
const features = Array.from({ length: 256 }, (_, i) => Math.sin(i) * 1.23456789)
const cleanQuality = { foundSpeech: true, speechSeconds: 0.62, snrDb: 31.4, clipped: 0 }
const good = { quality: cleanQuality, confidence: 0.9, soundsMatched: 100, soundsHeard: 4, targetSounds: [4, 4], features, featureCount: 256 }
const goodVoice = { quality: cleanQuality, confidence: 0.9, naturalVoice: true, soundsHeard: 20, targetSounds: [19, 20], features, featureCount: 256 }
const profile = { seen: true, country: 'GB', region: 'ENG_N', contribute: true, contributor: '3f2b8c1e-5a47-4d09-9b61-7e0c2a4d8f13' }

const failed = (input) => checkViability({ ...good, ...input }).checks.filter((c) => !c.passed).map((c) => c.id)

describe('checkViability', () => {
  it('passes a clean, confident, complete take and explains every check', () => {
    const { viable, checks } = checkViability(good)
    expect(viable).toBe(true)
    expect(checks.map((c) => c.id)).toEqual(['speech', 'snr', 'clipping', 'confidence', 'length', 'matched', 'features'])
    expect(checks.every((c) => c.label && c.detail)).toBe(true)
  })

  it('names the check that failed', () => {
    expect(failed({ quality: { ...cleanQuality, foundSpeech: false, speechSeconds: 0 } })).toEqual(['speech'])
    expect(failed({ quality: { ...cleanQuality, speechSeconds: 0.1 } })).toEqual(['speech'])
    expect(failed({ quality: { ...cleanQuality, snrDb: VIABILITY.minSnrDb - 0.1 } })).toEqual(['snr'])
    expect(failed({ quality: { ...cleanQuality, clipped: 0.01 } })).toEqual(['clipping'])
    expect(failed({ confidence: 0.3 })).toEqual(['confidence'])
    // A different word: nothing heard as written.
    expect(failed({ soundsMatched: 0 })).toEqual(['matched'])
    expect(failed({ soundsMatched: 25 })).toEqual(['matched'])
    expect(failed({ features: null })).toEqual(['features'])
    expect(failed({ features: features.slice(1) })).toEqual(['features'])
    expect(failed({ features: [NaN, ...features.slice(1)] })).toEqual(['features'])
    expect(failed({ features: [Infinity, ...features.slice(1)] })).toEqual(['features'])
    expect(failed({ quality: { ...cleanQuality, snrDb: 3, clipped: 0.2 }, confidence: 0.1 })).toEqual(['snr', 'clipping', 'confidence'])
  })

  it('accepts values exactly on each limit', () => {
    expect(checkViability({
      ...good,
      quality: { foundSpeech: true, speechSeconds: VIABILITY.minSpeechSeconds, snrDb: VIABILITY.minSnrDb, clipped: VIABILITY.maxClipped },
      confidence: VIABILITY.minConfidence,
      soundsMatched: VIABILITY.minSoundsMatched,
    }).viable).toBe(true)
  })
})

describe('checkViability for a strong accent', () => {
  it('keeps a practice take with only some sounds heard as written', () => {
    // "water" heard as W AA D AH would be 25%; W AH T as 50%. The bar was measured, not guessed.
    expect(VIABILITY.minSoundsMatched).toBe(30)
    expect(checkViability({ ...good, soundsMatched: 50 }).viable).toBe(true)
    expect(checkViability({ ...good, soundsMatched: 33 }).viable).toBe(true)
  })
})

describe('the length of a practice take', () => {
  const lengthOf = (soundsHeard, target = 4) =>
    checkViability({ ...good, soundsHeard, targetSounds: [target, target] }).checks.find((c) => c.id === 'length')

  it('allows between half and one and a half times the target', () => {
    expect(allowedSounds([4, 4])).toEqual([2, 6])
    expect(allowedSounds([3, 3])).toEqual([2, 4])
    expect(allowedSounds([5, 5])).toEqual([3, 7])
    expect(allowedSounds([20, 20], true)).toEqual([10, 40])
    for (const heard of [2, 4, 6]) expect(lengthOf(heard).passed).toBe(true)
  })

  it('names extra sounds as likely background noise, and too few as too few', () => {
    expect(lengthOf(7)).toMatchObject({ passed: false, reason: EXTRA_SOUNDS, detail: '7 sounds heard (needs 2 to 6)' })
    expect(lengthOf(1)).toMatchObject({ passed: false, reason: TOO_FEW_SOUNDS })
    expect(lengthOf(4).reason).toBeUndefined()
  })

  it('refuses a take with extra sounds, says why, and asks for a retake even when nothing is being kept', async () => {
    const heard = { phonemes: ['S', 'IY0', 'K', 'W', 'AO1', 'T', 'ER0'], confidence: 0.9, features, featureCount: 256, modelVersion: 'm' }
    const take = { targetAccent: 'ga', word: 'water', score: 5, heard, quality: cleanQuality, soundsMatched: 100, soundsSameClass: 100, targetSounds: [4, 4], featureCount: 256 }
    const store = createMemoryStore()
    const noisy = await considerTake({ ...take, profile, store })
    expect(noisy).toMatchObject({ stored: false, reason: EXTRA_SOUNDS, retake: true })
    expect(store.all()).toEqual([])
    const off = await considerTake({ ...take, profile: { ...profile, contribute: false }, store })
    expect(off).toMatchObject({ stored: false, reason: '"Help it learn" is off', retake: true })
    // The same word without the extra sounds is kept, with no retake.
    const clean = await considerTake({ ...take, heard: { ...heard, phonemes: heard.phonemes.slice(3) }, profile, store })
    expect(clean).toMatchObject({ stored: true, reason: null, retake: false })
    // Too few sounds is refused without blaming the room.
    const partial = await considerTake({ ...take, heard: { ...heard, phonemes: ['W'] }, profile, store })
    expect(partial).toMatchObject({ stored: false, reason: TOO_FEW_SOUNDS, retake: false })
  })
})

describe('heardNoSpeech', () => {
  it('treats fewer than two sounds as silence', () => {
    expect(MIN_SOUNDS).toBe(2)
    expect(heardNoSpeech([], 0.5)).toBe(true)
    expect(heardNoSpeech(['F'], 0.3)).toBe(true)
    expect(heardNoSpeech(['F', 'T'], 0.5)).toBe(false)
  })

  it('treats under one sound per second of "speech" as silence', () => {
    expect(MIN_SOUNDS_PER_SECOND).toBe(1)
    // A quiet room that cleanup took for four seconds of speech, as measured on a real microphone.
    expect(heardNoSpeech(['F', 'T'], 4.07)).toBe(true)
    expect(heardNoSpeech(['F', 'T', 'AH0'], 4.07)).toBe(true)
    // The slowest real take from the same session: "fish", three sounds in 1.26 s.
    expect(heardNoSpeech(['F', 'EH0', 'SH'], 1.26)).toBe(false)
    expect(heardNoSpeech(['W', 'T', 'AH0'], 0.51)).toBe(false)
    expect(heardNoSpeech(['F', 'T'], 2)).toBe(false)
  })
})

describe('checkViability in a normal voice', () => {
  const failedVoice = (input) => checkViability({ ...goodVoice, ...input }).checks.filter((c) => !c.passed).map((c) => c.id)

  it('never compares the reading with a dictionary', () => {
    const { viable, checks } = checkViability(goodVoice)
    expect(viable).toBe(true)
    expect(checks.map((c) => c.id)).toEqual(['speech', 'snr', 'clipping', 'confidence', 'length', 'features'])
    // Whatever was or was not "matched" is irrelevant.
    expect(checkViability({ ...goodVoice, soundsMatched: 0 }).viable).toBe(true)
  })

  it('only asks that the reading is about as long as the sentence', () => {
    expect(failedVoice({ soundsHeard: 10 })).toEqual([])
    expect(failedVoice({ soundsHeard: 40 })).toEqual([])
    expect(failedVoice({ soundsHeard: 9 })).toEqual(['length'])
    expect(failedVoice({ soundsHeard: 41 })).toEqual(['length'])
    expect(failedVoice({ soundsHeard: 0 })).toEqual(['length'])
    expect(checkViability({ ...goodVoice, soundsHeard: 9 }).checks.find((c) => c.id === 'length').detail)
      .toBe('9 sounds heard (needs 10 to 40)')
  })

  it('still needs clean, confident audio', () => {
    expect(failedVoice({ quality: { ...cleanQuality, snrDb: 5 }, confidence: 0.2 })).toEqual(['snr', 'confidence'])
  })
})

describe('checkEligibility', () => {
  it('keeps nothing unless the switch is on, a country is given, and there is room', () => {
    expect(checkEligibility({ ...profile, contribute: false }, 0)).toMatch(/off/)
    expect(checkEligibility({ ...profile, country: null }, 0)).toMatch(/country/)
    expect(checkEligibility(profile, MAX_STORED)).toMatch(/already holds/)
    expect(checkEligibility(profile, MAX_STORED - 1)).toBeNull()
  })
})

describe('buildContribution', () => {
  const build = (changes = {}) => buildContribution({
    profile, targetAccent: 'rp', naturalVoice: false, word: 'water', score: 8.4, features,
    quality: cleanQuality, confidence: 0.9123, soundsHeard: 4, soundsMatched: 75, soundsSameClass: 100, modelVersion: 'abc', ...changes,
  })

  it('has exactly the fields of an exported take, and no audio', () => {
    const take = build()
    expect(Object.keys(take)).toEqual(Object.keys(fixture.export.takes[0]))
    expect(Object.keys(take.quality)).toEqual(Object.keys(fixture.export.takes[0].quality))
    expect(take.quality).toMatchObject({ sounds_heard: 4, sounds_matched: 75, sounds_same_class: 100 })
    expect(build({ naturalVoice: true, soundsHeard: 20 }).quality).toMatchObject({ sounds_heard: 20, sounds_matched: null, sounds_same_class: null })
    expect(take.id).toMatch(UUID)
    expect(take.id).not.toBe(build().id)
    expect(take.features).toHaveLength(256)
    expect(take.features[1]).toBe(1.0389)
    expect(take.quality.confidence).toBe(0.912)
    expect(JSON.stringify(take)).not.toMatch(/samples|wav|audio|blob/i)
  })

  it('labels a take by whose accent it is', () => {
    expect(build()).toMatchObject({ country: 'GB', region: 'ENG_N', target_accent: 'rp', natural_voice: false, matches_home_accent: true, score: 8.4 })
    expect(build({ targetAccent: 'ga' })).toMatchObject({ target_accent: 'ga', matches_home_accent: false })
    expect(build({ profile: { ...profile, country: 'IN', region: null } })).toMatchObject({ country: 'IN', region: null, matches_home_accent: false })
    // In their normal voice nobody is attempting anything, whatever accent the practice page was set to.
    expect(build({ naturalVoice: true, targetAccent: 'ga', score: null }))
      .toMatchObject({ target_accent: null, natural_voice: true, matches_home_accent: false, score: null })
  })

  it('routes takes the way the ingest script does', () => {
    expect(HOME_ACCENT).toEqual(fixture.home_accent)
    for (const take of fixture.export.takes) {
      const [kind, keys] = fixture.routes[take.id]
      expect(routeOf(take)).toMatchObject({ kind, keys })
      const rebuilt = build({
        profile: { ...profile, country: take.country, region: take.region },
        targetAccent: take.target_accent, naturalVoice: take.natural_voice,
      })
      expect(rebuilt.matches_home_accent).toBe(take.matches_home_accent)
      expect(routeOf(rebuilt)).toMatchObject({ kind, keys })
    }
  })

  it('exports in the format the ingest script reads', () => {
    const parsed = JSON.parse(exportContributions(fixture.export.takes))
    expect(parsed).toMatchObject({ app: fixture.export.app, kind: fixture.export.kind, version: fixture.export.version })
    expect(parsed.takes).toEqual(fixture.export.takes)
  })
})

describe('profile', () => {
  it('starts with everything off and nothing known', () => {
    expect(normalizeProfile(null)).toEqual({ seen: false, country: null, region: null, contribute: false, contributor: null })
    expect(normalizeProfile('junk').contribute).toBe(false)
  })

  it('only turns contributing on for an explicit true', () => {
    expect(normalizeProfile({ contribute: 'yes' }).contribute).toBe(false)
    expect(normalizeProfile({ contribute: 1 }).contributor).toBeNull()
    const on = normalizeProfile({ contribute: true })
    expect(on.contributor).toMatch(UUID)
    expect(normalizeProfile(on).contributor).toBe(on.contributor)
  })

  it('rejects unknown countries', () => {
    expect(normalizeProfile({ country: 'IN' }).country).toBe('IN')
    expect(normalizeProfile({ country: 'India' }).country).toBeNull()
    expect(normalizeProfile({ country: 'XX' }).country).toBeNull()
  })

  it('keeps a region only when it is one of the country\'s fixed codes', () => {
    expect(normalizeProfile({ country: 'GB', region: 'SCT' }).region).toBe('SCT')
    expect(normalizeProfile({ country: 'IN', region: 'S' }).region).toBe('S')
    // Text typed into the old free-text field is dropped; the rest of the profile stays.
    expect(normalizeProfile({ seen: true, country: 'GB', region: 'Leeds', contribute: true, contributor: profile.contributor }))
      .toEqual({ seen: true, country: 'GB', region: null, contribute: true, contributor: profile.contributor })
    expect(normalizeProfile({ country: 'GB', region: 'sct' }).region).toBeNull()
    expect(normalizeProfile({ country: 'IN', region: 'SCT' }).region).toBeNull() // another country's code
    expect(normalizeProfile({ country: 'FR', region: 'N' }).region).toBeNull() // France offers none
    expect(normalizeProfile({ region: 'SCT' }).region).toBeNull() // no country
    expect(normalizeProfile({ country: 'GB', region: '' }).region).toBeNull()
    expect(normalizeProfile({ country: 'GB', region: 'toString' }).region).toBeNull()
    expect(normalizeProfile({ country: 'GB', region: 7 }).region).toBeNull()
  })
})

describe('countries', () => {
  it('lists every ISO country once, with a name', () => {
    const list = countries()
    expect(list).toHaveLength(249)
    expect(new Set(list.map((c) => c.code)).size).toBe(249)
    expect(list.every((c) => /^[A-Z]{2}$/.test(c.code) && c.name && c.name !== c.code)).toBe(true)
    expect(Object.keys(HOME_ACCENT).every(isCountryCode)).toBe(true)
  })
})

describe('regions', () => {
  it('is the same fixed list the ingest script accepts', () => {
    expect(REGIONS).toEqual(fixture.regions)
  })

  it('offers regions only for a few countries, by code', () => {
    expect(regionsOf('GB').map(([code]) => code)).toEqual(['ENG_S', 'ENG_N', 'SCT', 'WLS', 'NIR'])
    expect(regionsOf('US').map(([, name]) => name)).toEqual(['Northeast', 'South', 'Midwest', 'West'])
    expect(regionsOf('FR')).toEqual([])
    expect(regionsOf('')).toEqual([])
    expect(Object.keys(REGIONS).every(isCountryCode)).toBe(true)
    expect(isRegion('GB', 'SCT')).toBe(true)
    expect(isRegion('GB', 'constructor')).toBe(false)
    expect(regionName('GB', 'SCT')).toBe('Scotland')
    expect(regionName('GB', 'Leeds')).toBeNull()
  })

  const scot = { ...profile, country: 'GB', region: 'SCT' }
  const american = { ...profile, country: 'US', region: null }
  const built = (who, targetAccent, naturalVoice = false) => buildContribution({
    profile: who, targetAccent, naturalVoice, word: 'better', score: 8, features,
    quality: cleanQuality, confidence: 0.9, soundsHeard: 4, soundsMatched: 100, soundsSameClass: 100, modelVersion: 'abc',
  })
  const route = (...args) => routeOf(built(...args))

  it('treats a Scot practising British as an attempt', () => {
    // Standard British is not how Scotland sounds, nor a sample of Britain: it is an imitation.
    expect(route(scot, 'rp')).toMatchObject({ kind: 'attempt', keys: ['GB>rp'] })
  })

  it('treats an American practising American as an attempt', () => {
    expect(route(american, 'ga')).toMatchObject({ kind: 'attempt', keys: ['US>ga'] })
    expect(route({ ...american, region: 'S' }, 'ga')).toMatchObject({ kind: 'attempt', keys: ['US>ga'] })
  })

  it('treats a Scot in their normal voice as native to country and region', () => {
    expect(route(scot, 'rp', true)).toMatchObject({ kind: 'native', keys: ['GB', 'GB-SCT'] })
    expect(route({ ...scot, region: null }, 'rp', true)).toMatchObject({ kind: 'native', keys: ['GB'] })
  })

  it('stores the home-accent label without routing on it', () => {
    for (const [country, accent] of Object.entries(HOME_ACCENT)) {
      const take = built({ ...profile, country, region: null }, accent)
      expect(take.matches_home_accent).toBe(true)
      expect(routeOf(take)).toMatchObject({ kind: 'attempt', keys: [`${country}>${accent}`] })
    }
  })
})

describe('normal-voice sentences', () => {
  const lexicon = JSON.parse(readFileSync(new URL('../../public/lexicon/lexicon.json', import.meta.url), 'utf8'))

  it('offers five different short sentences the dictionary knows in every accent', () => {
    expect(SENTENCES).toHaveLength(5)
    expect(new Set(SENTENCES).size).toBe(5)
    for (const sentence of SENTENCES) {
      for (const accent of Object.keys(lexicon.accents)) {
        expect(textEntry(lexicon, sentence, accent).phonemes.length).toBeGreaterThan(10)
      }
    }
  })

  it('gives the range of lengths a sentence has across the dictionaries', () => {
    for (const sentence of SENTENCES) {
      const [fewest, most] = targetSounds(lexicon, sentence)
      expect(fewest).toBeGreaterThan(10)
      expect(most).toBeGreaterThanOrEqual(fewest)
      expect(most - fewest).toBeLessThan(4)
    }
    expect(targetSounds(lexicon, SENTENCES[0])).toEqual([20, 20])
  })
})
