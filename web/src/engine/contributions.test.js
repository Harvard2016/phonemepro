import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  buildContribution, checkEligibility, checkViability, exportContributions, HOME_ACCENT, MAX_STORED, routeOf, VIABILITY,
} from './contributions'
import { cleanRegion, normalizeProfile } from './profile'
import { countries, isCountryCode } from '../lib/countries'

// The same file is ingested by scripts/learn_regions.py in tests/test_learn_regions.py.
const fixture = JSON.parse(
  readFileSync(new URL('../../../tests/fixtures/contribution_export.json', import.meta.url), 'utf8'),
)

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
const features = Array.from({ length: 256 }, (_, i) => Math.sin(i) * 1.23456789)
const cleanQuality = { foundSpeech: true, speechSeconds: 0.62, snrDb: 31.4, clipped: 0 }
const good = { quality: cleanQuality, confidence: 0.9, soundsAttempted: 100, features, featureCount: 256 }
const profile = { seen: true, country: 'GB', region: 'Leeds', contribute: true, contributor: '3f2b8c1e-5a47-4d09-9b61-7e0c2a4d8f13' }

const failed = (input) => checkViability({ ...good, ...input }).checks.filter((c) => !c.passed).map((c) => c.id)

describe('checkViability', () => {
  it('passes a clean, confident, complete take and explains every check', () => {
    const { viable, checks } = checkViability(good)
    expect(viable).toBe(true)
    expect(checks.map((c) => c.id)).toEqual(['speech', 'snr', 'clipping', 'confidence', 'attempted', 'features'])
    expect(checks.every((c) => c.label && c.detail)).toBe(true)
  })

  it('names the check that failed', () => {
    expect(failed({ quality: { ...cleanQuality, foundSpeech: false, speechSeconds: 0 } })).toEqual(['speech'])
    expect(failed({ quality: { ...cleanQuality, speechSeconds: 0.1 } })).toEqual(['speech'])
    expect(failed({ quality: { ...cleanQuality, snrDb: VIABILITY.minSnrDb - 0.1 } })).toEqual(['snr'])
    expect(failed({ quality: { ...cleanQuality, clipped: 0.01 } })).toEqual(['clipping'])
    expect(failed({ confidence: 0.3 })).toEqual(['confidence'])
    expect(failed({ soundsAttempted: 50 })).toEqual(['attempted'])
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
      soundsAttempted: VIABILITY.minSoundsAttempted,
    }).viable).toBe(true)
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
    quality: cleanQuality, confidence: 0.9123, soundsAttempted: 100, modelVersion: 'abc', ...changes,
  })

  it('has exactly the fields of an exported take, and no audio', () => {
    const take = build()
    expect(Object.keys(take)).toEqual(Object.keys(fixture.export.takes[0]))
    expect(Object.keys(take.quality)).toEqual(Object.keys(fixture.export.takes[0].quality))
    expect(take.id).toMatch(UUID)
    expect(take.id).not.toBe(build().id)
    expect(take.features).toHaveLength(256)
    expect(take.features[1]).toBe(1.0389)
    expect(take.quality.confidence).toBe(0.912)
    expect(JSON.stringify(take)).not.toMatch(/samples|wav|audio|blob/i)
  })

  it('labels a take by whose accent it is', () => {
    expect(build()).toMatchObject({ country: 'GB', region: 'Leeds', target_accent: 'rp', natural_voice: false, matches_home_accent: true, score: 8.4 })
    expect(build({ targetAccent: 'ga' })).toMatchObject({ target_accent: 'ga', matches_home_accent: false })
    expect(build({ profile: { ...profile, country: 'IN', region: '' } })).toMatchObject({ country: 'IN', region: null, matches_home_accent: false })
    // In their normal voice nobody is attempting anything, whatever accent the practice page was set to.
    expect(build({ naturalVoice: true, targetAccent: 'ga', score: null }))
      .toMatchObject({ target_accent: null, natural_voice: true, matches_home_accent: false, score: null })
  })

  it('routes takes the way the ingest script does', () => {
    expect(HOME_ACCENT).toEqual(fixture.home_accent)
    for (const take of fixture.export.takes) {
      const [kind, key] = fixture.routes[take.id]
      expect(routeOf(take).kind).toBe(kind)
      const rebuilt = build({
        profile: { ...profile, country: take.country }, targetAccent: take.target_accent, naturalVoice: take.natural_voice,
      })
      expect(rebuilt.matches_home_accent).toBe(take.matches_home_accent)
      expect(routeOf(rebuilt).kind).toBe(kind)
      expect(key.startsWith(take.country)).toBe(true)
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
    expect(normalizeProfile(null)).toEqual({ seen: false, country: null, region: '', contribute: false, contributor: null })
    expect(normalizeProfile('junk').contribute).toBe(false)
  })

  it('only turns contributing on for an explicit true', () => {
    expect(normalizeProfile({ contribute: 'yes' }).contribute).toBe(false)
    expect(normalizeProfile({ contribute: 1 }).contributor).toBeNull()
    const on = normalizeProfile({ contribute: true })
    expect(on.contributor).toMatch(UUID)
    expect(normalizeProfile(on).contributor).toBe(on.contributor)
  })

  it('rejects unknown countries and tidies the region', () => {
    expect(normalizeProfile({ country: 'IN' }).country).toBe('IN')
    expect(normalizeProfile({ country: 'India' }).country).toBeNull()
    expect(normalizeProfile({ country: 'XX' }).country).toBeNull()
    expect(cleanRegion('  West\n Yorkshire\u0000 ')).toBe('West Yorkshire')
    expect(cleanRegion('x'.repeat(200))).toHaveLength(60)
    expect(cleanRegion(undefined)).toBe('')
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
