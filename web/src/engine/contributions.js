// Accent summaries a visitor has agreed to keep ("Help it learn").
//
// A contribution is never audio. It is the model's 256-number summary of one take
// plus the labels needed to use it honestly: where the speaker is from, which
// accent they were attempting, and whether that was their own. Contributions stay
// in this browser (IndexedDB); nothing uploads them. The "Your data" page lists,
// exports and deletes them, and scripts/learn_regions.py reads the export.
import { regionName } from '../lib/regions'
import { run as runIn } from './db'

export const EXPORT_VERSION = 2
// One device should not outweigh a region.
export const MAX_STORED = 60

// A take is kept only if it is clean enough to say something about an accent.
// The SNR bar is higher than the one for practising (hooks/useRecorder.js): with
// noise 15 dB below the speech the model still mishears about a quarter more sounds
// than on clean audio, even after cleanup (Docs/model/capture_results.json).
export const VIABILITY = {
  minSpeechSeconds: 0.25,
  minSnrDb: 20,
  maxClipped: 0.005, // loud but unclipped speech brushes full scale now and then; real clipping sits there
  // Lowered from 0.6, which dropped 12% of clean single words from native speakers. At 0.5 it
  // drops 1%, and 8% of words in noise 10 dB below the speech (Docs/model/take_checks.json).
  minConfidence: 0.5,
  // Practice takes: at least this share of the target sounds heard exactly as written. Measured
  // by scripts/evaluate_match.py (Docs/model/match_threshold.json) on single words: it keeps 76%
  // of words from learners rated 5/10 or lower and 98% from Scottish and Irish speakers, and lets
  // through 15% of wrong words. A looser, class-based match was tried and separated them worse.
  minSoundsMatched: 30,
  // Sounds heard, as a multiple of the target's. More than this on a practice take usually
  // means the model transcribed background noise around the word.
  minLengthRatio: 0.5,
  maxPracticeLengthRatio: 1.5,
  // Normal-voice takes are not compared with any dictionary: there is no right way to sound.
  // They only have to be about as long as the sentence that was shown, with more room.
  maxVoiceLengthRatio: 2,
}

export const EXTRA_SOUNDS = 'extra sounds, likely background noise'
export const TOO_FEW_SOUNDS = 'too few sounds heard'
export const RETAKE_FOR_EXTRA_SOUNDS = 'I heard extra sounds around that, probably background noise. Try again somewhere quieter.'

// The practice accent that is the standard of a country. It only sets the stored label
// `matches_home_accent`; it plays no part in what a take is used for (see routeOf).
// scripts/learn_regions.py holds the same table.
export const HOME_ACCENT = { US: 'ga', GB: 'rp', AU: 'au' }

const run = (mode, work) => runIn('contributions', mode, work)

// Every check with its outcome, so the "Your data" page can show why a take was or was not kept.
// `quality` comes from engine/cleanup.js. `soundsHeard` is how many sounds the model heard and
// `targetSounds` the [fewest, most] the target has (one number twice for a practice word, the
// range across the accent dictionaries for a normal-voice sentence). A practice take also
// passes `soundsMatched`, the share of target sounds heard as written (0..100).
export function checkViability({
  quality, confidence, naturalVoice = false, soundsMatched, soundsHeard, targetSounds, features, featureCount,
}) {
  const limits = VIABILITY
  const featuresOk = Boolean(features) && features.length === featureCount && Array.from(features).every(Number.isFinite)
  const checks = [
    ['speech', 'Speech found', quality.foundSpeech && quality.speechSeconds >= limits.minSpeechSeconds,
      `${quality.speechSeconds.toFixed(2)} s of speech (needs ${limits.minSpeechSeconds} s)`],
    ['snr', 'Quiet background', quality.snrDb >= limits.minSnrDb,
      `${quality.snrDb.toFixed(1)} dB above the background (needs ${limits.minSnrDb} dB)`],
    ['clipping', 'Not clipped', quality.clipped <= limits.maxClipped,
      `${(quality.clipped * 100).toFixed(2)}% of samples at full scale (limit ${limits.maxClipped * 100}%)`],
    ['confidence', 'Model was confident', confidence >= limits.minConfidence,
      `confidence ${confidence.toFixed(2)} (needs ${limits.minConfidence})`],
  ]
  const [fewest, most] = allowedSounds(targetSounds, naturalVoice)
  const lengthReason = soundsHeard > most ? EXTRA_SOUNDS : TOO_FEW_SOUNDS
  checks.push(['length', naturalVoice ? 'About as long as the sentence' : 'About as long as the word',
    soundsHeard >= fewest && soundsHeard <= most, `${soundsHeard} sounds heard (needs ${fewest} to ${most})`, lengthReason])
  if (!naturalVoice) {
    checks.push(['matched', 'Recognisably the target word', soundsMatched >= limits.minSoundsMatched,
      `${Math.round(soundsMatched)}% of target sounds heard as written (needs ${limits.minSoundsMatched}%)`])
  }
  checks.push(['features', 'Summary is well formed', featuresOk,
    featuresOk ? `${featureCount} finite numbers` : 'the model did not return a usable summary'])

  const labelled = checks.map(([id, label, passed, detail, reason]) => ({
    id, label, passed: Boolean(passed), detail, ...(reason && !passed ? { reason } : {}),
  }))
  return { viable: labelled.every((check) => check.passed), checks: labelled }
}

// The fewest and most sounds a take may have for its target.
export function allowedSounds(targetSounds, naturalVoice = false) {
  const limits = VIABILITY
  return [
    Math.ceil(limits.minLengthRatio * targetSounds[0]),
    Math.floor((naturalVoice ? limits.maxVoiceLengthRatio : limits.maxPracticeLengthRatio) * targetSounds[1]),
  ]
}

// Whether this browser is set up to keep takes at all, apart from how good a take is.
export function checkEligibility(profile, storedCount) {
  if (!profile.contribute) return '"Help it learn" is off'
  if (!profile.country) return 'no country in your profile, so the take cannot be placed in a region'
  if (storedCount >= MAX_STORED) return `this device already holds ${MAX_STORED} takes, the most it keeps`
  return null
}

const round = (value, digits) => Math.round(value * 10 ** digits) / 10 ** digits

// The stored record. `targetAccent` is null for a take in the speaker's normal voice.
export function buildContribution({
  profile, targetAccent, naturalVoice, word, score, features, quality, confidence, soundsHeard, soundsMatched, soundsSameClass,
  modelVersion,
}) {
  return {
    id: crypto.randomUUID(),
    created: new Date().toISOString(),
    contributor: profile.contributor,
    country: profile.country,
    region: profile.region ?? null,
    target_accent: naturalVoice ? null : targetAccent,
    natural_voice: Boolean(naturalVoice),
    matches_home_accent: !naturalVoice && HOME_ACCENT[profile.country] === targetAccent,
    word,
    score: naturalVoice ? null : score,
    features: Array.from(features, (value) => round(value, 4)),
    quality: {
      speech_seconds: quality.speechSeconds,
      snr_db: quality.snrDb,
      clipped: round(quality.clipped, 6),
      confidence: round(confidence, 3),
      sounds_heard: soundsHeard,
      // Against the target word. A normal-voice take is not compared with a dictionary.
      sounds_matched: naturalVoice ? null : Math.round(soundsMatched),
      sounds_same_class: naturalVoice ? null : Math.round(soundsSameClass),
    },
    model_version: modelVersion,
  }
}

// What a contribution will be used for once sharing exists. Mirrors `route` in scripts/learn_regions.py.
//
//   normal voice         -> native: country, and country-region when one was given
//   any practice take    -> attempt: country>target accent
//
// Only a take in the speaker's normal voice says how a place sounds. Every practice take is
// someone aiming at a target, even when the target is their own country's standard accent.
export function routeOf(contribution) {
  const { country, region } = contribution
  if (contribution.natural_voice) {
    const place = regionName(country, region)
    return {
      kind: 'native',
      keys: place ? [country, `${country}-${region}`] : [country],
      label: `how ${country}${place ? ` and ${place}` : ''} sounds`,
    }
  }
  return { kind: 'attempt', keys: [`${country}>${contribution.target_accent}`], label: `${country} attempting ${contribution.target_accent}` }
}

// Newest first.
export async function getContributions() {
  const rows = await run('readonly', (store) => store.getAll())
  return rows.sort((a, b) => b.created.localeCompare(a.created))
}

export const countContributions = () => run('readonly', (store) => store.count())

// Stores the take unless the device is at its cap. Resolves true when stored.
export async function saveContribution(contribution) {
  if (await countContributions() >= MAX_STORED) return false
  await run('readwrite', (store) => store.add(contribution))
  return true
}

export const clearContributions = () => run('readwrite', (store) => store.clear())

export function exportContributions(takes) {
  return JSON.stringify({
    app: 'phonemepro', kind: 'contributions', version: EXPORT_VERSION, exported: new Date().toISOString(), takes,
  }, null, 1)
}

// A short record of what happened to recent takes, kept only for this tab
// (sessionStorage) so the "Your data" page can explain takes that were not stored.
// It holds no summary numbers and is never exported.
const LOG_KEY = 'phonemepro.decisions'
const LOG_LENGTH = 30

export function recentDecisions() {
  try {
    const parsed = JSON.parse(sessionStorage.getItem(LOG_KEY))
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function logDecision(decision) {
  try {
    sessionStorage.setItem(LOG_KEY, JSON.stringify([decision, ...recentDecisions()].slice(0, LOG_LENGTH)))
  } catch {
    // No session storage: the page simply has nothing to explain.
  }
}

export function clearDecisions() {
  try {
    sessionStorage.removeItem(LOG_KEY)
  } catch {
    // Nothing to clear.
  }
}

// Where kept takes go. The app uses this browser's IndexedDB and the per-tab log.
const deviceStore = { save: saveContribution, log: logDecision }

// A store that lives only in memory, for the label check page (pages/CheckPage.jsx): it
// applies the same cap, touches nothing real, and is gone when the page is left.
export function createMemoryStore() {
  const takes = []
  return {
    save: async (contribution) => {
      if (takes.length >= MAX_STORED) return false
      takes.push(contribution)
      return true
    },
    log: () => {},
    all: () => [...takes],
    remove: (id) => {
      const at = takes.findIndex((take) => take.id === id)
      if (at !== -1) takes.splice(at, 1)
    },
  }
}

// Decide what to do with one analysed take, store it if it qualifies, and log the outcome.
// Returns { stored, reason, checks, contribution, retake }. `retake` is set when the take had
// extra sounds, whether or not anything is being kept: the learner should be told either way.
export async function considerTake({
  profile, targetAccent, naturalVoice = false, word, score, heard, quality, soundsMatched, soundsSameClass, targetSounds,
  featureCount, store = deviceStore,
}) {
  const { viable, checks } = checkViability({
    quality, confidence: heard.confidence, naturalVoice, soundsMatched, soundsHeard: heard.phonemes.length, targetSounds,
    features: heard.features, featureCount,
  })
  let contribution = null
  let stored = false
  let reason = checkEligibility(profile, 0)
  const failing = checks.filter((check) => !check.passed)
  const retake = failing.some((check) => check.reason === EXTRA_SOUNDS)
  if (!reason && !viable) {
    reason = failing.find((check) => check.reason)?.reason
      ?? `did not pass: ${failing.map((check) => check.label.toLowerCase()).join(', ')}`
  }
  if (!reason) {
    contribution = buildContribution({
      profile, targetAccent, naturalVoice, word, score, features: heard.features, quality,
      confidence: heard.confidence, soundsHeard: heard.phonemes.length, soundsMatched, soundsSameClass,
      modelVersion: heard.modelVersion,
    })
    try {
      stored = await store.save(contribution)
      if (!stored) reason = checkEligibility(profile, MAX_STORED)
    } catch {
      reason = 'this browser could not open its local storage'
    }
  }
  const failedOn = stored ? null : reason
  store.log({
    at: new Date().toISOString(), word, target_accent: naturalVoice ? null : targetAccent, natural_voice: naturalVoice,
    stored, reason: failedOn, id: stored ? contribution.id : null, checks,
  })
  return { stored, reason: failedOn, checks, contribution: stored ? contribution : null, retake }
}
