// Accent summaries a visitor has agreed to keep ("Help it learn").
//
// A contribution is never audio. It is the model's 256-number summary of one take
// plus the labels needed to use it honestly: where the speaker is from, which
// accent they were attempting, and whether that was their own. Contributions stay
// in this browser (IndexedDB); nothing uploads them. The "Your data" page lists,
// exports and deletes them, and scripts/learn_regions.py reads the export.
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
  minConfidence: 0.6,
  minSoundsAttempted: 70,
}

// The practice accent that is native to a country. A take counts as that
// country's own accent only when the speaker chose this accent (or recorded in
// their normal voice). Kept deliberately narrow; scripts/learn_regions.py holds the same table.
export const HOME_ACCENT = { US: 'ga', GB: 'rp', AU: 'au' }

const run = (mode, work) => runIn('contributions', mode, work)

// Every check with its outcome, so the "Your data" page can show why a take was or was not kept.
// `quality` comes from engine/cleanup.js; `soundsAttempted` is the completeness metric (0..100).
export function checkViability({ quality, confidence, soundsAttempted, features, featureCount }) {
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
    ['attempted', 'Most target sounds attempted', soundsAttempted >= limits.minSoundsAttempted,
      `${Math.round(soundsAttempted)}% of target sounds (needs ${limits.minSoundsAttempted}%)`],
    ['features', 'Summary is well formed', featuresOk,
      featuresOk ? `${featureCount} finite numbers` : 'the model did not return a usable summary'],
  ].map(([id, label, passed, detail]) => ({ id, label, passed: Boolean(passed), detail }))
  return { viable: checks.every((check) => check.passed), checks }
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
  profile, targetAccent, naturalVoice, word, score, features, quality, confidence, soundsAttempted, modelVersion,
}) {
  return {
    id: crypto.randomUUID(),
    created: new Date().toISOString(),
    contributor: profile.contributor,
    country: profile.country,
    region: profile.region || null,
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
      sounds_attempted: Math.round(soundsAttempted),
    },
    model_version: modelVersion,
  }
}

// What a contribution will be used for once sharing exists. Mirrors `route` in scripts/learn_regions.py.
export function routeOf(contribution) {
  return contribution.natural_voice || contribution.matches_home_accent
    ? { kind: 'native', label: `how ${contribution.country} sounds` }
    : { kind: 'attempt', label: `${contribution.country} attempting ${contribution.target_accent}` }
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

// Decide what to do with one analysed take, store it if it qualifies, and log the outcome.
// Returns { stored, reason, checks, contribution }.
export async function considerTake({ profile, targetAccent, naturalVoice = false, word, score, heard, quality, soundsAttempted, featureCount }) {
  const { viable, checks } = checkViability({
    quality, confidence: heard.confidence, soundsAttempted, features: heard.features, featureCount,
  })
  let contribution = null
  let stored = false
  let reason = checkEligibility(profile, 0)
  if (!reason && !viable) {
    reason = `did not pass: ${checks.filter((check) => !check.passed).map((check) => check.label.toLowerCase()).join(', ')}`
  }
  if (!reason) {
    contribution = buildContribution({
      profile, targetAccent, naturalVoice, word, score, features: heard.features, quality,
      confidence: heard.confidence, soundsAttempted, modelVersion: heard.modelVersion,
    })
    try {
      stored = await saveContribution(contribution)
      if (!stored) reason = checkEligibility(profile, MAX_STORED)
    } catch {
      reason = 'this browser could not open its local storage'
    }
  }
  const failedOn = stored ? null : reason
  logDecision({
    at: new Date().toISOString(), word, target_accent: naturalVoice ? null : targetAccent, natural_voice: naturalVoice,
    stored, reason: failedOn, id: stored ? contribution.id : null, checks,
  })
  return { stored, reason: failedOn, checks, contribution: stored ? contribution : null }
}
