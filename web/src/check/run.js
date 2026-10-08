// Runs one analysed take through the app's own scoring and keeping rules for a check step,
// against a sandbox store, and reports what happened in the terms the step table uses.
import { considerTake, routeOf } from '../engine/contributions'
import { textEntry } from '../engine/lexicon'
import { readingOf } from '../engine/naturalVoice'
import { scoreAttempt, stripStress } from '../engine/scoring'

const RETAKE_REASONS = { silent: 'no speech', noisy: 'too noisy to score', short: 'too short' }

const EMPTY = {
  scored: false, score: null, kept: false, reason: null, failed_checks: [], kind: null, joins: null,
  natural_voice: null, matches_home_accent: null, country: null, region: null, target_accent: null,
  first_sound: null, snr_db: null, retake_notice: false, stored_added: 0, take_id: null, detail: '',
}

// The take was refused before the model heard it, and the app asked for another.
export function retakeOutcome(detail = {}) {
  const snr = detail.quality?.snrDb ?? null
  return {
    ...EMPTY,
    reason: RETAKE_REASONS[detail.reason] ?? detail.reason ?? 'refused',
    retake_notice: true,
    snr_db: snr,
    detail: `retake asked: ${detail.reason ?? 'unknown'}${snr === null ? '' : ` · snr ${snr} dB`}`,
  }
}

// `heard` is what engine/model.js returned; `quality` what engine/cleanup.js measured.
export async function settleTake({ step, lexicon, heard, quality, store, contributor }) {
  const profile = { seen: true, ...step.profile, contributor: step.profile.contribute ? contributor : null }
  let score = null
  let metrics
  if (step.naturalVoice) {
    metrics = readingOf(lexicon, step.word, heard)
  } else {
    const scored = scoreAttempt(textEntry(lexicon, step.word, step.accent).phonemes, heard.phonemes, heard.confidence, heard.margin, {
      nativeScore: heard.nativeScore ?? null, spans: heard.spans ?? null,
    })
    score = scored.score
    metrics = { soundsAttempted: scored.metrics.completeness, soundsMatched: scored.metrics.accuracy }
  }

  const before = store.all().length
  const decision = await considerTake({
    profile, targetAccent: step.accent, naturalVoice: Boolean(step.naturalVoice), word: step.word, score,
    heard, quality, ...metrics, featureCount: heard.featureCount, store,
  })
  const take = decision.contribution
  const route = take ? routeOf(take) : null

  return {
    ...EMPTY,
    scored: true,
    score,
    kept: decision.stored,
    reason: decision.reason,
    failed_checks: decision.checks.filter((check) => !check.passed).map((check) => check.id),
    kind: route?.kind ?? null,
    joins: route?.keys ?? null,
    natural_voice: take?.natural_voice ?? null,
    matches_home_accent: take?.matches_home_accent ?? null,
    country: take?.country ?? null,
    region: take?.region ?? null,
    target_accent: take?.target_accent ?? null,
    first_sound: heard.phonemes.length ? stripStress(heard.phonemes[0]) : null,
    snr_db: quality.snrDb,
    stored_added: store.all().length - before,
    take_id: take?.id ?? null,
    detail: [
      `heard ${heard.phonemes.join(' ') || 'nothing'}`,
      score === null ? null : `score ${score}`,
      `speech ${quality.speechSeconds} s`,
      `snr ${quality.snrDb} dB`,
      `clipped ${(quality.clipped * 100).toFixed(2)}%`,
      `confidence ${heard.confidence.toFixed(2)}`,
      `attempted ${Math.round(metrics.soundsAttempted)}%`,
      `matched ${Math.round(metrics.soundsMatched)}%`,
    ].filter(Boolean).join(' · '),
  }
}
