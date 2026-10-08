// Runs one analysed take through the app's own scoring and keeping rules for a check step,
// against a sandbox store, and reports what happened in the terms the step table uses.
import { allowedSounds, considerTake, routeOf } from '../engine/contributions'
import { textEntry } from '../engine/lexicon'
import { targetSounds } from '../engine/naturalVoice'
import { scoreAttempt, stripStress } from '../engine/scoring'
import { soundsSameClass } from '../engine/soundMatch'
import { heardNoSpeech } from '../engine/takeChecks'

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
  // As on the Practice page: if the model heard next to nothing, this was silence.
  if (heardNoSpeech(heard.phonemes, quality.speechSeconds)) {
    const refused = retakeOutcome({ reason: 'silent', quality })
    return { ...refused, detail: `${refused.detail} · model heard ${heard.phonemes.join(' ') || 'nothing'} in ${quality.speechSeconds} s` }
  }

  let score = null
  let measures
  if (step.naturalVoice) {
    measures = { targetSounds: targetSounds(lexicon, step.word) }
  } else {
    const target = textEntry(lexicon, step.word, step.accent).phonemes
    const scored = scoreAttempt(target, heard.phonemes, heard.confidence, heard.margin, {
      nativeScore: heard.nativeScore ?? null, spans: heard.spans ?? null,
    })
    score = scored.score
    measures = {
      soundsMatched: scored.metrics.accuracy, soundsSameClass: soundsSameClass(target, heard.phonemes),
      targetSounds: [target.length, target.length],
    }
  }

  const before = store.all().length
  const decision = await considerTake({
    profile, targetAccent: step.accent, naturalVoice: Boolean(step.naturalVoice), word: step.word, score,
    heard, quality, ...measures, featureCount: heard.featureCount, store,
  })
  const take = decision.contribution
  const allowed = allowedSounds(measures.targetSounds, Boolean(step.naturalVoice))
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
    retake_notice: decision.retake,
    stored_added: store.all().length - before,
    take_id: take?.id ?? null,
    detail: [
      `heard ${heard.phonemes.join(' ') || 'nothing'}`,
      score === null ? null : `score ${score}`,
      `speech ${quality.speechSeconds} s`,
      `snr ${quality.snrDb} dB`,
      `clipped ${(quality.clipped * 100).toFixed(2)}%`,
      `confidence ${heard.confidence.toFixed(2)}`,
      `${heard.phonemes.length} sounds heard, ${allowed.join(' to ')} allowed for a target of ${
        measures.targetSounds[0] === measures.targetSounds[1] ? measures.targetSounds[0] : measures.targetSounds.join(' to ')}`,
      step.naturalVoice
        ? null
        : `heard as written ${Math.round(measures.soundsMatched)}% · same class ${Math.round(measures.soundsSameClass)}%`,
    ].filter(Boolean).join(' · '),
  }
}
