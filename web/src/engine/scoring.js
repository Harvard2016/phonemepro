// Phoneme alignment and pronunciation scoring, in the browser.
//
// This mirrors pronunciation/scoring.py for the phoneme-model path. Both are
// checked against the same cases in tests/fixtures/scoring_cases.json, so a
// change to one that is not made to the other fails a test.
import { CLARITY_COACHING, PHONEME_TIPS } from './tips'

const PHONEME_ACCURACY_WEIGHT = 0.9
const ACOUSTIC_WEIGHT = 0.1
const CONFIDENCE_RANGE = [0.6, 0.9]
const MARGIN_RANGE = [0.5, 0.9]
const LEARNED_SCORE_WEIGHT = 0.6
const LEARNED_SCORE_CAP_POINTS = 3.0

// The model never emits these, so it cannot judge them either way.
export const UNSCORABLE_PHONEMES = new Set(['ZH', 'OY'])
const UNSCORABLE_TIP = 'The model cannot judge this sound yet, so it is not counted.'

export const stripStress = (phoneme) => phoneme.replace(/[0-2]$/, '')
export const stressOf = (phoneme) => (/[0-2]$/.test(phoneme) ? phoneme.slice(-1) : null)

export const getTip = (phoneme) =>
  PHONEME_TIPS[stripStress(phoneme)] ?? `Focus on clearly articulating '${phoneme}'.`

// Round a non-negative number the way Python's round() does: on the exact binary
// value, with true ties going to the even neighbour. Multiplying by a power of ten
// first (the usual JavaScript idiom) turns values such as 1.115 into false ties.
export function roundHalfEven(value, digits = 0) {
  const exact = value.toFixed(40)
  const point = exact.indexOf('.')
  const tail = exact.slice(point + 1 + digits)
  if (tail[0] === '5' && /^0*$/.test(tail.slice(1))) {
    const down = Number(exact.slice(0, point + 1 + digits))
    const lastDigit = Number(exact[point + digits] === '.' ? exact[point - 1] : exact[point + digits])
    return lastDigit % 2 === 0 ? down : Number((down + 10 ** -digits).toFixed(digits))
  }
  return Number(value.toFixed(digits))
}

function buildClarityCoaching(targetPhonemes) {
  const coaching = []
  for (const phoneme of targetPhonemes) {
    const tip = CLARITY_COACHING[stripStress(phoneme)]
    if (tip && !coaching.includes(tip)) coaching.push(tip)
  }
  if (coaching.length === 0) {
    coaching.push('Slow the word down and keep each sound clean without adding extra vowel sounds.')
  }
  return coaching.slice(0, 3)
}

// Minimum edit distance alignment of two token sequences.
export function align(ref, hyp) {
  const rows = ref.length + 1
  const cols = hyp.length + 1
  const cost = Array.from({ length: rows }, (_, i) => {
    const row = new Array(cols).fill(0)
    row[0] = i
    return row
  })
  for (let j = 0; j < cols; j += 1) cost[0][j] = j

  for (let i = 1; i < rows; i += 1) {
    for (let j = 1; j < cols; j += 1) {
      const diagonal = cost[i - 1][j - 1] + (ref[i - 1] !== hyp[j - 1] ? 1 : 0)
      cost[i][j] = Math.min(diagonal, cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    }
  }

  const alignment = { ops: [], hits: 0, substitutions: 0, deletions: 0, insertions: 0 }
  let i = ref.length
  let j = hyp.length
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && cost[i][j] === cost[i - 1][j - 1] + (ref[i - 1] !== hyp[j - 1] ? 1 : 0)) {
      if (ref[i - 1] === hyp[j - 1]) {
        alignment.ops.push({ kind: 'equal', refIndex: i - 1, hypIndex: j - 1 })
        alignment.hits += 1
      } else {
        alignment.ops.push({ kind: 'substitute', refIndex: i - 1, hypIndex: j - 1 })
        alignment.substitutions += 1
      }
      i -= 1
      j -= 1
    } else if (i > 0 && cost[i][j] === cost[i - 1][j] + 1) {
      alignment.ops.push({ kind: 'delete', refIndex: i - 1, hypIndex: null })
      alignment.deletions += 1
      i -= 1
    } else {
      alignment.ops.push({ kind: 'insert', refIndex: null, hypIndex: j - 1 })
      alignment.insertions += 1
      j -= 1
    }
  }
  alignment.ops.reverse()
  return alignment
}

const errorsOf = (alignment) => alignment.substitutions + alignment.deletions + alignment.insertions

const scale = (value, [low, high]) => Math.max(0, Math.min(1, (value - low) / (high - low)))

const acousticCertainty = (confidence, margin) =>
  0.7 * scale(confidence, CONFIDENCE_RANGE) + 0.3 * scale(margin, MARGIN_RANGE)

export function combineScores(alignmentScore, learnedScore) {
  const blended = LEARNED_SCORE_WEIGHT * learnedScore + (1 - LEARNED_SCORE_WEIGHT) * alignmentScore
  return Math.max(0, Math.min(10, blended, alignmentScore + LEARNED_SCORE_CAP_POINTS))
}

function emptyAttempt(targetPhonemes) {
  return {
    score: 0,
    wer: 1.0,
    per_with_stress: 1.0,
    metrics: { accuracy: 0, completeness: 0, fluency: 0 },
    stress: { matched: 0, total: 0 },
    phoneme_results: targetPhonemes.map((p) => ({ phoneme: p, status: 'missed', heard: null, tip: getTip(p) })),
    errors: [],
    feedback: "I didn't hear anything clearly. Try again?",
    coaching_tips: [],
  }
}

// Score one attempt and build per-phoneme feedback.
// `spans` are optional [start, end] seconds for each predicted phoneme.
export function scoreAttempt(targetPhonemes, predictedPhonemes, confidence, margin, { nativeScore = null, spans = null } = {}) {
  if (predictedPhonemes.length === 0) return emptyAttempt(targetPhonemes)

  const ref = targetPhonemes.map(stripStress)
  const hyp = predictedPhonemes.map(stripStress)
  const alignment = align(ref, hyp)
  const totalTarget = Math.max(targetPhonemes.length, 1)

  // Sounds the phoneme model cannot recognize are given the benefit of the doubt.
  const unscored = new Set()
  for (const op of alignment.ops) {
    if ((op.kind === 'substitute' || op.kind === 'delete') && UNSCORABLE_PHONEMES.has(ref[op.refIndex])) {
      unscored.add(op.refIndex)
      alignment.hits += 1
      if (op.kind === 'substitute') alignment.substitutions -= 1
      else alignment.deletions -= 1
    }
  }

  const per = errorsOf(alignment) / totalTarget
  const perWithStress = errorsOf(align(targetPhonemes, predictedPhonemes)) / totalTarget
  const accuracy = Math.max(0, 1 - per)
  const certainty = acousticCertainty(confidence, margin)

  let score = 10 * (PHONEME_ACCURACY_WEIGHT * accuracy + ACOUSTIC_WEIGHT * certainty)
  if (nativeScore !== null) {
    score = combineScores(score, Math.max(0, Math.min(1, nativeScore)) * 10)
  }
  score = roundHalfEven(score, 1)

  const maxInsertionsTolerated = Math.max(2, Math.floor(totalTarget / 2))
  const insertionComponent = Math.max(0, 1 - alignment.insertions / maxInsertionsTolerated)
  const metrics = {
    accuracy: roundHalfEven((alignment.hits / totalTarget) * 100),
    completeness: roundHalfEven(((alignment.hits + alignment.substitutions) / totalTarget) * 100),
    fluency: roundHalfEven((0.55 * insertionComponent + 0.45 * certainty) * 100),
  }

  const phonemeResults = targetPhonemes.map((p) => ({ phoneme: p, status: 'missed', heard: null, tip: getTip(p) }))
  const errors = []
  let stressTotal = 0
  let stressMatched = 0

  for (const op of alignment.ops) {
    if (op.kind === 'insert') continue

    const target = targetPhonemes[op.refIndex]
    const entry = phonemeResults[op.refIndex]
    if (spans && op.hypIndex !== null) {
      [entry.start, entry.end] = spans[op.hypIndex]
    }

    if (unscored.has(op.refIndex)) {
      entry.status = 'unscored'
      entry.heard = op.hypIndex !== null ? predictedPhonemes[op.hypIndex] : null
      entry.tip = UNSCORABLE_TIP
      continue
    }

    if (op.kind === 'equal') {
      const heard = predictedPhonemes[op.hypIndex]
      entry.status = 'correct'
      entry.heard = heard
      entry.tip = ''
      if (stressOf(target) !== null) {
        stressTotal += 1
        const matched = stressOf(heard) === stressOf(target)
        stressMatched += matched ? 1 : 0
        entry.stress_match = matched
      }
    } else if (op.kind === 'substitute') {
      const heard = predictedPhonemes[op.hypIndex]
      entry.status = 'error'
      entry.heard = heard
      entry.tip = `Heard ${stripStress(heard)} instead.`
      errors.push({ type: 'substitute', index: op.refIndex, expected: [target], heard: [heard], tip: getTip(target) })
    } else {
      errors.push({ type: 'delete', index: op.refIndex, expected: [target], heard: [], tip: getTip(target) })
    }
  }

  let feedback
  let coachingTips = []
  if (errors.length === 0 && score >= 8.5) {
    feedback = 'Great! The phoneme match is strong.'
  } else if (errors.length === 0) {
    feedback = 'Every sound was there. Say it a little more clearly for a top score.'
    coachingTips = buildClarityCoaching(targetPhonemes)
  } else {
    feedback = 'Focus on the highlighted sounds.'
    for (const error of errors) {
      if (!coachingTips.includes(error.tip)) coachingTips.push(error.tip)
    }
    coachingTips = coachingTips.slice(0, 3)
  }

  return {
    score,
    wer: roundHalfEven(per, 2),
    per_with_stress: roundHalfEven(perWithStress, 2),
    metrics,
    stress: { matched: stressMatched, total: stressTotal },
    phoneme_results: phonemeResults,
    errors,
    feedback,
    coaching_tips: coachingTips,
  }
}
