// Turn raw CTC logits into phonemes, timings and confidence signals.
// Mirrors PhonemeRecognizer in pronunciation/model.py.

// wav2vec 2.0 emits one frame per 320 samples at 16 kHz.
export const FRAME_SECONDS = 0.02
// CTC emits a short spike per phoneme, so the last one is padded to an audible length.
const FINAL_PHONEME_FRAMES = 6

const round3 = (value) => Math.round(value * 1000) / 1000

// `logits` is a flat Float32Array of frames x vocabSize. `meta` comes from model.json.
export function decodeCtc(logits, frames, meta) {
  const size = meta.vocab.length
  const special = new Set(meta.specialIds)
  const phonemes = []
  const startFrames = []
  let confidenceSum = 0
  let marginSum = 0
  let counted = 0
  let previous = -1

  for (let frame = 0; frame < frames; frame += 1) {
    const offset = frame * size
    let best = 0
    for (let token = 1; token < size; token += 1) {
      if (logits[offset + token] > logits[offset + best]) best = token
    }
    if (best === previous) continue
    previous = best
    if (best === meta.blankId) continue

    // Softmax over this frame for the probability of the winner and its lead over the runner-up.
    const peak = logits[offset + best]
    let total = 0
    let second = -Infinity
    for (let token = 0; token < size; token += 1) {
      const value = logits[offset + token]
      total += Math.exp(value - peak)
      if (token !== best && value > second) second = value
    }
    const top = 1 / total
    confidenceSum += top
    marginSum += top - Math.exp(second - peak) / total
    counted += 1

    if (!special.has(best)) {
      phonemes.push(meta.vocab[best])
      startFrames.push(frame)
    }
  }

  const spans = startFrames.map((start, index) => {
    const end = index + 1 < startFrames.length ? startFrames[index + 1] : Math.min(start + FINAL_PHONEME_FRAMES, frames)
    return [round3(start * FRAME_SECONDS), round3(end * FRAME_SECONDS)]
  })

  return {
    phonemes,
    spans,
    confidence: counted ? confidenceSum / counted : 0,
    margin: counted ? marginSum / counted : 0,
  }
}

// The model expects zero-mean, unit-variance audio (Wav2Vec2FeatureExtractor, do_normalize).
export function normalizeAudio(samples) {
  let mean = 0
  for (let i = 0; i < samples.length; i += 1) mean += samples[i]
  mean /= samples.length
  let variance = 0
  for (let i = 0; i < samples.length; i += 1) variance += (samples[i] - mean) ** 2
  const deviation = Math.sqrt(variance / samples.length + 1e-7)
  const normalized = new Float32Array(samples.length)
  for (let i = 0; i < samples.length; i += 1) normalized[i] = (samples[i] - mean) / deviation
  return normalized
}

export function softmax(values) {
  const peak = Math.max(...values)
  const exps = Array.from(values, (value) => Math.exp(value - peak))
  const total = exps.reduce((sum, value) => sum + value, 0)
  return exps.map((value) => value / total)
}
