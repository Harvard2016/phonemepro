// Clean up a recording before it reaches the model: find where the speech is,
// trim to it with half a second of quiet each side, reduce steady background
// noise, and measure how clean the take was.
//
// Mirrors pronunciation/audio_cleanup.py, where each step is switchable and the
// reasons for these choices are recorded. Both are checked against
// tests/fixtures/cleanup_cases.json.

export const SAMPLE_RATE = 16000
const HIGH_PASS_HZ = 70
const FRAME = 320 // 20 ms
const HOP = 160 // 10 ms
const FLOOR_DB = -80
// A frame counts as speech when it is this far above the noise floor and within
// this range of the loudest part of the take.
const ABOVE_NOISE_DB = 6
const BELOW_PEAK_DB = 32
// Below this spread between loud and quiet frames there is no edge to find: the take
// is either steady noise or, if it is loud, speech from the first sample to the last.
const MIN_DYNAMIC_RANGE_DB = 6
const LOUD_DB = -35
const MIN_RUN_FRAMES = 4 // ignore blips shorter than 40 ms
const MAX_GAP_FRAMES = 35 // a silence longer than 350 ms separates groups of speech
const MIN_GROUP_FRAMES = 12 // a group shorter than 120 ms, standing apart, is a noise burst
const PAD_SECONDS = 0.5 // quiet kept on each side of the speech, as in the clips the model was trained on
const CLIP_LEVEL = 0.985

// Spectral gate
const FFT_SIZE = 512
const FFT_HOP = 128
const MIN_NOISE_FRAMES = 6
const OVER_SUBTRACTION = 1.5
const GAIN_FLOOR = 0.12

// First-order high-pass that removes DC offset and rumble below speech.
// Used only to locate the speech; the audio handed to the model is not filtered.
export function highPass(samples) {
  const rc = 1 / (2 * Math.PI * HIGH_PASS_HZ)
  const alpha = rc / (rc + 1 / SAMPLE_RATE)
  const out = new Float32Array(samples.length)
  let previousIn = 0
  let previousOut = 0
  for (let i = 0; i < samples.length; i += 1) {
    previousOut = alpha * (previousOut + samples[i] - previousIn)
    previousIn = samples[i]
    out[i] = previousOut
  }
  return out
}

// Energy of each 20 ms frame in dB relative to full scale.
export function frameLevels(samples) {
  const count = Math.max(0, 1 + Math.floor((samples.length - FRAME) / HOP))
  const levels = new Float64Array(count).fill(FLOOR_DB)
  for (let frame = 0; frame < count; frame += 1) {
    let power = 0
    for (let i = frame * HOP; i < frame * HOP + FRAME; i += 1) power += samples[i] * samples[i]
    power /= FRAME
    if (power > 0) levels[frame] = Math.max(FLOOR_DB, 10 * Math.log10(power))
  }
  return levels
}

// Nearest-rank percentile.
function percentile(values, fraction) {
  const ordered = Float64Array.from(values).sort()
  return ordered[Math.floor(fraction * (ordered.length - 1))]
}

// Boolean mask of speech frames and the noise floor, or null if there is no clear speech.
export function speechFrames(levels) {
  if (levels.length < MIN_RUN_FRAMES) return null
  const floor = percentile(levels, 0.1)
  const peak = percentile(levels, 0.95)
  if (peak - floor < MIN_DYNAMIC_RANGE_DB) {
    return floor > LOUD_DB ? { mask: new Uint8Array(levels.length).fill(1), floor } : null
  }
  const threshold = Math.max(floor + ABOVE_NOISE_DB, peak - BELOW_PEAK_DB)

  // Drop short blips (a key press, a click).
  const mask = new Uint8Array(levels.length)
  let runStart = -1
  for (let i = 0; i <= levels.length; i += 1) {
    const on = i < levels.length && levels[i] > threshold
    if (on && runStart === -1) {
      runStart = i
    } else if (!on && runStart !== -1) {
      if (i - runStart >= MIN_RUN_FRAMES) mask.fill(1, runStart, i)
      runStart = -1
    }
  }
  const indices = []
  mask.forEach((on, i) => { if (on) indices.push(i) })
  if (indices.length === 0) return null

  // Split into groups at long silences. A short burst standing apart from the rest
  // (a key press, a cough) is dropped; real speech on both sides of a pause is kept.
  const groups = []
  let groupStart = indices[0]
  let previous = indices[0]
  for (const index of indices.slice(1)) {
    if (index - previous > MAX_GAP_FRAMES) {
      groups.push([groupStart, previous])
      groupStart = index
    }
    previous = index
  }
  groups.push([groupStart, previous])
  const sizes = groups.map(([start, end]) => mask.slice(start, end + 1).reduce((sum, on) => sum + on, 0))
  const largest = Math.max(...sizes)
  const solid = groups.filter((_, i) => sizes[i] >= MIN_GROUP_FRAMES || sizes[i] === largest)

  const kept = new Uint8Array(levels.length)
  kept.fill(1, solid[0][0], solid[solid.length - 1][1] + 1)
  return { mask: kept, floor }
}

// In-place radix-2 FFT of FFT_SIZE complex points; `inverse` leaves the result unscaled.
const BIT_REVERSED = (() => {
  const bits = Math.log2(FFT_SIZE)
  return Uint16Array.from({ length: FFT_SIZE }, (_, i) => {
    let reversed = 0
    for (let bit = 0; bit < bits; bit += 1) reversed |= ((i >> bit) & 1) << (bits - 1 - bit)
    return reversed
  })
})()
const TWIDDLE_COS = Float64Array.from({ length: FFT_SIZE / 2 }, (_, i) => Math.cos((2 * Math.PI * i) / FFT_SIZE))
const TWIDDLE_SIN = Float64Array.from({ length: FFT_SIZE / 2 }, (_, i) => Math.sin((2 * Math.PI * i) / FFT_SIZE))
// Periodic Hann window, as numpy.hanning(FFT_SIZE + 1)[:-1].
const WINDOW = Float64Array.from({ length: FFT_SIZE }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / FFT_SIZE))

function fft(real, imaginary, inverse) {
  for (let i = 0; i < FFT_SIZE; i += 1) {
    const j = BIT_REVERSED[i]
    if (j > i) {
      ;[real[i], real[j]] = [real[j], real[i]]
      ;[imaginary[i], imaginary[j]] = [imaginary[j], imaginary[i]]
    }
  }
  for (let size = 2; size <= FFT_SIZE; size *= 2) {
    const half = size / 2
    const step = FFT_SIZE / size
    for (let start = 0; start < FFT_SIZE; start += size) {
      for (let k = 0; k < half; k += 1) {
        const cos = TWIDDLE_COS[k * step]
        const sin = inverse ? TWIDDLE_SIN[k * step] : -TWIDDLE_SIN[k * step]
        const a = start + k
        const b = a + half
        const re = real[b] * cos - imaginary[b] * sin
        const im = real[b] * sin + imaginary[b] * cos
        real[b] = real[a] - re
        imaginary[b] = imaginary[a] - im
        real[a] += re
        imaginary[a] += im
      }
    }
  }
}

// Subtract the average spectrum of `noise` (audio known to hold no speech) from `samples`.
export function spectralGate(samples, noise) {
  if (noise.length < FFT_SIZE + (MIN_NOISE_FRAMES - 1) * FFT_HOP || samples.length < FFT_SIZE) return samples
  const bins = FFT_SIZE / 2 + 1
  const real = new Float64Array(FFT_SIZE)
  const imaginary = new Float64Array(FFT_SIZE)
  const transform = (audio, start) => {
    for (let i = 0; i < FFT_SIZE; i += 1) {
      real[i] = audio[start + i] * WINDOW[i]
      imaginary[i] = 0
    }
    fft(real, imaginary, false)
  }

  const profile = new Float64Array(bins)
  const noiseFrames = 1 + Math.floor((noise.length - FFT_SIZE) / FFT_HOP)
  for (let frame = 0; frame < noiseFrames; frame += 1) {
    transform(noise, frame * FFT_HOP)
    for (let bin = 0; bin < bins; bin += 1) profile[bin] += Math.hypot(real[bin], imaginary[bin]) / noiseFrames
  }

  // Pad with a window of silence each side so the edges are covered by full overlap.
  const padded = new Float64Array(samples.length + 2 * FFT_SIZE)
  padded.set(samples, FFT_SIZE)
  const out = new Float64Array(padded.length)
  const weight = new Float64Array(padded.length)
  const frames = 1 + Math.floor((padded.length - FFT_SIZE) / FFT_HOP)
  for (let frame = 0; frame < frames; frame += 1) {
    const start = frame * FFT_HOP
    transform(padded, start)
    for (let bin = 0; bin < bins; bin += 1) {
      const magnitude = Math.hypot(real[bin], imaginary[bin])
      const gain = Math.min(1, Math.max(GAIN_FLOOR, (magnitude - OVER_SUBTRACTION * profile[bin]) / Math.max(magnitude, 1e-12)))
      real[bin] *= gain
      imaginary[bin] *= gain
      // The upper half mirrors the lower, keeping the signal real.
      if (bin > 0 && bin < bins - 1) {
        real[FFT_SIZE - bin] = real[bin]
        imaginary[FFT_SIZE - bin] = -imaginary[bin]
      }
    }
    fft(real, imaginary, true)
    for (let i = 0; i < FFT_SIZE; i += 1) {
      out[start + i] += (real[i] / FFT_SIZE) * WINDOW[i]
      weight[start + i] += WINDOW[i] * WINDOW[i]
    }
  }

  const cleaned = new Float32Array(samples.length)
  for (let i = 0; i < samples.length; i += 1) cleaned[i] = out[FFT_SIZE + i] / Math.max(weight[FFT_SIZE + i], 1e-8)
  return cleaned
}

const meanPowerDb = (levels) =>
  10 * Math.log10(levels.reduce((sum, level) => sum + 10 ** (level / 10), 0) / levels.length)

const round = (value, digits) => Math.round(value * 10 ** digits) / 10 ** digits

// Find the speech, trim to it, reduce steady noise, and measure quality.
// Returns { samples, foundSpeech, start, end, speechSeconds, snrDb, clipped }.
export function cleanTake(samples) {
  let clippedCount = 0
  for (let i = 0; i < samples.length; i += 1) if (Math.abs(samples[i]) >= CLIP_LEVEL) clippedCount += 1
  const clipped = samples.length ? clippedCount / samples.length : 0

  // Speech is located on the filtered signal, where rumble cannot hide it.
  const levels = frameLevels(highPass(samples))
  const found = speechFrames(levels)
  if (!found) {
    return { samples, foundSpeech: false, start: 0, end: samples.length, speechSeconds: 0, snrDb: 0, clipped }
  }

  const { mask, floor } = found
  const first = mask.indexOf(1)
  const last = mask.lastIndexOf(1)
  const speechStart = first * HOP
  const speechEnd = Math.min(samples.length, last * HOP + FRAME)
  const pad = Math.trunc(PAD_SECONDS * SAMPLE_RATE)
  const start = Math.max(0, speechStart - pad)
  const end = Math.min(samples.length, speechEnd + pad)

  // Signal-to-noise ratio: speech frames against the typical frame outside them (the
  // median, so a click or cough does not count as background), or the quietest
  // frames when the take is speech from edge to edge.
  const inside = []
  const outside = []
  levels.forEach((level, i) => (mask[i] ? inside : outside).push(level))
  const noiseLevel = outside.length >= MIN_RUN_FRAMES ? percentile(outside, 0.5) : floor
  const snrDb = Math.max(0, Math.min(60, meanPowerDb(inside) - noiseLevel))

  // Everything outside the speech is the noise the gate learns from.
  const noise = new Float32Array(speechStart + samples.length - speechEnd)
  noise.set(samples.subarray(0, speechStart))
  noise.set(samples.subarray(speechEnd), speechStart)

  return {
    samples: spectralGate(samples.slice(start, end), noise),
    foundSpeech: true,
    start,
    end,
    speechSeconds: round((speechEnd - speechStart) / SAMPLE_RATE, 3),
    snrDb: round(snrDb, 1),
    clipped,
  }
}
