// The app's own synthetic voices, for text with no human recording.
//
// A device's built-in voices are different everywhere and most devices have none for most
// accents, so each accent has one Piper voice (scripts/fetch_voices.py) that runs here with
// the same ONNX runtime as the phoneme model. Nothing is uploaded: the text is turned into
// sounds and then into audio on this device.
import { fetchCached, ort } from './model'

const CACHE_NAME = 'phonemepro-voices'
// Slightly slower than the voice's natural pace, so each sound can be heard.
const PACE = 1.1
const PEAK = 0.9

let manifest = null
let current = null // { accent, session } - one voice in memory at a time
let loading = null // { accent, promise }
let context = null

function loadManifest() {
  if (!manifest) {
    manifest = fetch('/voices/voices.json').then((r) => {
      if (!r.ok) throw new Error('The voice files are missing from this deployment.')
      return r.json()
    })
    manifest.catch(() => { manifest = null })
  }
  return manifest
}

/** Piper's input: the start mark and every known symbol, each followed by a pad, then the end mark. */
export function phonemeIds(phonemes, idMap) {
  const ids = [...idMap['^'], ...idMap._]
  for (const symbol of phonemes.normalize('NFD')) {
    if (!idMap[symbol]) continue
    ids.push(...idMap[symbol], ...idMap._)
  }
  ids.push(...idMap.$)
  return ids
}

async function loadVoice(accent, onProgress) {
  const voice = (await loadManifest())[accent]
  if (!voice) throw new Error(`No voice for ${accent}.`)
  let received = 0
  const parts = []
  for (const chunk of voice.chunks) {
    parts.push(...await fetchCached(`/voices/${chunk}?v=${voice.version}`, (bytes) => {
      received += bytes
      onProgress?.(Math.min(1, received / voice.bytes))
    }, CACHE_NAME))
  }
  const buffer = new Uint8Array(received)
  let offset = 0
  for (const part of parts) {
    buffer.set(part, offset)
    offset += part.length
  }
  // Only one voice is held in memory; the files stay in the browser cache.
  if (current) {
    await current.session.release().catch(() => {})
    current = null
  }
  const session = await ort.InferenceSession.create(buffer, { executionProviders: ['wasm'] })
  current = { accent, voice, session }
  return current
}

function voiceFor(accent, onProgress) {
  if (current?.accent === accent) return Promise.resolve(current)
  if (loading?.accent !== accent) {
    const promise = loadVoice(accent, onProgress).finally(() => {
      if (loading?.promise === promise) loading = null
    })
    loading = { accent, promise }
  }
  return loading.promise
}

/** Audio for `text` in this accent: { samples: Float32Array, sampleRate, voice }. */
export async function synthesize(text, accent, onProgress) {
  const [{ voice, session }, { phonemize }] = await Promise.all([voiceFor(accent, onProgress), import('phonemizer')])
  const phonemes = (await phonemize(text, voice.espeak)).join(' ')
  const ids = phonemeIds(phonemes, voice.phonemeIds)
  const [noise, length, noiseWidth] = voice.scales
  const feeds = {
    input: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]),
    input_lengths: new ort.Tensor('int64', BigInt64Array.of(BigInt(ids.length)), [1]),
    scales: new ort.Tensor('float32', Float32Array.of(noise, length * PACE, noiseWidth), [3]),
  }
  if (voice.speakerId !== null) feeds.sid = new ort.Tensor('int64', BigInt64Array.of(BigInt(voice.speakerId)), [1])
  const output = await session.run(feeds)
  return { samples: output[session.outputNames[0]].data, sampleRate: voice.sampleRate, voice: voice.name }
}

/** Say `text` aloud. Resolves with the voice's name when it has finished playing. */
export async function speakNeural(text, accent, onProgress) {
  // Opened before any waiting, while the tap that asked for it still counts as a user gesture.
  context ??= new (window.AudioContext || window.webkitAudioContext)()
  const resumed = context.resume()
  const { samples, sampleRate, voice } = await synthesize(text, accent, onProgress)
  await resumed
  // The voices differ in level; bring each to the same peak.
  let peak = 0
  for (const value of samples) peak = Math.max(peak, Math.abs(value))
  const level = peak > 0 ? PEAK / peak : 1
  const buffer = context.createBuffer(1, samples.length, sampleRate)
  buffer.copyToChannel(samples.map((value) => value * level), 0)
  const source = context.createBufferSource()
  source.buffer = buffer
  source.connect(context.destination)
  await new Promise((resolve) => {
    source.onended = resolve
    source.start()
  })
  return voice
}
