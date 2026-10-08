// Loads the phoneme model and runs it in the browser with ONNX Runtime Web.
// Nothing is uploaded: audio is scored on the device that recorded it.
import * as ort from 'onnxruntime-web/wasm'
import { decodeCtc, normalizeAudio, softmax } from './decode'

const CACHE_NAME = 'phonemepro-model'

// The runtime's WebAssembly binary and its loader are copied into /ort by scripts/copy-ort.mjs.
// In a production build the loader must be its own file. Left bundled, each worker thread loads
// the whole app bundle as its script, which has no `document` to run in: the model then
// downloads and never starts. The dev server keeps modules apart, so the bundled loader is fine
// there, and it will not serve a file from /public as a module anyway.
ort.env.wasm.wasmPaths = import.meta.env.DEV
  ? { wasm: '/ort/ort-wasm-simd-threaded.wasm' }
  : { wasm: '/ort/ort-wasm-simd-threaded.wasm', mjs: '/ort/ort-wasm-simd-threaded.mjs' }
// Threads need cross-origin isolation headers; fall back to one thread without them.
ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1

let loading = null

async function fetchCached(url, onBytes) {
  const cache = 'caches' in globalThis ? await caches.open(CACHE_NAME).catch(() => null) : null
  let response = cache ? await cache.match(url) : null
  if (!response) {
    response = await fetch(url)
    if (!response.ok) throw new Error(`Could not download ${url} (${response.status}).`)
    if (cache) await cache.put(url, response.clone()).catch(() => {})
  }

  // Stream so the caller can show progress.
  const reader = response.body.getReader()
  const parts = []
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    parts.push(value)
    onBytes(value.length)
  }
  return parts
}

async function load(onProgress) {
  const meta = await fetch('/model/model.json').then((r) => {
    if (!r.ok) throw new Error('The model files are missing from this deployment.')
    return r.json()
  })

  // The model is split into chunks to stay under file-size limits; stitch them back together.
  let received = 0
  const parts = []
  for (const chunk of meta.chunks) {
    parts.push(...await fetchCached(`/model/${chunk}?v=${meta.version}`, (bytes) => {
      received += bytes
      onProgress?.(Math.min(1, received / meta.bytes))
    }))
  }
  const buffer = new Uint8Array(received)
  let offset = 0
  for (const part of parts) {
    buffer.set(part, offset)
    offset += part.length
  }

  // Drop chunks cached for older model versions.
  if ('caches' in globalThis) {
    const cache = await caches.open(CACHE_NAME).catch(() => null)
    for (const request of (await cache?.keys()) ?? []) {
      if (!request.url.endsWith(`?v=${meta.version}`)) cache.delete(request)
    }
  }

  const session = await ort.InferenceSession.create(buffer, { executionProviders: ['wasm'] })
  return { meta, session }
}

// Starts (or joins) the one-time model download. `onProgress` gets 0..1.
export function loadModel(onProgress) {
  if (!loading) {
    loading = load(onProgress)
    loading.catch(() => { loading = null })
  }
  return loading
}

// `samples`: Float32Array of 16 kHz mono audio.
export async function recognize(samples) {
  const { meta, session } = await loadModel()
  const input = new ort.Tensor('float32', normalizeAudio(samples), [1, samples.length])
  const output = await session.run({ input_values: input })

  const frames = output.logits.dims[1]
  const decoded = decodeCtc(output.logits.data, frames, meta)
  const accentShares = softmax(output.accent_logits.data)

  return {
    ...decoded,
    nativeScore: output.score.data[0],
    // The fixed 256-number accent summary of this take (scripts/learn_regions.py), and
    // the model that produced it: summaries from different models cannot be mixed.
    features: output.accent_features ? Array.from(output.accent_features.data) : null,
    featureCount: meta.features ?? 0,
    modelVersion: meta.version,
    accents: meta.accents
      .map((accent, index) => ({ ...accent, share: accentShares[index] }))
      .sort((a, b) => b.share - a.share),
  }
}
