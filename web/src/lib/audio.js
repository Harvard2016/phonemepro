// Browser recordings arrive as WebM/MP4; the model wants 16 kHz mono PCM.

const TARGET_RATE = 16000

export const createAudioContext = () => new (window.AudioContext || window.webkitAudioContext)()

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  const writeText = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i))
  }

  writeText(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  writeText(8, 'WAVE')
  writeText(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeText(36, 'data')
  view.setUint32(40, samples.length * 2, true)

  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
  }
  return new Blob([buffer], { type: 'audio/wav' })
}

export async function toWav(blob) {
  const context = createAudioContext()
  try {
    const decoded = await context.decodeAudioData(await blob.arrayBuffer())
    const frames = Math.max(1, Math.ceil(decoded.duration * TARGET_RATE))
    const offline = new OfflineAudioContext(1, frames, TARGET_RATE)
    const source = offline.createBufferSource()
    source.buffer = decoded
    source.connect(offline.destination)
    source.start()
    const rendered = await offline.startRendering()
    return encodeWav(rendered.getChannelData(0), TARGET_RATE)
  } finally {
    context.close()
  }
}

export function speak(text, rate = 0.76) {
  return new Promise((resolve) => {
    if (!window.speechSynthesis) {
      resolve()
      return
    }
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = 'en-US'
    utterance.rate = rate
    utterance.onend = resolve
    utterance.onerror = resolve
    window.speechSynthesis.speak(utterance)
  })
}
