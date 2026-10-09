// Playback helpers: a WAV of the learner's take, reference recordings, and a synthetic fallback.
import { loadVoices, pickVoice } from './voices'

export const createAudioContext = () => new (window.AudioContext || window.webkitAudioContext)()

// 16-bit mono WAV, so a take can be played back and sliced by time.
export function encodeWav(samples, sampleRate) {
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

// Play a human reference recording.
export function playClip(url) {
  return new Promise((resolve, reject) => {
    const audio = new Audio(url)
    audio.onended = resolve
    audio.onerror = () => reject(new Error('The recording could not be played.'))
    audio.play().catch(reject)
  })
}

// Synthetic fallback for text with no human recording. Resolves with the name of the voice
// used, or null when this device has no voice for the accent: saying the text in a
// different accent would teach the wrong thing, so nothing is spoken.
export async function speak(text, accent = 'ga', rate = 0.8) {
  const synth = window.speechSynthesis
  if (!synth) return null
  const voice = pickVoice(await loadVoices(synth), accent)
  if (!voice) return null
  return new Promise((resolve) => {
    synth.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.voice = voice
    utterance.lang = voice.lang
    utterance.rate = rate
    utterance.onend = () => resolve(voice.name)
    utterance.onerror = () => resolve(voice.name)
    synth.speak(utterance)
  })
}
