// Choosing the browser voice for text with no human recording.
//
// Browsers list voices in no useful order. On a Mac the first American English voice
// is "Albert", a novelty voice, and the list is empty until the browser has loaded it,
// so taking the first match (or none) gives a voice that is not the accent being practised.

export const VOICE_LANGUAGE = { ga: 'en-US', rp: 'en-GB', au: 'en-AU' }
export const ACCENT_NAME = { ga: 'American', rp: 'British', au: 'Australian' }

// Natural-sounding voices, best first. Matched against the start of the voice name.
const PREFERRED = {
  ga: ['Samantha', 'Ava', 'Allison', 'Alex', 'Susan', 'Tom', 'Microsoft Aria', 'Microsoft Jenny', 'Microsoft Guy', 'Microsoft Zira', 'Microsoft David', 'Google US English'],
  rp: ['Daniel', 'Kate', 'Serena', 'Oliver', 'Stephanie', 'Microsoft Sonia', 'Microsoft Libby', 'Microsoft Ryan', 'Microsoft Hazel', 'Microsoft Susan', 'Microsoft George', 'Google UK English Female', 'Google UK English Male'],
  au: ['Karen', 'Lee', 'Matilda', 'Catherine', 'Microsoft Natasha', 'Microsoft William', 'Microsoft Catherine', 'Microsoft James'],
}

// macOS novelty voices and the robotic Eloquence set. Never a model to copy.
const UNSUITABLE = new Set([
  'Albert', 'Bad News', 'Bahh', 'Bells', 'Boing', 'Bubbles', 'Cellos', 'Deranged', 'Fred', 'Good News', 'Hysterical',
  'Jester', 'Junior', 'Kathy', 'Organ', 'Pipe Organ', 'Princess', 'Ralph', 'Superstar', 'Trinoids', 'Whisper', 'Wobble', 'Zarvox',
  'Eddy', 'Flo', 'Grandma', 'Grandpa', 'Reed', 'Rocko', 'Sandy', 'Shelley',
])

// "Eddy (English (United Kingdom))" and "Daniel (Enhanced)" are listed under their plain names.
const baseName = (voice) => voice.name.replace(/\s*\(.*$/, '').trim()

/** The best installed voice for this accent, or null when the device has none worth using. */
export function pickVoice(voices, accent) {
  const language = VOICE_LANGUAGE[accent]
  const preferred = PREFERRED[accent] ?? []
  const rank = (voice) => {
    const index = preferred.findIndex((name) => voice.name.startsWith(name))
    return index === -1 ? preferred.length : index
  }
  const candidates = voices
    .filter((voice) => voice.lang.replace('_', '-') === language && !UNSUITABLE.has(baseName(voice)))
    // A voice on the device before one that sends the text to a server, then by quality.
    .map((voice, order) => ({ voice, order, remote: voice.localService === false ? 1 : 0, rank: rank(voice) }))
    .sort((a, b) => a.remote - b.remote || a.rank - b.rank || a.order - b.order)
  return candidates[0]?.voice ?? null
}

/** The voice list, waiting for the browser to load it. Chrome returns nothing on the first call. */
export function loadVoices(synth = window.speechSynthesis, timeoutMs = 1500) {
  return new Promise((resolve) => {
    const voices = synth.getVoices()
    if (voices.length) {
      resolve(voices)
      return
    }
    const done = () => {
      synth.removeEventListener?.('voiceschanged', done)
      clearTimeout(timer)
      resolve(synth.getVoices())
    }
    const timer = setTimeout(done, timeoutMs)
    synth.addEventListener?.('voiceschanged', done)
  })
}
