// One shared microphone session.
//
// Opening a microphone takes 100 to 300 ms, which used to clip the first sound of
// a word. The session therefore stays open between takes and always holds the
// last moment of audio, so a take can begin slightly before the button was pressed.
// It is released after a minute without use and whenever the page is hidden.
import workletUrl from './capture-worklet.js?worker&url'

const TARGET_RATE = 16000
const PRE_ROLL_SECONDS = 0.35
const TAIL_SECONDS = 0.2
const IDLE_RELEASE_MS = 60000

let session = null
let idleTimer = null

async function open() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  })
  const context = new (window.AudioContext || window.webkitAudioContext)()
  await context.audioWorklet.addModule(workletUrl)
  const source = context.createMediaStreamSource(stream)
  const analyser = context.createAnalyser()
  analyser.fftSize = 2048
  const capture = new AudioWorkletNode(context, 'capture', { numberOfOutputs: 0 })
  source.connect(analyser)
  source.connect(capture)

  const opened = { stream, context, analyser, blocks: [], held: 0, recording: false }
  const preRollSamples = PRE_ROLL_SECONDS * context.sampleRate
  capture.port.onmessage = ({ data }) => {
    opened.blocks.push(data)
    opened.held += data.length
    // While idle, keep only the most recent moment.
    while (!opened.recording && opened.held - opened.blocks[0].length >= preRollSamples) {
      opened.held -= opened.blocks.shift().length
    }
  }
  return opened
}

export function releaseMic() {
  clearTimeout(idleTimer)
  if (!session) return
  session.stream.getTracks().forEach((track) => track.stop())
  session.context.close()
  session = null
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden && !session?.recording) releaseMic()
})

// Begin a take. Resolves with the live analyser once audio is being kept.
// `preRoll: false` discards what was heard before now (used right after the app itself played audio).
export async function startTake({ preRoll = true } = {}) {
  clearTimeout(idleTimer)
  const fresh = !session
  if (fresh) session = await open()
  if (fresh || !preRoll) {
    session.blocks = []
    session.held = 0
  }
  session.recording = true
  return session.analyser
}

// End the take and return it as 16 kHz mono samples.
export async function finishTake() {
  const current = session
  if (!current?.recording) return null
  // People release the key as they finish speaking; keep listening briefly for the last sound.
  await new Promise((resolve) => { setTimeout(resolve, TAIL_SECONDS * 1000) })

  const captured = new Float32Array(current.held)
  let offset = 0
  for (const block of current.blocks) {
    captured.set(block, offset)
    offset += block.length
  }
  current.recording = false
  current.blocks = []
  current.held = 0
  idleTimer = setTimeout(releaseMic, IDLE_RELEASE_MS)

  if (captured.length === 0) return captured
  const rate = current.context.sampleRate
  if (rate === TARGET_RATE) return captured
  const offline = new OfflineAudioContext(1, Math.ceil((captured.length / rate) * TARGET_RATE), TARGET_RATE)
  const buffer = offline.createBuffer(1, captured.length, rate)
  buffer.copyToChannel(captured, 0)
  const node = offline.createBufferSource()
  node.buffer = buffer
  node.connect(offline.destination)
  node.start()
  return (await offline.startRendering()).getChannelData(0)
}

export function cancelTake() {
  if (!session) return
  session.recording = false
  idleTimer = setTimeout(releaseMic, IDLE_RELEASE_MS)
}
