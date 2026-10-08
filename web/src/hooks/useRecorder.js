import { useCallback, useEffect, useRef, useState } from 'react'
import { cancelTake, finishTake, startTake } from '../audio/mic'
import { cleanTake, SAMPLE_RATE } from '../engine/cleanup'
import { encodeWav } from '../lib/audio'

// Below this the model mishears too much for the feedback to mean anything: with noise
// 5 dB below the speech it gets four sounds in ten wrong (Docs/model/capture_results.json).
export const MIN_PRACTICE_SNR_DB = 10

export const RECORDER_ERRORS = {
  permission: 'Microphone access is needed to listen. Allow it in your browser and try again.',
  short: "I couldn't catch that. Hold record a little longer and speak a bit slower.",
  silent: "I didn't hear any speech in that take. Move closer to the microphone and try again.",
  noisy: 'There was too much background noise to hear you clearly. Find a quieter spot, or move closer, and try again.',
  failed: 'The microphone stopped unexpectedly. Try once more.',
}

// Records from the shared microphone session, cleans the take, and hands back
// 16 kHz samples, a WAV blob of the same samples for playback, and how clean the take was.
// `analyserRef` exposes a live AnalyserNode while recording, for the waveform.
export function useRecorder({ onComplete, onError }) {
  const [isStarting, setIsStarting] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const analyserRef = useRef(null)
  const wantedRef = useRef(false)
  const liveRef = useRef(false)
  const callbacksRef = useRef({ onComplete, onError })

  useEffect(() => {
    callbacksRef.current = { onComplete, onError }
  })

  const finish = useCallback(async () => {
    liveRef.current = false
    analyserRef.current = null
    setIsRecording(false)
    let captured
    try {
      captured = await finishTake()
    } catch {
      callbacksRef.current.onError(RECORDER_ERRORS.failed)
      return
    }
    if (!captured) return
    if (captured.length === 0) {
      callbacksRef.current.onError(RECORDER_ERRORS.short)
      return
    }

    const { samples, ...quality } = cleanTake(captured)
    if (!quality.foundSpeech) {
      callbacksRef.current.onError(RECORDER_ERRORS.silent)
    } else if (quality.snrDb < MIN_PRACTICE_SNR_DB) {
      callbacksRef.current.onError(RECORDER_ERRORS.noisy)
    } else {
      callbacksRef.current.onComplete({
        samples, seconds: samples.length / SAMPLE_RATE, wav: encodeWav(samples, SAMPLE_RATE), quality,
      })
    }
  }, [])

  // `preRoll: false` drops what the microphone heard before now, for when the app itself was just playing audio.
  const start = useCallback(async ({ preRoll = true } = {}) => {
    if (wantedRef.current) return
    wantedRef.current = true
    setIsStarting(true)

    let analyser
    try {
      analyser = await startTake({ preRoll })
    } catch {
      wantedRef.current = false
      setIsStarting(false)
      callbacksRef.current.onError(RECORDER_ERRORS.permission)
      return
    }

    // The key or button may have been released while the permission prompt was open.
    if (!wantedRef.current) {
      cancelTake()
      setIsStarting(false)
      return
    }
    liveRef.current = true
    analyserRef.current = analyser
    setIsStarting(false)
    setIsRecording(true)
  }, [])

  const stop = useCallback(() => {
    wantedRef.current = false
    setIsStarting(false)
    if (liveRef.current) finish()
  }, [finish])

  // Leaving the page abandons a take in progress.
  useEffect(() => () => {
    wantedRef.current = false
    if (liveRef.current) {
      liveRef.current = false
      cancelTake()
    }
  }, [])

  return { isStarting, isRecording, start, stop, analyserRef }
}
