import { useCallback, useEffect, useRef, useState } from 'react'
import { createAudioContext, toTake } from '../lib/audio'

const MIN_BLOB_BYTES = 1200

export const RECORDER_ERRORS = {
  permission: 'Microphone access is needed to listen. Allow it in your browser and try again.',
  short: "I couldn't catch that. Hold record a little longer and speak a bit slower.",
  decode: "I couldn't read that recording. Try once more.",
}

// Records from the microphone and hands back 16 kHz samples plus a WAV blob for playback.
// `analyserRef` exposes a live AnalyserNode while recording, for the waveform.
export function useRecorder({ onComplete, onError }) {
  const [isStarting, setIsStarting] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const analyserRef = useRef(null)
  const recorderRef = useRef(null)
  const wantedRef = useRef(false)
  const callbacksRef = useRef({ onComplete, onError })

  useEffect(() => {
    callbacksRef.current = { onComplete, onError }
  })

  const start = useCallback(async () => {
    if (wantedRef.current) return
    wantedRef.current = true
    setIsStarting(true)

    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      wantedRef.current = false
      setIsStarting(false)
      callbacksRef.current.onError(RECORDER_ERRORS.permission)
      return
    }

    // The key or button may have been released while the permission prompt was open.
    if (!wantedRef.current) {
      stream.getTracks().forEach((track) => track.stop())
      return
    }

    const context = createAudioContext()
    const analyser = context.createAnalyser()
    analyser.fftSize = 2048
    context.createMediaStreamSource(stream).connect(analyser)
    analyserRef.current = analyser

    const recorder = new MediaRecorder(stream)
    const chunks = []
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data)
    }
    recorder.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop())
      analyserRef.current = null
      context.close()

      const blob = new Blob(chunks, { type: recorder.mimeType })
      if (blob.size < MIN_BLOB_BYTES) {
        callbacksRef.current.onError(RECORDER_ERRORS.short)
        return
      }
      try {
        callbacksRef.current.onComplete(await toTake(blob))
      } catch {
        callbacksRef.current.onError(RECORDER_ERRORS.decode)
      }
    }

    recorderRef.current = recorder
    recorder.start()
    setIsStarting(false)
    setIsRecording(true)
  }, [])

  const stop = useCallback(() => {
    wantedRef.current = false
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
    setIsStarting(false)
    setIsRecording(false)
  }, [])

  useEffect(() => stop, [stop])

  return { isStarting, isRecording, start, stop, analyserRef }
}
