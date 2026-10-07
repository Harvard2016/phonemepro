import { useEffect, useRef } from 'react'

// The rule that runs across the stage. Idle it is a hairline; while recording it
// traces the microphone signal, and while the model works it carries a slow ripple.
function Waveform({ analyserRef, mode }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    let frame = null
    let samples = null

    const resize = () => {
      const ratio = window.devicePixelRatio || 1
      canvas.width = canvas.clientWidth * ratio
      canvas.height = canvas.clientHeight * ratio
      context.setTransform(ratio, 0, 0, ratio, 0, 0)
    }

    const draw = (time) => {
      const width = canvas.clientWidth
      const height = canvas.clientHeight
      const middle = height / 2
      context.clearRect(0, 0, width, height)
      context.strokeStyle = getComputedStyle(canvas).color
      context.lineWidth = mode === 'idle' ? 1 : 1.5
      context.lineJoin = 'round'
      context.beginPath()

      const analyser = analyserRef.current
      if (mode === 'recording' && analyser) {
        if (!samples || samples.length !== analyser.fftSize) samples = new Uint8Array(analyser.fftSize)
        analyser.getByteTimeDomainData(samples)
        const step = width / (samples.length - 1)
        for (let i = 0; i < samples.length; i += 1) {
          const y = middle + ((samples[i] - 128) / 128) * middle * 1.6
          if (i === 0) context.moveTo(0, y)
          else context.lineTo(i * step, y)
        }
      } else if (mode === 'thinking' && !reducedMotion) {
        for (let x = 0; x <= width; x += 4) {
          const envelope = Math.sin((x / width) * Math.PI)
          const y = middle + Math.sin(x * 0.035 - time * 0.006) * envelope * height * 0.16
          if (x === 0) context.moveTo(0, y)
          else context.lineTo(x, y)
        }
      } else {
        context.moveTo(0, middle)
        context.lineTo(width, middle)
      }
      context.stroke()

      if (mode !== 'idle' && !(mode === 'thinking' && reducedMotion)) {
        frame = requestAnimationFrame(draw)
      }
    }

    const observer = new ResizeObserver(() => {
      resize()
      if (frame === null) draw(0)
    })
    observer.observe(canvas)
    resize()
    frame = null
    draw(0)

    return () => {
      observer.disconnect()
      if (frame !== null) cancelAnimationFrame(frame)
    }
  }, [analyserRef, mode])

  return <canvas ref={canvasRef} className={`waveform waveform--${mode}`} aria-hidden="true" />
}

export default Waveform
