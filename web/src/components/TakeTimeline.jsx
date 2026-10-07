import { useEffect, useState } from 'react'
import { createAudioContext } from '../lib/audio'
import { toIpa } from '../lib/phonemes'

const BARS = 160

// Reduce the recording to one peak value per bar.
async function readPeaks(blob) {
  const context = createAudioContext()
  try {
    const audio = await context.decodeAudioData(await blob.arrayBuffer())
    const samples = audio.getChannelData(0)
    const size = Math.max(1, Math.floor(samples.length / BARS))
    const peaks = []
    for (let bar = 0; bar < BARS; bar += 1) {
      let peak = 0
      for (let i = bar * size; i < Math.min((bar + 1) * size, samples.length); i += 1) {
        peak = Math.max(peak, Math.abs(samples[i]))
      }
      peaks.push(peak)
    }
    const loudest = Math.max(...peaks, 0.01)
    return { peaks: peaks.map((peak) => peak / loudest), duration: audio.duration }
  } finally {
    context.close()
  }
}

// The learner's own recording with each heard sound pinned where it occurred.
// Tapping a sound plays just that slice.
function TakeTimeline({ blob, results, onPlay }) {
  const [shape, setShape] = useState(null)

  useEffect(() => {
    let cancelled = false
    readPeaks(blob)
      .then((next) => { if (!cancelled) setShape(next) })
      .catch(() => { if (!cancelled) setShape(null) })
    return () => { cancelled = true }
  }, [blob])

  const timed = results.filter((result) => result.start !== undefined)
  if (!shape || timed.length === 0) return null

  return (
    <div className="timeline">
      <p className="label">Where each sound landed · tap one to hear it</p>
      <div className="timeline__track">
        <svg className="timeline__wave" viewBox={`0 0 ${BARS} 40`} preserveAspectRatio="none" aria-hidden="true">
          {shape.peaks.map((peak, bar) => (
            <rect key={bar} x={bar + 0.2} width="0.6" y={20 - peak * 18} height={Math.max(peak * 36, 0.6)} />
          ))}
        </svg>
        {timed.map((result) => (
          <button
            key={`${result.phoneme}-${result.start}`}
            type="button"
            className={`timeline__mark timeline__mark--${result.status}`}
            style={{ left: `${Math.min(98, (result.start / shape.duration) * 100)}%` }}
            onClick={() => onPlay(result.start, result.end)}
            aria-label={`Play ${result.heard ?? result.phoneme}${result.status === 'error' ? ', substituted' : ''}`}
          >
            <span className="ipa">{toIpa(result.heard ?? result.phoneme)}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

export default TakeTimeline
