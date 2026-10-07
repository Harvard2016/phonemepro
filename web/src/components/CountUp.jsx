import { useEffect, useState } from 'react'

// Counts from zero to `value` once, easing out.
function CountUp({ value, decimals = 1, duration = 900 }) {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const [shown, setShown] = useState(reducedMotion ? value : 0)

  useEffect(() => {
    if (reducedMotion) return undefined

    let frame
    const started = performance.now()
    const tick = (now) => {
      const progress = Math.min(1, (now - started) / duration)
      setShown(value * (1 - (1 - progress) ** 3))
      if (progress < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [value, duration, reducedMotion])

  return <>{(reducedMotion ? value : shown).toFixed(decimals)}</>
}

export default CountUp
