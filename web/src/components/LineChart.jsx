import { useMemo, useState } from 'react'

const WIDTH = 720
const HEIGHT = 300
const PAD = { top: 16, right: 16, bottom: 34, left: 44 }

const niceLogTicks = (min, max) => {
  const ticks = []
  for (let power = Math.floor(Math.log10(min)); power <= Math.ceil(Math.log10(max)); power += 1) {
    for (const multiple of [1, 2, 5]) {
      const value = multiple * 10 ** power
      if (value >= min && value <= max) ticks.push(value)
    }
  }
  return ticks
}

// Round tick positions for a linear axis: at most `count` steps of 1, 2 or 5 times a power of ten.
const niceLinearTicks = (min, max, count = 5) => {
  const rough = (max - min) / count || 1
  const magnitude = 10 ** Math.floor(Math.log10(rough))
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= rough)
  const ticks = []
  for (let tick = Math.ceil(min / step) * step; tick <= max + step * 1e-6; tick += step) {
    ticks.push(Number(tick.toPrecision(12)))
  }
  return ticks
}

// Small SVG line chart with a hover readout. `series` is [{ name, tone, points: [{x, y}] }].
// `yDomain` fixes the linear y range, e.g. [0, 10] for scores.
function LineChart({ series, logY = false, yDomain, xLabel, formatY = (v) => v.toFixed(2) }) {
  const [hoverX, setHoverX] = useState(null)

  const { scaleX, scaleY, yTicks, xTicks, xValues } = useMemo(() => {
    const all = series.flatMap((s) => s.points)
    const xMin = Math.min(...all.map((p) => p.x))
    const xMax = Math.max(...all.map((p) => p.x))
    const yMin = Math.min(...all.map((p) => p.y))
    const yMax = Math.max(...all.map((p) => p.y))
    const transform = logY ? Math.log10 : (v) => v
    const linearLo = yDomain ? yDomain[0] : Math.min(0, yMin)
    const linearHi = yDomain ? yDomain[1] : yMax * 1.08
    const lo = logY ? Math.log10(yMin * 0.85) : linearLo
    const hi = logY ? Math.log10(yMax * 1.08) : linearHi
    const innerW = WIDTH - PAD.left - PAD.right
    const innerH = HEIGHT - PAD.top - PAD.bottom

    return {
      scaleX: (x) => PAD.left + ((x - xMin) / (xMax - xMin || 1)) * innerW,
      scaleY: (y) => PAD.top + (1 - (transform(y) - lo) / (hi - lo || 1)) * innerH,
      yTicks: logY ? niceLogTicks(yMin * 0.85, yMax * 1.08) : niceLinearTicks(linearLo, linearHi),
      xTicks: niceLinearTicks(xMin, xMax).filter(Number.isInteger),
      xValues: [...new Set(all.map((p) => p.x))].sort((a, b) => a - b),
    }
  }, [series, logY, yDomain])

  const onMove = (event) => {
    const bounds = event.currentTarget.getBoundingClientRect()
    const svgX = ((event.clientX - bounds.left) / bounds.width) * WIDTH
    let nearest = xValues[0]
    for (const x of xValues) {
      if (Math.abs(scaleX(x) - svgX) < Math.abs(scaleX(nearest) - svgX)) nearest = x
    }
    setHoverX(nearest)
  }

  const readout = hoverX === null ? null : series
    .map((s) => ({ name: s.name, tone: s.tone, point: s.points.find((p) => p.x === hoverX) }))
    .filter((entry) => entry.point)

  return (
    <figure className="chart">
      <figcaption className="chart__legend">
        {series.map((s) => (
          <span key={s.name} className={`chart__key chart__key--${s.tone}`}>{s.name}</span>
        ))}
        <span className="chart__readout" aria-live="off">
          {readout && readout.length > 0
            ? `${xLabel} ${hoverX} · ${readout.map((r) => `${r.name} ${formatY(r.point.y)}`).join(' · ')}`
            : 'Hover the chart for values'}
        </span>
      </figcaption>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="chart__svg"
        role="img"
        aria-label={`${series.map((s) => s.name).join(' and ')} by ${xLabel}`}
        onMouseMove={onMove}
        onMouseLeave={() => setHoverX(null)}
      >
        {yTicks.map((tick) => (
          <g key={tick}>
            <line className="chart__grid" x1={PAD.left} x2={WIDTH - PAD.right} y1={scaleY(tick)} y2={scaleY(tick)} />
            <text className="chart__tick" x={PAD.left - 8} y={scaleY(tick)} textAnchor="end" dominantBaseline="middle">
              {Number(tick.toFixed(1))}
            </text>
          </g>
        ))}
        {xTicks.map((tick) => (
          <text key={tick} className="chart__tick" x={scaleX(tick)} y={HEIGHT - 10} textAnchor="middle">
            {tick}
          </text>
        ))}
        {series.map((s) => (
          <polyline
            key={s.name}
            className={`chart__line chart__line--${s.tone}`}
            points={s.points.map((p) => `${scaleX(p.x)},${scaleY(p.y)}`).join(' ')}
          />
        ))}
        {hoverX !== null && (
          <line className="chart__cursor" x1={scaleX(hoverX)} x2={scaleX(hoverX)} y1={PAD.top} y2={HEIGHT - PAD.bottom} />
        )}
        {readout?.map((r) => (
          <circle key={r.name} className={`chart__dot chart__dot--${r.tone}`} cx={scaleX(r.point.x)} cy={scaleY(r.point.y)} r="4" />
        ))}
      </svg>
    </figure>
  )
}

export default LineChart
