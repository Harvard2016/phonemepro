import { useCallback, useEffect, useMemo, useState } from 'react'
import { motion } from 'motion/react'
import LineChart from '../components/LineChart'
import PhonemeStrip from '../components/PhonemeStrip'
import { LoadingState, MessageState } from '../components/States'
import { clearRealHistory, getHistory, getInsights, historyExportUrl } from '../lib/api'
import { toIpa } from '../lib/phonemes'

const FILTERS = [
  ['all', 'All'],
  ['high', '8 and up'],
  ['low', 'Under 5'],
  ['real', 'Mine'],
  ['sample', 'Samples'],
]

const SCORE_DOMAIN = [0, 10]

const formatTime = (iso) =>
  new Date(iso).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })

// Rebuild per-phoneme marks for a stored attempt from its error list.
function attemptResults(item) {
  const pending = [...(item.errors ?? [])]
  return item.target_phonemes.map((phoneme, index) => {
    let at = pending.findIndex((error) => error.index === index)
    if (at === -1) at = pending.findIndex((error) => error.index === undefined && error.expected?.[0] === phoneme)
    if (at === -1) return { phoneme, status: 'correct' }
    const [error] = pending.splice(at, 1)
    return { phoneme, status: error.type === 'delete' ? 'missed' : 'error', heard: error.heard?.[0] }
  })
}

function HistoryPage() {
  const [history, setHistory] = useState(null)
  const [insights, setInsights] = useState(null)
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState('all')
  const [confirmingClear, setConfirmingClear] = useState(false)
  const [clearing, setClearing] = useState(false)

  const load = useCallback(() => {
    setError(null)
    Promise.all([getHistory(), getInsights()])
      .then(([attempts, phonemeInsights]) => {
        setHistory(attempts)
        setInsights(phonemeInsights)
      })
      .catch((failure) => setError(failure.message))
  }, [])

  useEffect(load, [load])

  const stats = useMemo(() => {
    if (!history?.length) return null
    const real = history.filter((item) => item.source !== 'sample')
    const basis = real.length ? real : history
    const oldestFirst = [...basis].reverse()

    const byWord = {}
    oldestFirst.forEach((item) => {
      byWord[item.word] = [...(byWord[item.word] ?? []), Number(item.score)]
    })
    let improved = null
    Object.entries(byWord).forEach(([word, scores]) => {
      const delta = scores[scores.length - 1] - scores[0]
      if (scores.length > 1 && delta > (improved?.delta ?? 0)) improved = { word, delta }
    })

    return {
      usesSamples: real.length === 0,
      attempts: basis.length,
      average: basis.reduce((sum, item) => sum + Number(item.score), 0) / basis.length,
      best: basis.reduce((best, item) => (item.score > best.score ? item : best)),
      improved,
      trend: oldestFirst.map((item, index) => ({ x: index + 1, y: Number(item.score) })),
    }
  }, [history])

  const clearHistory = async () => {
    setClearing(true)
    try {
      await clearRealHistory()
      setConfirmingClear(false)
      load()
    } catch (failure) {
      setError(failure.message)
    } finally {
      setClearing(false)
    }
  }

  if (error) {
    return (
      <MessageState
        title="History is unavailable"
        action={<button type="button" className="button" onClick={load}>Try again</button>}
      >
        {error}
      </MessageState>
    )
  }
  if (!history) return <LoadingState label="Opening the ledger" />
  if (history.length === 0) {
    return (
      <MessageState title="Nothing in the ledger yet">
        Say your first word on the Practice page and it will be logged here.
      </MessageState>
    )
  }

  const visible = history.filter((item) => {
    if (filter === 'high') return item.score >= 8
    if (filter === 'low') return item.score < 5
    if (filter === 'real') return item.source === 'real'
    if (filter === 'sample') return item.source === 'sample'
    return true
  })
  const hasReal = history.some((item) => item.source === 'real')

  return (
    <div className="page">
      <header className="page__head">
        <p className="label">Practice history</p>
        <h1 className="page__title">The ledger</h1>
        <p className="page__lede">
          {stats.usesSamples
            ? 'These are sample entries. Your own attempts replace them in the figures below as soon as you record.'
            : `${stats.attempts} attempt${stats.attempts === 1 ? '' : 's'} on record.`}
        </p>
      </header>

      <dl className="figures">
        <div className="figure">
          <dt className="label">Average score</dt>
          <dd className="figure__value">{stats.average.toFixed(1)}</dd>
        </div>
        <div className="figure">
          <dt className="label">Best word</dt>
          <dd className="figure__value figure__value--word">{stats.best.word}</dd>
          <dd className="figure__note">{stats.best.score} / 10</dd>
        </div>
        <div className="figure">
          <dt className="label">Most improved</dt>
          <dd className="figure__value figure__value--word">{stats.improved?.word ?? 'none yet'}</dd>
          <dd className="figure__note">
            {stats.improved ? `up ${stats.improved.delta.toFixed(1)} points` : 'repeat a word to see it move'}
          </dd>
        </div>
        <div className="figure">
          <dt className="label">Attempts</dt>
          <dd className="figure__value">{stats.attempts}</dd>
        </div>
      </dl>

      <div className="spread">
        <section className="spread__main" aria-label="Score trend">
          <h2 className="section-title">Score by attempt</h2>
          {stats.trend.length > 1 ? (
            <LineChart
              series={[{ name: 'Score', tone: 'ink', points: stats.trend }]}
              xLabel="Attempt"
              yDomain={SCORE_DOMAIN}
              formatY={(value) => value.toFixed(1)}
            />
          ) : (
            <p className="focus__empty">A trend line appears after your second attempt.</p>
          )}
        </section>

        <section className="spread__side" aria-label="Weak sounds">
          <h2 className="section-title">Weakest sounds</h2>
          {insights?.weak_phonemes?.length > 0 ? (
            <ul className="bars">
              {insights.weak_phonemes.map((item) => (
                <li key={item.phoneme} className="bars__row">
                  <span className="bars__name">
                    <span className="ipa">{toIpa(item.phoneme)}</span> <span className="bars__code">{item.phoneme}</span>
                  </span>
                  <span className="meter" aria-hidden="true">
                    <span className="meter__fill meter__fill--accent" style={{ transform: `scaleX(${item.error_rate})` }} />
                  </span>
                  <span className="bars__value">{Math.round(item.error_rate * 100)}%</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="focus__empty">No missed sounds in your own attempts so far.</p>
          )}
        </section>
      </div>

      <div className="toolbar">
        <div className="tabs" role="group" aria-label="Filter attempts">
          {FILTERS.map(([id, label]) => (
            <button
              key={id}
              type="button"
              className={`tabs__tab${filter === id ? ' is-active' : ''}`}
              aria-pressed={filter === id}
              onClick={() => setFilter(id)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="toolbar__actions">
          <a className="text-button" href={historyExportUrl}>Export CSV</a>
          {hasReal && (
            confirmingClear ? (
              <span className="confirm">
                Clear your attempts?
                <button type="button" className="text-button text-button--accent" onClick={clearHistory} disabled={clearing}>
                  {clearing ? 'Clearing' : 'Yes, clear'}
                </button>
                <button type="button" className="text-button" onClick={() => setConfirmingClear(false)} disabled={clearing}>
                  Keep
                </button>
              </span>
            ) : (
              <button type="button" className="text-button" onClick={() => setConfirmingClear(true)}>
                Clear my attempts
              </button>
            )
          )}
        </div>
      </div>

      {visible.length === 0 ? (
        <p className="focus__empty">No attempts match this filter.</p>
      ) : (
        <ol className="ledger">
          {visible.map((item, index) => (
            <motion.li
              key={item.id}
              className="ledger__row"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: Math.min(index, 10) * 0.03 }}
            >
              <span className="ledger__time">
                {formatTime(item.timestamp)}
                {item.source === 'sample' && <span className="tag">sample</span>}
              </span>
              <span className="ledger__word">{item.word}</span>
              <PhonemeStrip results={attemptResults(item)} animate={false} compact />
              <span className={`ledger__score${item.score < 5 ? ' ledger__score--low' : ''}`}>
                {Number(item.score).toFixed(1)}
              </span>
            </motion.li>
          ))}
        </ol>
      )}
    </div>
  )
}

export default HistoryPage
