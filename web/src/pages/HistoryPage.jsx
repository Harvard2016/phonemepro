import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { motion } from 'motion/react'
import LineChart from '../components/LineChart'
import PhonemeStrip from '../components/PhonemeStrip'
import { LoadingState, MessageState } from '../components/States'
import { clearHistory, exportHistory, getHistory, importHistory, toCsv } from '../engine/history'
import { buildInsights } from '../engine/insights'
import { toIpa } from '../lib/phonemes'
import { useApp } from '../state'

const SCORE_DOMAIN = [0, 10]
const SCORE_FILTERS = [
  ['all', 'All'],
  ['high', '8 and up'],
  ['low', 'Under 5'],
]

const formatTime = (iso) =>
  new Date(iso).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })

// Rebuild per-phoneme marks for a stored attempt from its error list.
function attemptResults(item) {
  const pending = [...(item.errors ?? [])]
  return item.target_phonemes.map((phoneme, index) => {
    const at = pending.findIndex((error) => error.index === index)
    if (at === -1) return { phoneme, status: 'correct' }
    const [error] = pending.splice(at, 1)
    return { phoneme, status: error.type === 'delete' ? 'missed' : 'error', heard: error.heard?.[0] }
  })
}

function download(filename, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

function HistoryPage() {
  const { lexicon, words, accent } = useApp()
  const [history, setHistory] = useState(null)
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)
  const [scoreFilter, setScoreFilter] = useState('all')
  const [accentFilter, setAccentFilter] = useState('all')
  const [confirmingClear, setConfirmingClear] = useState(false)
  const fileInput = useRef(null)

  const load = useCallback(() => {
    getHistory().then(setHistory).catch(() => setError('This browser would not open its local storage.'))
  }, [])

  useEffect(load, [load])

  const retry = () => {
    setError(null)
    load()
  }

  const scoped = useMemo(
    () => (history ?? []).filter((item) => accentFilter === 'all' || (item.accent ?? 'ga') === accentFilter),
    [history, accentFilter],
  )

  const stats = useMemo(() => {
    if (!scoped.length) return null
    const oldestFirst = [...scoped].reverse()
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
      attempts: scoped.length,
      average: scoped.reduce((sum, item) => sum + Number(item.score), 0) / scoped.length,
      best: scoped.reduce((best, item) => (item.score > best.score ? item : best)),
      improved,
      trend: oldestFirst.map((item, index) => ({ x: index + 1, y: Number(item.score) })),
    }
  }, [scoped])

  // Weak sounds are judged within one accent, since the targets differ between accents.
  const insightAccent = accentFilter === 'all' ? accent : accentFilter
  const insights = useMemo(
    () => (history && words
      ? buildInsights(history.filter((item) => (item.accent ?? 'ga') === insightAccent), words)
      : null),
    [history, words, insightAccent],
  )

  const onImport = async (event) => {
    const [file] = event.target.files
    event.target.value = ''
    if (!file) return
    try {
      const added = await importHistory(await file.text())
      setNotice(added ? `Added ${added} attempt${added === 1 ? '' : 's'} from that file.` : 'Everything in that file was already here.')
      load()
    } catch (failure) {
      setNotice(failure.message)
    }
  }

  const clearAll = async () => {
    await clearHistory()
    setConfirmingClear(false)
    load()
  }

  if (error) {
    return (
      <MessageState
        title="History is unavailable"
        action={<button type="button" className="button" onClick={retry}>Try again</button>}
      >
        {error}
      </MessageState>
    )
  }
  if (!history || !lexicon) return <LoadingState label="Opening the ledger" />

  const importControl = (
    <>
      <input ref={fileInput} type="file" accept="application/json,.json" className="visually-hidden" onChange={onImport} tabIndex={-1} />
      <button type="button" className="text-button" onClick={() => fileInput.current?.click()}>Import a backup</button>
    </>
  )

  if (history.length === 0) {
    return (
      <MessageState title="Nothing in the ledger yet" action={importControl}>
        {notice ?? 'Say your first word on the Practice page and it will be logged here. History is kept in this browser only.'}
      </MessageState>
    )
  }

  const visible = scoped.filter((item) => {
    if (scoreFilter === 'high') return item.score >= 8
    if (scoreFilter === 'low') return item.score < 5
    return true
  })
  const accentName = (id) => lexicon.accents[id]?.name ?? id
  const stamp = new Date().toISOString().slice(0, 10)

  return (
    <div className="page">
      <header className="page__head">
        <p className="label">Practice history</p>
        <h1 className="page__title">The ledger</h1>
        <p className="page__lede">
          {history.length} attempt{history.length === 1 ? '' : 's'} on record, kept in this browser. Nothing is
          uploaded, so save a backup if you want to move to another device or clear your browser data.
        </p>
      </header>

      <div className="tabs tabs--lead" role="group" aria-label="Filter by accent">
        {[['all', 'All accents'], ...Object.entries(lexicon.accents).map(([id, info]) => [id, info.name])].map(([id, label]) => (
          <button
            key={id}
            type="button"
            className={`tabs__tab${accentFilter === id ? ' is-active' : ''}`}
            aria-pressed={accentFilter === id}
            onClick={() => setAccentFilter(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {stats ? (
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
      ) : (
        <p className="focus__empty">No attempts in {accentName(accentFilter)} yet.</p>
      )}

      {stats && (
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
            <h2 className="section-title">Weakest sounds, {accentName(insightAccent)}</h2>
            {insights?.weak_phonemes?.length > 0 ? (
              <ul className="bars">
                {insights.weak_phonemes.map((item) => (
                  <li key={item.phoneme} className="bars__row">
                    <span className="bars__name">
                      <span className="ipa">{toIpa(item.phoneme, insightAccent)}</span> <span className="bars__code">{item.phoneme}</span>
                    </span>
                    <span className="meter" aria-hidden="true">
                      <span className="meter__fill meter__fill--accent" style={{ transform: `scaleX(${item.error_rate})` }} />
                    </span>
                    <span className="bars__value">{Math.round(item.error_rate * 100)}%</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="focus__empty">No missed sounds on record for this accent.</p>
            )}
          </section>
        </div>
      )}

      <div className="toolbar">
        <div className="tabs" role="group" aria-label="Filter by score">
          {SCORE_FILTERS.map(([id, label]) => (
            <button
              key={id}
              type="button"
              className={`tabs__tab${scoreFilter === id ? ' is-active' : ''}`}
              aria-pressed={scoreFilter === id}
              onClick={() => setScoreFilter(id)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="toolbar__actions">
          <button
            type="button"
            className="text-button"
            onClick={() => download(`phonemepro-history-${stamp}.json`, exportHistory(history), 'application/json')}
          >
            Save a backup
          </button>
          {importControl}
          <button type="button" className="text-button" onClick={() => download(`phonemepro-history-${stamp}.csv`, toCsv(history), 'text/csv')}>
            Export CSV
          </button>
          {confirmingClear ? (
            <span className="confirm">
              Erase all {history.length}?
              <button type="button" className="text-button text-button--accent" onClick={clearAll}>Yes, erase</button>
              <button type="button" className="text-button" onClick={() => setConfirmingClear(false)}>Keep</button>
            </span>
          ) : (
            <button type="button" className="text-button" onClick={() => setConfirmingClear(true)}>Erase history</button>
          )}
        </div>
      </div>

      {notice && <p className="notice notice--static" role="status">{notice}</p>}

      {visible.length === 0 ? (
        <p className="focus__empty">No attempts match these filters.</p>
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
                <span className="tag">{accentName(item.accent ?? 'ga')}</span>
              </span>
              <span className="ledger__word">{item.word}</span>
              <PhonemeStrip results={attemptResults(item)} accent={item.accent ?? 'ga'} animate={false} compact />
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
