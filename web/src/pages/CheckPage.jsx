import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { retakeOutcome, settleTake } from '../check/run'
import { checkExport, describeSetup, evaluateStep, formatResults, STEPS } from '../check/steps'
import { LoadingState } from '../components/States'
import Waveform from '../components/Waveform'
import { createMemoryStore, exportContributions } from '../engine/contributions'
import { recognize } from '../engine/model'
import { useRecorder } from '../hooks/useRecorder'
import { useApp } from '../state'
import '../check/check.css'

const STATUS_WORD = { pass: 'Pass', fail: 'Fail', info: 'Noted', skipped: 'Skipped' }

const isTypingTarget = (target) =>
  target instanceof HTMLElement && ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', 'A'].includes(target.tagName)

function Rows({ rows }) {
  return (
    <table className="table check__rows">
      <thead>
        <tr><th>Label</th><th>Expected</th><th>Actual</th><th><span className="visually-hidden">Result</span></th></tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.label} className={row.pass === false ? 'check__row--failed' : ''}>
            <td>{row.label}</td>
            <td>{row.expected}</td>
            <td>{row.actual}</td>
            <td>{row.pass === null ? '' : row.pass ? '✓' : '✕'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// A guided check of how takes are labelled, for development only (the route does not exist
// in a production build). It tells the tester what to say, runs the take through the app's
// own pipeline, and shows expected against actual for every label.
//
// It is a sandbox: each step uses its own made-up profile, and anything kept goes to a store
// in memory. The real profile, real contributions and practice history are never read or
// written, and everything here is gone when the page is left.
function CheckPage() {
  const { lexicon, model, ensureModel } = useApp()
  const [index, setIndex] = useState(0)
  const [results, setResults] = useState({})
  const [analysing, setAnalysing] = useState(false)
  const [problem, setProblem] = useState(null)
  const [copied, setCopied] = useState(false)
  const store = useMemo(createMemoryStore, [])
  const contributor = useMemo(() => crypto.randomUUID(), [])
  const modelVersionRef = useRef('unknown')
  const stepRef = useRef(STEPS[0])
  const spaceHeldRef = useRef(false)

  const finished = index >= STEPS.length
  const step = finished ? null : STEPS[index]
  const result = step ? results[step.id] : null

  useEffect(() => {
    stepRef.current = step
  }, [step])

  useEffect(() => {
    ensureModel().then(({ meta }) => { modelVersionRef.current = meta.version }).catch(() => {})
  }, [ensureModel])

  const record = useCallback((target, outcome) => {
    setResults((current) => ({ ...current, [target.id]: { outcome, evaluation: evaluateStep(target, outcome) } }))
  }, [])

  const handleTake = useCallback(async ({ samples, quality }) => {
    const target = stepRef.current
    if (!target || target.kind === 'export') return
    setAnalysing(true)
    try {
      await ensureModel()
      await new Promise((resolve) => { setTimeout(resolve, 40) })
      const heard = await recognize(samples)
      record(target, await settleTake({ step: target, lexicon, heard, quality, store, contributor }))
    } catch (error) {
      setProblem(error.message || 'The model could not analyze that recording.')
    } finally {
      setAnalysing(false)
    }
  }, [contributor, ensureModel, lexicon, record, store])

  // A refused take is still an outcome: silence is meant to be refused.
  const handleRefusal = useCallback((message, detail = {}) => {
    const target = stepRef.current
    if (!target || target.kind === 'export') return
    if (['silent', 'noisy', 'short'].includes(detail.reason)) record(target, retakeOutcome(detail))
    else setProblem(message)
  }, [record])

  const { isStarting, isRecording, start, stop, analyserRef } = useRecorder({ onComplete: handleTake, onError: handleRefusal })
  const modelReady = model.status === 'ready'
  const canRecord = Boolean(step) && step.kind !== 'export' && modelReady && lexicon && !analysing && !result
  const live = isStarting || isRecording

  const begin = useCallback(() => {
    if (!canRecord || live) return
    setProblem(null)
    start()
  }, [canRecord, live, start])

  // Hold space to record, release to stop, as on the Practice page.
  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.code !== 'Space' || isTypingTarget(event.target)) return
      event.preventDefault()
      if (event.repeat || spaceHeldRef.current) return
      spaceHeldRef.current = true
      begin()
    }
    const onKeyUp = (event) => {
      if (event.code !== 'Space' || !spaceHeldRef.current) return
      event.preventDefault()
      spaceHeldRef.current = false
      stop()
    }
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
    }
  }, [begin, stop])

  // Forget this step's result, and the take it kept, so the export only ever holds the last try.
  const tryAgain = () => {
    const takeId = results[step.id]?.outcome?.take_id
    if (takeId) store.remove(takeId)
    setProblem(null)
    setResults((current) => {
      const { [step.id]: _dropped, ...rest } = current
      return rest
    })
  }

  const runExportCheck = () => {
    const kept = Object.fromEntries(Object.entries(results).map(([id, entry]) => [id, entry.outcome?.take_id ?? null]))
    const evaluation = checkExport(store.all(), kept, modelVersionRef.current)
    setResults((current) => ({ ...current, [step.id]: { outcome: { detail: `${store.all().length} takes in the session store` }, evaluation } }))
  }

  const downloadExport = () => {
    const url = URL.createObjectURL(new Blob([exportContributions(store.all())], { type: 'application/json' }))
    const link = document.createElement('a')
    link.href = url
    link.download = 'phonemepro-check-export.json'
    link.click()
    URL.revokeObjectURL(url)
  }

  const summary = finished ? formatResults(results, { modelVersion: modelVersionRef.current }) : ''

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(summary)
      setCopied(true)
    } catch {
      setCopied(false)
      document.querySelector('.check__text')?.select()
    }
  }

  if (!lexicon) return <LoadingState label="Opening the dictionary" />

  const accentNames = Object.fromEntries(Object.entries(lexicon.accents).map(([id, info]) => [id, info.name]))

  if (finished) {
    return (
      <div className="page check">
        <header className="page__head">
          <p className="label">Label check · development only</p>
          <h1 className="page__title">Results</h1>
          <p className="page__lede">
            Nothing here touched your real profile, contributions or history. The session store is emptied when
            you leave this page.
          </p>
        </header>
        <table className="table">
          <thead><tr><th>Step</th><th>What</th><th>Result</th><th>Where it differed</th></tr></thead>
          <tbody>
            {STEPS.map((item, at) => {
              const entry = results[item.id]
              const status = entry?.evaluation.status ?? 'skipped'
              return (
                <tr key={item.id}>
                  <td>{item.id}</td>
                  <td><button type="button" className="text-button" onClick={() => setIndex(at)}>{item.title}</button></td>
                  <td className={`check__status check__status--${status}`}>{STATUS_WORD[status]}</td>
                  <td>
                    {(entry?.evaluation.rows ?? []).filter((row) => row.pass === false)
                      .map((row) => `${row.label}: expected ${row.expected}, got ${row.actual}`).join(' · ')}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <div className="check__actions">
          <button type="button" className="button" onClick={copy}>{copied ? 'Copied' : 'Copy results'}</button>
          <button type="button" className="text-button" onClick={() => setIndex(0)}>Back to step 1</button>
        </div>
        <textarea className="check__text" readOnly value={summary} rows={18} aria-label="Results as plain text" />
      </div>
    )
  }

  const status = result?.evaluation.status
  const mode = isRecording ? 'recording' : analysing ? 'thinking' : 'idle'
  const hint = model.status === 'error'
    ? 'The listening model did not load'
    : !modelReady
      ? `Downloading the listening model · ${Math.round(model.progress * 100)}%`
      : analysing
        ? 'Listening back'
        : isRecording
          ? 'Recording. Release space or tap to stop'
          : result
            ? 'Done. Try again or go on'
            : 'Hold space and speak, or tap to record'

  return (
    <div className="page check">
      <p className="label">
        Label check · development only · step {step.id} of {STEPS.length} · sandbox: nothing real is read or written
      </p>
      <ol className="check__progress" aria-label="Steps">
        {STEPS.map((item, at) => {
          const state = results[item.id]?.evaluation.status ?? (at === index ? 'current' : 'todo')
          return (
            <li key={item.id}>
              <button
                type="button"
                className={`check__dot check__dot--${state}${at === index ? ' is-current' : ''}`}
                onClick={() => setIndex(at)}
                disabled={live || analysing}
                aria-label={`Step ${item.id}: ${item.title}`}
                aria-current={at === index ? 'step' : undefined}
              >
                {item.id}
              </button>
            </li>
          )
        })}
      </ol>

      <section className="check__stage" aria-label="Current step">
        <p className="label label--accent">{step.title}</p>
        <p className="check__setup">{describeSetup(step, accentNames)}</p>
        <h1 className="check__say">{step.say}</h1>
        <p className="check__how">{step.how}</p>

        {step.kind === 'export' ? (
          <div className="check__actions">
            <button type="button" className="button" onClick={runExportCheck}>Run the export check</button>
            <button type="button" className="text-button" onClick={downloadExport} disabled={store.all().length === 0}>
              Download this session’s export
            </button>
          </div>
        ) : (
          <>
            <div className={`stave stave--${mode}`}>
              <Waveform analyserRef={analyserRef} mode={mode} />
              <button
                type="button"
                className={`record${isRecording ? ' record--live' : ''}`}
                onClick={live ? stop : begin}
                disabled={!live && !canRecord}
                aria-label={live ? 'Stop recording' : 'Start recording'}
              >
                <span className="record__dot" />
              </button>
            </div>
            <p className="stage__hint label" role="status">{hint}</p>
          </>
        )}
        {problem && <p className="notice notice--static" role="alert">{problem}</p>}
      </section>

      {result && (
        <section className="check__result" aria-label="Result" aria-live="polite">
          <p className={`check__verdict check__status--${status}`}>{STATUS_WORD[status]}</p>
          <Rows rows={result.evaluation.rows} />
          {result.outcome?.detail && <p className="check__detail">{result.outcome.detail}</p>}
        </section>
      )}

      <div className="check__actions">
        {result && <button type="button" className="button button--ghost" onClick={tryAgain} disabled={live || analysing}>Try again</button>}
        <button type="button" className="button" onClick={() => setIndex(index + 1)} disabled={live || analysing}>
          {index === STEPS.length - 1 ? 'Finish' : result ? 'Next' : 'Skip'}
        </button>
      </div>
    </div>
  )
}

export default CheckPage
