import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { LoadingState, MessageState } from '../components/States'
import {
  clearContributions, clearDecisions, exportContributions, getContributions, MAX_STORED, recentDecisions, routeOf,
} from '../engine/contributions'
import { countryName } from '../lib/countries'
import { regionName } from '../lib/regions'
import { useApp } from '../state'

const formatTime = (iso) =>
  new Date(iso).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', second: '2-digit' })

const yesNo = (value) => (value ? 'yes' : 'no')

function download(filename, text) {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

function Checks({ checks }) {
  return (
    <ul className="checks">
      {checks.map((check) => (
        <li key={check.id} className={`checks__item${check.passed ? '' : ' checks__item--failed'}`}>
          <span className="checks__mark" aria-hidden="true">{check.passed ? '✓' : '✕'}</span>
          <span className="visually-hidden">{check.passed ? 'Passed:' : 'Failed:'}</span>
          {check.label} <span className="checks__detail">{check.detail}</span>
        </li>
      ))}
    </ul>
  )
}

// A local inspection page: exactly what "Help it learn" has kept in this browser,
// how each take was labelled, and why recent takes were or were not kept.
function DataPage() {
  const { lexicon, profile, saveProfile, openWelcome } = useApp()
  const [takes, setTakes] = useState(null)
  const [decisions, setDecisions] = useState(recentDecisions)
  const [error, setError] = useState(null)
  const [confirming, setConfirming] = useState(false)

  const load = useCallback(() => {
    getContributions().then(setTakes).catch(() => setError('This browser would not open its local storage.'))
  }, [])

  useEffect(load, [load])

  if (error) return <MessageState title="Your data could not be read">{error}</MessageState>
  if (!takes) return <LoadingState label="Reading what this browser keeps" />

  const accentName = (id) => lexicon?.accents[id]?.name ?? id
  const stamp = new Date().toISOString().slice(0, 10)

  const deleteEverything = async () => {
    await clearContributions().catch(() => {})
    clearDecisions()
    // A new random id next time, so later takes cannot be tied to the deleted ones.
    saveProfile({ ...profile, contributor: null })
    setConfirming(false)
    setDecisions([])
    load()
  }

  return (
    <div className="page">
      <header className="page__head">
        <p className="label">Your data · local test page</p>
        <h1 className="page__title">What this browser keeps</h1>
        <p className="page__lede">
          Everything “Help it learn” has stored, with every label, exactly as it would be exported.
          It all lives in this browser. Nothing has been sent anywhere, because sharing is not switched on.{' '}
          <Link to="/privacy">Privacy notice</Link>.
        </p>
      </header>

      <section aria-label="Your profile">
        <h2 className="section-title">Your profile</h2>
        <dl className="spec">
          <div><dt>Help it learn</dt><dd>{profile.contribute ? 'On' : 'Off: nothing new is being kept'}</dd></div>
          <div><dt>Country</dt><dd>{profile.country ? `${countryName(profile.country)} (${profile.country})` : 'Not given'}</dd></div>
          <div><dt>Region</dt><dd>{profile.region ? `${regionName(profile.country, profile.region)} (${profile.region})` : 'Not given'}</dd></div>
          <div><dt>Random id for this browser</dt><dd className="mono">{profile.contributor ?? 'None yet'}</dd></div>
        </dl>
        <button type="button" className="button button--ghost" onClick={openWelcome}>Change or withdraw</button>
      </section>

      <section aria-label="Stored takes">
        <h2 className="section-title">Stored takes · {takes.length} of {MAX_STORED}</h2>
        <div className="toolbar">
          <p className="prose">
            Each is a 256-number summary of one take plus its labels. No audio is kept, here or anywhere.
            Only a normal-voice take counts as how your country or region sounds. Every practice take counts
            as an attempt at its target accent, whatever your country.
          </p>
          <div className="toolbar__actions">
            <button
              type="button"
              className="text-button"
              disabled={takes.length === 0}
              onClick={() => download(`phonemepro-contributions-${stamp}.json`, exportContributions(takes))}
            >
              Export to a JSON file
            </button>
            {confirming ? (
              <span className="confirm">
                Delete all {takes.length} and the log below?
                <button type="button" className="text-button text-button--accent" onClick={deleteEverything}>Yes, delete</button>
                <button type="button" className="text-button" onClick={() => setConfirming(false)}>Keep</button>
              </span>
            ) : (
              <button
                type="button"
                className="text-button"
                disabled={takes.length === 0 && decisions.length === 0}
                onClick={() => setConfirming(true)}
              >
                Delete everything
              </button>
            )}
          </div>
        </div>

        {takes.length === 0 ? (
          <p className="focus__empty">
            Nothing is stored. {profile.contribute
              ? 'Record a clear take on Practice and it will appear here.'
              : 'Turn on “Help it learn” in your profile, then record a take on Practice.'}
          </p>
        ) : (
          <ol className="dataset">
            {takes.map((take) => {
              const use = routeOf(take)
              return (
                <li key={take.id} className="dataset__row">
                  <div className="dataset__head">
                    <span className="dataset__word">{take.word}</span>
                    <span className="label">{formatTime(take.created)}</span>
                    <span className={`tag${use.kind === 'native' ? ' tag--accent' : ''}`}>
                      counts as {use.kind === 'native' ? 'native accent' : 'attempt'}: {use.label}
                    </span>
                  </div>
                  <dl className="dataset__labels">
                    <div><dt>country</dt><dd>{take.country}</dd></div>
                    <div><dt>region</dt><dd>{take.region ?? 'none'} <small>{regionName(take.country, take.region)}</small></dd></div>
                    <div><dt>target_accent</dt><dd>{take.target_accent ?? 'none'} <small>{take.target_accent ? accentName(take.target_accent) : 'normal voice'}</small></dd></div>
                    <div><dt>natural_voice</dt><dd>{yesNo(take.natural_voice)}</dd></div>
                    <div><dt>matches_home_accent</dt><dd>{yesNo(take.matches_home_accent)}</dd></div>
                    <div><dt>score</dt><dd>{take.score ?? 'not scored'}</dd></div>
                    <div><dt>speech</dt><dd>{take.quality.speech_seconds} s</dd></div>
                    <div><dt>snr</dt><dd>{take.quality.snr_db} dB</dd></div>
                    <div><dt>clipped</dt><dd>{(take.quality.clipped * 100).toFixed(2)}%</dd></div>
                    <div><dt>confidence</dt><dd>{take.quality.confidence}</dd></div>
                    <div><dt>sounds heard</dt><dd>{take.quality.sounds_heard ?? 'not recorded'}</dd></div>
                    <div><dt>heard as written</dt><dd>{take.quality.sounds_matched == null ? 'not compared' : `${take.quality.sounds_matched}%`}</dd></div>
                    <div><dt>same sound class</dt><dd>{take.quality.sounds_same_class == null ? 'not compared' : `${take.quality.sounds_same_class}%`}</dd></div>
                    <div><dt>model_version</dt><dd>{take.model_version}</dd></div>
                    <div className="dataset__wide"><dt>joins totals</dt><dd>{use.keys.join(', ')}</dd></div>
                    <div className="dataset__wide"><dt>id</dt><dd>{take.id}</dd></div>
                    <div className="dataset__wide"><dt>contributor</dt><dd>{take.contributor}</dd></div>
                  </dl>
                  <details className="dataset__features">
                    <summary>features · {take.features.length} numbers · starts {take.features.slice(0, 4).join(', ')} …</summary>
                    <p>{take.features.join(' ')}</p>
                  </details>
                </li>
              )
            })}
          </ol>
        )}
      </section>

      <section aria-label="Recent takes">
        <h2 className="section-title">Recent takes and why</h2>
        <p className="prose">
          The last takes in this tab, kept or not, with each check. This log holds no summaries, is never
          exported, and disappears when the tab closes.
        </p>
        {decisions.length === 0 ? (
          <p className="focus__empty">No takes yet in this tab.</p>
        ) : (
          <ol className="dataset">
            {decisions.map((decision) => (
              <li key={decision.at} className="dataset__row">
                <div className="dataset__head">
                  <span className="dataset__word">{decision.word}</span>
                  <span className="label">{formatTime(decision.at)}</span>
                  <span className="label">{decision.natural_voice ? 'normal voice' : `practising ${accentName(decision.target_accent)}`}</span>
                  <span className={`tag${decision.stored ? ' tag--accent' : ''}`}>
                    {decision.stored ? 'stored' : `not stored: ${decision.reason}`}
                  </span>
                </div>
                <Checks checks={decision.checks} />
              </li>
            ))}
          </ol>
        )}
      </section>
    </div>
  )
}

export default DataPage
