import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { clearContributions, clearDecisions, considerTake, countContributions } from '../engine/contributions'
import { textEntry } from '../engine/lexicon'
import { recognize } from '../engine/model'
import { scoreAttempt } from '../engine/scoring'
import { useRecorder } from '../hooks/useRecorder'
import { countries } from '../lib/countries'
import { isRegion, regionsOf } from '../lib/regions'
import { useApp } from '../state'

// Short, ordinary sentences. They are read in the speaker's own accent, not scored.
const SENTENCES = ['the weather is very cold today', 'my mother walks in the garden']

// How much of the sentence was attempted, against whichever dictionary fits the speaker best.
function soundsAttempted(lexicon, sentence, heard) {
  return Math.max(...Object.keys(lexicon.accents).map((accent) =>
    scoreAttempt(textEntry(lexicon, sentence, accent).phonemes, heard.phonemes, heard.confidence, heard.margin)
      .metrics.completeness))
}

// The first-visit screen, and later the place to change or withdraw what was said here.
// Every part is optional and nothing on it leaves the device.
function WelcomeSheet() {
  const { lexicon, model, ensureModel, profile, saveProfile, closeWelcome } = useApp()
  const dialogRef = useRef(null)
  const [country, setCountry] = useState(profile.country ?? '')
  const [region, setRegion] = useState(profile.region ?? '')
  const [contribute, setContribute] = useState(profile.contribute)
  // Natural-voice takes wait here until the screen is saved; they are stored only if the switch is on.
  const [voiceTakes, setVoiceTakes] = useState({})
  const [active, setActive] = useState(null)
  const [analysing, setAnalysing] = useState(false)
  const [voiceNote, setVoiceNote] = useState(null)
  const [storedCount, setStoredCount] = useState(0)
  const activeRef = useRef(null)
  const countryList = useMemo(countries, [])
  const regionList = regionsOf(country)
  const returning = profile.seen

  useEffect(() => {
    const dialog = dialogRef.current
    if (!dialog.open) dialog.showModal()
    countContributions().then(setStoredCount).catch(() => {})
    return () => dialog.close()
  }, [])

  const handleVoice = useCallback(async ({ samples, quality }) => {
    const sentence = activeRef.current
    if (!sentence) return
    setAnalysing(true)
    try {
      await ensureModel()
      await new Promise((resolve) => { setTimeout(resolve, 40) })
      const heard = await recognize(samples)
      setVoiceTakes((takes) => ({
        ...takes,
        [sentence]: { heard, quality, soundsAttempted: soundsAttempted(lexicon, sentence, heard) },
      }))
    } catch (error) {
      setVoiceNote(error.message || 'The model could not analyze that recording.')
    } finally {
      setAnalysing(false)
      setActive(null)
    }
  }, [ensureModel, lexicon])

  const handleVoiceError = useCallback((message) => {
    setVoiceNote(message)
    setActive(null)
  }, [])

  const { isStarting, isRecording, start, stop } = useRecorder({ onComplete: handleVoice, onError: handleVoiceError })
  const recording = isStarting || isRecording

  const toggleRecording = (sentence) => {
    if (recording) {
      stop()
      return
    }
    activeRef.current = sentence
    setActive(sentence)
    setVoiceNote(null)
    ensureModel().catch(() => {})
    start()
  }

  const finish = async (next) => {
    if (recording) stop()
    const saved = saveProfile({ ...profile, ...next, seen: true })
    // Natural-voice takes are kept only when the visitor has opted in; otherwise they are dropped here.
    for (const [sentence, take] of Object.entries(next.keepVoice ? voiceTakes : {})) {
      await considerTake({
        profile: saved, targetAccent: null, naturalVoice: true, word: sentence, score: null,
        heard: take.heard, quality: take.quality, soundsAttempted: take.soundsAttempted, featureCount: take.heard.featureCount,
      })
    }
    closeWelcome()
  }

  const save = (event) => {
    event.preventDefault()
    finish({ country: country || null, region: isRegion(country, region) ? region : null, contribute, keepVoice: true })
  }

  // Closing without saving keeps whatever was saved before.
  const skip = () => finish({})

  const withdraw = async () => {
    await clearContributions().catch(() => {})
    clearDecisions()
    finish({ contribute: false, contributor: null })
  }

  return (
    <dialog ref={dialogRef} className="welcome" aria-labelledby="welcome-title" onCancel={(event) => { event.preventDefault(); skip() }}>
      <form className="welcome__sheet" onSubmit={save}>
        <header className="welcome__head">
          <p className="label">{returning ? 'Your profile' : 'Before you start'}</p>
          <h2 id="welcome-title" className="welcome__title">
            {returning ? 'Change what you told us' : <>Three optional questions</>}
          </h2>
          <p className="welcome__lede">
            PhonemePro listens on this device. Nothing you say or type here is sent anywhere.
            Answer what you like, or skip all of it.
          </p>
        </header>

        <section className="welcome__part" aria-labelledby="welcome-where">
          <p className="label">One · optional</p>
          <h3 id="welcome-where" className="welcome__question">Where are you from?</h3>
          <div className="welcome__fields">
            <label className="select welcome__field">
              <span className="label">Country</span>
              <select
                value={country}
                onChange={(event) => {
                  setCountry(event.target.value)
                  setRegion('')
                }}
              >
                <option value="">Prefer not to say</option>
                {countryList.map(({ code, name }) => <option key={code} value={code}>{name}</option>)}
              </select>
            </label>
            {regionList.length > 0 && (
              <label className="select welcome__field">
                <span className="label">Region</span>
                <select value={region} onChange={(event) => setRegion(event.target.value)}>
                  <option value="">Rather not say</option>
                  {regionList.map(([code, name]) => <option key={code} value={code}>{name}</option>)}
                </select>
              </label>
            )}
          </div>
          {regionList.length > 0 && (
            <p className="welcome__text">Regions are broad on purpose. There is nowhere to type a town or city.</p>
          )}
        </section>

        <section className="welcome__part" aria-labelledby="welcome-voice">
          <p className="label">Two · optional</p>
          <h3 id="welcome-voice" className="welcome__question">Say this in your normal voice</h3>
          <p className="welcome__text">
            Your own accent, not one you are practising. The recording is turned into a short summary and then
            discarded. The summary is kept only if you turn on the switch below, apart from your practice takes.
          </p>
          <ul className="welcome__sentences">
            {SENTENCES.map((sentence) => {
              const take = voiceTakes[sentence]
              const mine = active === sentence
              return (
                <li key={sentence} className="welcome__sentence">
                  <button
                    type="button"
                    className={`record record--small${mine && isRecording ? ' record--live' : ''}`}
                    onClick={() => toggleRecording(sentence)}
                    disabled={!lexicon || analysing || (recording && !mine)}
                    aria-label={mine && recording ? `Stop recording "${sentence}"` : `Record "${sentence}"`}
                  >
                    <span className="record__dot" />
                  </button>
                  <span className="welcome__say">“{sentence[0].toUpperCase()}{sentence.slice(1)}.”</span>
                  <span className="label" role="status">
                    {mine && isRecording ? 'Recording. Tap to stop'
                      : mine && analysing ? 'Listening back'
                        : take ? `Heard · ${take.quality.speechSeconds.toFixed(1)} s · tap to redo`
                          : 'Tap to record'}
                  </span>
                </li>
              )
            })}
          </ul>
          {model.status === 'loading' && (active || analysing) && (
            <p className="label">Downloading the listening model, once · {Math.round(model.progress * 100)}%</p>
          )}
          {voiceNote && <p className="welcome__note" role="alert">{voiceNote}</p>}
        </section>

        <section className="welcome__part" aria-labelledby="welcome-learn">
          <p className="label">Three · optional · off unless you turn it on</p>
          <h3 id="welcome-learn" className="welcome__question">Help it learn</h3>
          <button
            type="button"
            className={`toggle${contribute ? ' is-active' : ''}`}
            role="switch"
            aria-checked={contribute}
            onClick={() => setContribute((on) => !on)}
          >
            <span className="toggle__knob" aria-hidden="true" />
            {contribute ? 'On: keep summaries of my clear takes on this device' : 'Off: keep nothing for learning'}
          </button>
          <p className="welcome__text">
            When on, each clear take is reduced to 256 numbers and kept in this browser with the labels above.
            It is not audio and cannot be played back. Sharing is not switched on: nothing is sent anywhere yet,
            and you can see or delete everything on <Link to="/data" onClick={skip}>Your data</Link>.{' '}
            <Link to="/privacy" onClick={skip}>Read the privacy notice</Link>.
          </p>
          {contribute && !country && (
            <p className="welcome__note">Takes are kept only once a country is chosen, so they can be placed in a region.</p>
          )}
        </section>

        <footer className="welcome__actions">
          <button type="submit" className="button">{returning ? 'Save changes' : 'Save and start'}</button>
          <button type="button" className="text-button" onClick={skip}>{returning ? 'Close without saving' : 'Skip for now'}</button>
          {returning && (profile.contribute || storedCount > 0) && (
            <button type="button" className="text-button text-button--accent" onClick={withdraw}>
              Withdraw: turn off and delete the {storedCount} kept
            </button>
          )}
        </footer>
      </form>
    </dialog>
  )
}

// Mounted afresh each time it opens, so the form always starts from the saved profile.
function Welcome() {
  const { welcomeOpen } = useApp()
  return welcomeOpen ? <WelcomeSheet /> : null
}

export default Welcome
