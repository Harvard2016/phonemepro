import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import CountUp from '../components/CountUp'
import Headword from '../components/Headword'
import PhonemeStrip from '../components/PhonemeStrip'
import TakeTimeline from '../components/TakeTimeline'
import { LoadingState, MessageState } from '../components/States'
import Waveform from '../components/Waveform'
import { useRecorder } from '../hooks/useRecorder'
import { analyze, getInsights, getPhonemes, getTodayCount, getWords } from '../lib/api'
import { speak } from '../lib/audio'
import { basePhoneme, toIpa } from '../lib/phonemes'

const LEVELS = ['easy', 'medium', 'hard']
const DAILY_GOAL = 10
const PHRASES = ['thirty three thin thieves', 'red lorry yellow lorry', 'the weather is very cold', 'she sells sea shells']
// Played around a single sound so a 40 ms phoneme is still audible.
const SLICE_PADDING = 0.08
const EASE = [0.2, 0.8, 0.2, 1]

const shuffled = (words, exclude) => {
  const pool = words.filter((word) => word.word !== exclude)
  for (let i = pool.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[pool[i], pool[j]] = [pool[j], pool[i]]
  }
  return pool
}

const isTypingTarget = (target) =>
  target instanceof HTMLElement
  && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', 'A'].includes(target.tagName))

function PracticePage() {
  const [words, setWords] = useState(null)
  const [loadError, setLoadError] = useState(null)
  const [level, setLevel] = useState('medium')
  const [word, setWord] = useState(null)
  const [result, setResult] = useState(null)
  const [notice, setNotice] = useState(null)
  const [analysing, setAnalysing] = useState(false)
  const [playingWord, setPlayingWord] = useState(false)
  const [repeatAfterMe, setRepeatAfterMe] = useState(false)
  const [streak, setStreak] = useState(0)
  const [todayCount, setTodayCount] = useState(0)
  const [insights, setInsights] = useState(null)
  const [take, setTake] = useState(null)
  const [customText, setCustomText] = useState('')
  const [lookingUp, setLookingUp] = useState(false)
  const [armed, setArmed] = useState(false)

  const wordRef = useRef(null)
  const queueRef = useRef([])
  const armedRef = useRef(false)
  const resultRef = useRef(null)
  const spaceHeldRef = useRef(false)

  const levelWords = useMemo(() => (words ?? []).filter((w) => w.difficulty === level), [words, level])

  const refreshProgress = useCallback(() => {
    getTodayCount().then((data) => setTodayCount(data.count ?? 0)).catch(() => {})
    getInsights().then(setInsights).catch(() => {})
  }, [])

  const showWord = useCallback((next) => {
    wordRef.current = next
    setWord(next)
    setResult(null)
    setNotice(null)
    setTake((previous) => {
      if (previous) URL.revokeObjectURL(previous.url)
      return null
    })
  }, [])

  const loadWords = useCallback(() => {
    setLoadError(null)
    getWords()
      .then((data) => {
        setWords(data)
        const pool = shuffled(data.filter((w) => w.difficulty === 'medium'))
        queueRef.current = pool.slice(1)
        showWord(pool[0] ?? data[0] ?? null)
      })
      .catch((error) => setLoadError(error.message))
  }, [showWord])

  useEffect(() => {
    loadWords()
    refreshProgress()
  }, [loadWords, refreshProgress])

  const changeLevel = (nextLevel) => {
    if (nextLevel === level || !words) return
    setLevel(nextLevel)
    const pool = shuffled(words.filter((w) => w.difficulty === nextLevel))
    queueRef.current = pool.slice(1)
    showWord(pool[0])
  }

  const goToWord = useCallback((next) => {
    if (!next || !words) return
    setLevel(next.difficulty)
    queueRef.current = shuffled(words.filter((w) => w.difficulty === next.difficulty), next.word)
    showWord(next)
  }, [words, showWord])

  const nextWord = useCallback(() => {
    if (!queueRef.current.length) queueRef.current = shuffled(levelWords, wordRef.current?.word)
    const [next, ...rest] = queueRef.current
    queueRef.current = rest
    if (next) showWord(next)
  }, [levelWords, showWord])

  const handleTake = useCallback(async (wavBlob) => {
    const taken = wordRef.current
    if (!taken) return
    setAnalysing(true)
    try {
      const data = await analyze(wavBlob, taken)
      // Ignore the answer if the learner already moved on to another word.
      if (wordRef.current?.word !== taken.word) return
      setResult(data)
      setTake((previous) => {
        if (previous) URL.revokeObjectURL(previous.url)
        return { blob: wavBlob, url: URL.createObjectURL(wavBlob) }
      })
      setStreak((current) => (data.score >= 8 ? current + 1 : 0))
      refreshProgress()
    } catch (error) {
      setNotice(error.message)
    } finally {
      setAnalysing(false)
    }
  }, [refreshProgress])

  const handleRecorderError = useCallback((message) => {
    armedRef.current = false
    setArmed(false)
    setNotice(message)
  }, [])

  const { isStarting, isRecording, start, stop, analyserRef } = useRecorder({ onComplete: handleTake, onError: handleRecorderError })
  const busy = analysing || playingWord
  // A take is in progress from the first tap until it is stopped, including the
  // word playback and microphone start-up, so a second tap always cancels it.
  const takeActive = armed || isStarting || isRecording

  const beginTake = useCallback(async () => {
    if (analysing || armedRef.current || !wordRef.current) return
    armedRef.current = true
    setArmed(true)
    setResult(null)
    setNotice(null)
    if (repeatAfterMe) {
      setPlayingWord(true)
      await speak(wordRef.current.word)
      setPlayingWord(false)
    }
    // Space may have been released while the word was playing.
    if (armedRef.current) start()
  }, [analysing, repeatAfterMe, start])

  const endTake = useCallback(() => {
    armedRef.current = false
    setArmed(false)
    stop()
  }, [stop])

  // Play the whole take, or only the slice between two timestamps.
  const playTake = useCallback((start, end) => {
    if (!take) return
    const audio = new Audio(take.url)
    if (start !== undefined) {
      audio.currentTime = Math.max(0, start - SLICE_PADDING)
      const stopAt = end + SLICE_PADDING
      audio.ontimeupdate = () => {
        if (audio.currentTime >= stopAt) audio.pause()
      }
    }
    audio.play()
  }, [take])

  const practiseText = async (text) => {
    if (!text.trim() || lookingUp) return
    setLookingUp(true)
    setNotice(null)
    try {
      const entry = await getPhonemes(text)
      if (entry.difficulty === 'custom') {
        queueRef.current = []
        showWord(entry)
      } else {
        goToWord(entry)
      }
      setCustomText('')
    } catch (error) {
      setNotice(error.message)
    } finally {
      setLookingUp(false)
    }
  }

  const hearWord = async () => {
    if (busy || takeActive || !word) return
    setPlayingWord(true)
    await speak(word.word)
    setPlayingWord(false)
  }

  // Bring a fresh result into view; it lands below the fold on short screens.
  useEffect(() => {
    if (!result) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    resultRef.current?.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'nearest' })
  }, [result])

  // Hold space to record, release to stop.
  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.code !== 'Space' || isTypingTarget(event.target)) return
      event.preventDefault()
      if (event.repeat || spaceHeldRef.current) return
      spaceHeldRef.current = true
      beginTake()
    }
    const onKeyUp = (event) => {
      if (event.code !== 'Space' || !spaceHeldRef.current) return
      event.preventDefault()
      spaceHeldRef.current = false
      endTake()
    }
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
    }
  }, [beginTake, endTake])

  const recommended = useMemo(() => {
    if (!result || !words) return null
    const missed = result.phoneme_results.filter((p) => p.status !== 'correct').map((p) => basePhoneme(p.phoneme))
    const candidates = [...levelWords, ...words].filter((w) => w.word !== result.word)
    for (const phoneme of missed) {
      const match = candidates.find((w) => w.phonemes.some((p) => basePhoneme(p) === phoneme))
      if (match) return { word: match, phoneme }
    }
    return null
  }, [result, words, levelWords])

  if (loadError) {
    return (
      <MessageState
        title="The word list did not load"
        action={<button type="button" className="button" onClick={loadWords}>Try again</button>}
      >
        {loadError}
      </MessageState>
    )
  }
  if (!words || !word) return <LoadingState label="Opening the word list" />

  const isCustom = word.difficulty === 'custom'
  const entryNumber = levelWords.findIndex((w) => w.word === word.word) + 1
  const mode = isRecording ? 'recording' : analysing ? 'thinking' : 'idle'
  const hint = analysing
    ? 'Listening back'
    : playingWord
      ? 'Listen first'
      : isRecording
        ? 'Recording. Release space or tap to stop'
        : repeatAfterMe
          ? 'Hold space or tap. The word plays, then you record'
          : 'Hold space or tap to record'

  return (
    <div className="practice">
      <div className="practice__meta">
        <span className="label">
          {isCustom ? 'Your own text' : `Entry ${String(entryNumber).padStart(2, '0')} / ${levelWords.length} · ${level}`}
        </span>
        <span className="goal" title={`Daily goal: ${DAILY_GOAL} words`}>
          <span className="label">Today {Math.min(todayCount, 999)} / {DAILY_GOAL}</span>
          <span className="goal__ticks" aria-hidden="true">
            {Array.from({ length: DAILY_GOAL }, (_, i) => (
              <span key={i} className={`goal__tick${i < todayCount ? ' goal__tick--done' : ''}`} />
            ))}
          </span>
          {streak >= 3 && <span className="label label--accent">{streak} in a row</span>}
        </span>
      </div>

      <section className="stage" aria-label="Current word">
        <Headword word={word} />
        <button type="button" className="text-button" onClick={hearWord} disabled={busy || takeActive}>
          {playingWord ? 'Playing' : 'Hear it'}
        </button>

        <div className={`stave stave--${mode}`}>
          <Waveform analyserRef={analyserRef} mode={mode} />
          <button
            type="button"
            className={`record${isRecording ? ' record--live' : ''}`}
            onClick={takeActive ? endTake : beginTake}
            disabled={analysing}
            aria-label={takeActive ? 'Stop recording' : 'Start recording'}
          >
            <span className="record__dot" />
          </button>
        </div>
        <p className="stage__hint label" role="status">{hint}</p>
      </section>

      <section className="controls" aria-label="Word settings">
        <div className="segmented" role="group" aria-label="Difficulty">
          {LEVELS.map((option) => (
            <button
              key={option}
              type="button"
              className={`segmented__option${option === level && !isCustom ? ' is-active' : ''}`}
              aria-pressed={option === level && !isCustom}
              onClick={() => (isCustom ? goToWord(words.find((w) => w.difficulty === option)) : changeLevel(option))}
            >
              {option}
            </button>
          ))}
        </div>
        <label className="select">
          <span className="visually-hidden">Choose a word</span>
          <select value={word.word} onChange={(e) => goToWord(levelWords.find((w) => w.word === e.target.value))}>
            {isCustom && <option value={word.word} disabled>your own text</option>}
            {levelWords.map((w) => <option key={w.word} value={w.word}>{w.word}</option>)}
          </select>
        </label>
        <button type="button" className="button button--ghost" onClick={nextWord} disabled={busy || takeActive}>
          Shuffle
        </button>
        <button
          type="button"
          className={`toggle${repeatAfterMe ? ' is-active' : ''}`}
          aria-pressed={repeatAfterMe}
          onClick={() => setRepeatAfterMe((on) => !on)}
        >
          <span className="toggle__knob" aria-hidden="true" />
          Repeat after me
        </button>
      </section>

      <form
        className="own-text"
        onSubmit={(event) => {
          event.preventDefault()
          practiseText(customText)
        }}
      >
        <label className="own-text__field">
          <span className="label">Or practise your own word or phrase</span>
          <input
            type="text"
            value={customText}
            onChange={(event) => setCustomText(event.target.value)}
            maxLength={60}
            placeholder="Type up to eight words"
            autoComplete="off"
            spellCheck="false"
          />
        </label>
        <button type="submit" className="button button--ghost" disabled={!customText.trim() || lookingUp || takeActive}>
          {lookingUp ? 'Looking up' : 'Practise this'}
        </button>
        <div className="own-text__ideas">
          {PHRASES.map((phrase) => (
            <button key={phrase} type="button" className="text-button" onClick={() => practiseText(phrase)} disabled={lookingUp || takeActive}>
              {phrase}
            </button>
          ))}
        </div>
      </form>

      <AnimatePresence mode="wait">
        {notice && (
          <motion.div
            key="notice"
            className="notice"
            role="alert"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            <p>{notice}</p>
            <button type="button" className="text-button" onClick={() => setNotice(null)}>Dismiss</button>
          </motion.div>
        )}

        {result && (
          <motion.section
            key={`result-${result.word}`}
            ref={resultRef}
            className="result"
            aria-label="Result"
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5, ease: EASE }}
          >
            <div className="result__score">
              <p className={`numeral${result.score < 5 ? ' numeral--low' : ''}`}>
                <CountUp value={result.score} />
                <span className="numeral__unit">/ 10</span>
              </p>
              <p className="result__verdict">{result.feedback}</p>
              {take && (
                <button type="button" className="text-button" onClick={() => playTake()}>
                  Play my take
                </button>
              )}
            </div>

            <div className="result__detail">
              <p className="label">What I heard</p>
              {result.predicted_phonemes.length > 0
                ? <PhonemeStrip results={result.phoneme_results} onPlay={take ? playTake : undefined} />
                : <p className="result__empty">Nothing recognizable in that take.</p>}

              {take && <TakeTimeline blob={take.blob} results={result.phoneme_results} onPlay={playTake} />}

              <dl className="marginalia">
                {[
                  ['Accuracy', result.metrics.accuracy, 'sounds matched'],
                  ['Completeness', result.metrics.completeness, 'sounds attempted'],
                  ['Fluency', result.metrics.fluency, 'clean, no extras'],
                ].map(([name, value, gloss]) => (
                  <div key={name} className="marginalia__item">
                    <dt className="label">{name}</dt>
                    <dd>
                      <span className="marginalia__value">{value}<small>%</small></span>
                      <span className="meter" aria-hidden="true">
                        <motion.span
                          className="meter__fill"
                          initial={{ scaleX: 0 }}
                          animate={{ scaleX: Math.min(value, 100) / 100 }}
                          transition={{ duration: 0.8, delay: 0.3, ease: EASE }}
                        />
                      </span>
                      <span className="marginalia__gloss">{gloss}</span>
                    </dd>
                  </div>
                ))}
              </dl>

              {result.coaching_tips?.length > 0 && (
                <ol className="footnotes">
                  {result.coaching_tips.map((tip) => <li key={tip}>{tip}</li>)}
                </ol>
              )}
            </div>

            <div className="result__actions">
              <button type="button" className="button button--ghost" onClick={() => showWord(word)}>
                Say it again
              </button>
              {recommended ? (
                <button type="button" className="button" onClick={() => goToWord(recommended.word)}>
                  Next: {recommended.word.word}
                  <span className="button__aside">more /{recommended.phoneme}/</span>
                </button>
              ) : (
                <button type="button" className="button" onClick={nextWord}>Next word</button>
              )}
            </div>
          </motion.section>
        )}
      </AnimatePresence>

      <section className="focus" aria-label="Sounds to focus on">
        <h2 className="section-title">Your focus sounds</h2>
        {insights?.weak_phonemes?.length > 0 ? (
          <div className="focus__body">
            <ul className="focus__sounds">
              {insights.weak_phonemes.map((item) => (
                <li key={item.phoneme} className="focus__sound">
                  <span className="focus__symbol">
                    <span className="ipa">{toIpa(item.phoneme)}</span> <span className="bars__code">{item.phoneme}</span>
                  </span>
                  <span className="meter" aria-hidden="true">
                    <span className="meter__fill meter__fill--accent" style={{ transform: `scaleX(${item.error_rate})` }} />
                  </span>
                  <span className="focus__rate">
                    missed {item.errors} of {item.attempts}
                  </span>
                </li>
              ))}
            </ul>
            <div className="focus__words">
              <p className="label">Words that train them</p>
              <div className="focus__list">
                {insights.recommended_words.map((w) => (
                  <button key={w.word} type="button" className="chip" onClick={() => goToWord(w)}>
                    {w.word}
                    <span className="chip__aside">{w.focus.join(' ')}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <p className="focus__empty">
            Record a few words. The sounds you miss most will collect here, with words chosen to train them.
          </p>
        )}
      </section>
    </div>
  )
}

export default PracticePage
