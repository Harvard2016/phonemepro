import { useEffect, useState } from 'react'
import { BrowserRouter, NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { AnimatePresence, MotionConfig, motion } from 'motion/react'
import ErrorBoundary from './ErrorBoundary'
import { getModelInfo } from './lib/api'
import HistoryPage from './pages/HistoryPage'
import ModelInfoPage from './pages/ModelInfoPage'
import PracticePage from './pages/PracticePage'

const NAV = [
  ['/', 'Practice'],
  ['/history', 'History'],
  ['/model', 'Model'],
]

const STATUS = {
  loading: null,
  finetuned: { label: 'Fine-tuned model', tone: 'ok' },
  multitask: { label: 'Multitask model', tone: 'ok' },
  'asr-fallback': { label: 'Fallback model', tone: 'warn' },
  offline: { label: 'Backend offline', tone: 'warn' },
}

function Shell() {
  const location = useLocation()
  const [mode, setMode] = useState('loading')
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    getModelInfo()
      .then((info) => setMode(info.runtime?.mode ?? 'asr-fallback'))
      .catch(() => setMode('offline'))
  }, [])

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const status = STATUS[mode]

  return (
    <div className="app">
      <a className="skip-link" href="#main">Skip to content</a>
      <header className={`masthead${scrolled ? ' masthead--scrolled' : ''}`}>
        <NavLink to="/" className="wordmark" aria-label="PhonemePro home">
          Phoneme<em>Pro</em>
        </NavLink>
        <nav className="masthead__nav" aria-label="Main">
          {NAV.map(([to, label]) => (
            <NavLink key={to} to={to} end={to === '/'} className="masthead__link">
              {label}
            </NavLink>
          ))}
        </nav>
        {status && (
          <span className={`status status--compact${status.tone === 'warn' ? ' status--warn' : ''}`}>
            <span className="status__dot" aria-hidden="true" />
            {status.label}
          </span>
        )}
      </header>

      {mode === 'offline' && (
        <p className="banner" role="alert">
          The backend is not running. Start it with <code>uvicorn api:app</code> and reload.
        </p>
      )}

      <main id="main" className="main">
        <ErrorBoundary key={location.pathname}>
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.35, ease: [0.2, 0.8, 0.2, 1] }}
            >
              <Routes location={location}>
                <Route path="/" element={<PracticePage />} />
                <Route path="/history" element={<HistoryPage />} />
                <Route path="/model" element={<ModelInfoPage />} />
                <Route path="*" element={<PracticePage />} />
              </Routes>
            </motion.div>
          </AnimatePresence>
        </ErrorBoundary>
      </main>

      <footer className="colophon">
        <span>wav2vec 2.0 fine-tuned on SpeechOcean762</span>
        <span>Set in Fraunces, Instrument Sans and IBM Plex Mono</span>
      </footer>

      <svg className="grain" aria-hidden="true">
        <filter id="grain-filter">
          <feTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="2" stitchTiles="stitch" />
          <feColorMatrix type="saturate" values="0" />
        </filter>
        <rect width="100%" height="100%" filter="url(#grain-filter)" />
      </svg>
    </div>
  )
}

function App() {
  return (
    <MotionConfig reducedMotion="user">
      <BrowserRouter>
        <Shell />
      </BrowserRouter>
    </MotionConfig>
  )
}

export default App
