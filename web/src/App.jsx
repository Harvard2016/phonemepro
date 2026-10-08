import { useEffect, useState } from 'react'
import { BrowserRouter, NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { AnimatePresence, MotionConfig, motion } from 'motion/react'
import ErrorBoundary from './ErrorBoundary'
import Welcome from './components/Welcome'
import CreditsPage from './pages/CreditsPage'
import DataPage from './pages/DataPage'
import HistoryPage from './pages/HistoryPage'
import ModelInfoPage from './pages/ModelInfoPage'
import PracticePage from './pages/PracticePage'
import PrivacyPage from './pages/PrivacyPage'
import { AppProvider, useApp } from './state'

const NAV = [
  ['/', 'Practice'],
  ['/history', 'History'],
  ['/model', 'Model'],
]

function ModelStatus() {
  const { model } = useApp()
  if (model.status === 'idle') return null
  const label = {
    loading: `Loading model ${Math.round(model.progress * 100)}%`,
    ready: 'Model ready, on this device',
    error: 'Model did not load',
  }[model.status]
  return (
    <span className={`status status--compact${model.status === 'error' ? ' status--warn' : ''}`}>
      <span className="status__dot" aria-hidden="true" />
      {label}
    </span>
  )
}

function Shell() {
  const { openWelcome } = useApp()
  const location = useLocation()
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

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
        <ModelStatus />
      </header>

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
                <Route path="/credits" element={<CreditsPage />} />
                <Route path="/data" element={<DataPage />} />
                <Route path="/privacy" element={<PrivacyPage />} />
                <Route path="*" element={<PracticePage />} />
              </Routes>
            </motion.div>
          </AnimatePresence>
        </ErrorBoundary>
      </main>

      <footer className="colophon">
        <span>Runs in your browser. Your voice never leaves this device.</span>
        <span className="colophon__links">
          <button type="button" className="colophon__button" onClick={openWelcome}>Your profile</button>
          <NavLink to="/data">Your data</NavLink>
          <NavLink to="/privacy">Privacy</NavLink>
          <NavLink to="/credits">Credits, data and voices</NavLink>
        </span>
      </footer>

      <Welcome />

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
        <AppProvider>
          <Shell />
        </AppProvider>
      </BrowserRouter>
    </MotionConfig>
  )
}

export default App
