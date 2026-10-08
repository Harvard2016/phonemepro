import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { DEFAULT_ACCENT, loadLexicon, practiceWords } from './engine/lexicon'
import { loadModel } from './engine/model'
import { loadProfile, storeProfile } from './engine/profile'

const ACCENT_KEY = 'phonemepro.accent'
const AppContext = createContext(null)

const storedAccent = () => {
  try {
    return localStorage.getItem(ACCENT_KEY)
  } catch {
    return null
  }
}

// Shared app state: the chosen accent, the dictionaries, the in-browser model, and the visitor's profile.
export function AppProvider({ children }) {
  const [accent, setAccentState] = useState(() => storedAccent() ?? DEFAULT_ACCENT)
  const [lexicon, setLexicon] = useState(null)
  const [lexiconError, setLexiconError] = useState(null)
  const [model, setModel] = useState({ status: 'idle', progress: 0, error: null })
  const [profile, setProfile] = useState(loadProfile)
  // The welcome screen opens by itself once, on the first visit.
  // Not over the pilot instructions, which open it themselves at the right step.
  const [welcomeOpen, setWelcomeOpen] = useState(() => !loadProfile().seen && globalThis.location?.pathname !== '/pilot')

  const requestLexicon = useCallback(() => {
    loadLexicon().then(setLexicon).catch(() => setLexiconError('The dictionary did not load. Check your connection.'))
  }, [])

  useEffect(requestLexicon, [requestLexicon])

  // Retry after a failed load.
  const fetchLexicon = useCallback(() => {
    setLexiconError(null)
    requestLexicon()
  }, [requestLexicon])

  const setAccent = useCallback((next) => {
    setAccentState(next)
    try {
      localStorage.setItem(ACCENT_KEY, next)
    } catch {
      // Private browsing: the choice just will not be remembered.
    }
  }, [])

  // Starts the one-time model download; safe to call repeatedly.
  const ensureModel = useCallback(() => {
    setModel((current) => (current.status === 'ready' ? current : { status: 'loading', progress: current.progress, error: null }))
    return loadModel((progress) => setModel((current) => (current.status === 'loading' ? { ...current, progress } : current)))
      .then((loaded) => {
        setModel({ status: 'ready', progress: 1, error: null })
        return loaded
      })
      .catch((error) => {
        setModel({ status: 'error', progress: 0, error: error.message })
        throw error
      })
  }, [])

  // Returns the profile as stored, for callers that act on it straight away.
  const saveProfile = useCallback((next) => {
    const stored = storeProfile(next)
    setProfile(stored)
    return stored
  }, [])
  const openWelcome = useCallback(() => setWelcomeOpen(true), [])
  const closeWelcome = useCallback(() => setWelcomeOpen(false), [])

  const activeAccent = lexicon && !lexicon.accents[accent] ? DEFAULT_ACCENT : accent
  const words = useMemo(() => (lexicon ? practiceWords(lexicon, activeAccent) : null), [lexicon, activeAccent])

  const value = useMemo(
    () => ({
      accent: activeAccent, setAccent, lexicon, lexiconError, fetchLexicon, words, model, ensureModel,
      profile, saveProfile, welcomeOpen, openWelcome, closeWelcome,
    }),
    [
      activeAccent, setAccent, lexicon, lexiconError, fetchLexicon, words, model, ensureModel,
      profile, saveProfile, welcomeOpen, openWelcome, closeWelcome,
    ],
  )
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export const useApp = () => useContext(AppContext)
