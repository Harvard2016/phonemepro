import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { DEFAULT_ACCENT, loadLexicon, practiceWords } from './engine/lexicon'
import { loadModel } from './engine/model'

const ACCENT_KEY = 'phonemepro.accent'
const AppContext = createContext(null)

const storedAccent = () => {
  try {
    return localStorage.getItem(ACCENT_KEY)
  } catch {
    return null
  }
}

// Shared app state: the chosen accent, the dictionaries, and the in-browser model.
export function AppProvider({ children }) {
  const [accent, setAccentState] = useState(() => storedAccent() ?? DEFAULT_ACCENT)
  const [lexicon, setLexicon] = useState(null)
  const [lexiconError, setLexiconError] = useState(null)
  const [model, setModel] = useState({ status: 'idle', progress: 0, error: null })

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

  const activeAccent = lexicon && !lexicon.accents[accent] ? DEFAULT_ACCENT : accent
  const words = useMemo(() => (lexicon ? practiceWords(lexicon, activeAccent) : null), [lexicon, activeAccent])

  const value = useMemo(
    () => ({ accent: activeAccent, setAccent, lexicon, lexiconError, fetchLexicon, words, model, ensureModel }),
    [activeAccent, setAccent, lexicon, lexiconError, fetchLexicon, words, model, ensureModel],
  )
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export const useApp = () => useContext(AppContext)
