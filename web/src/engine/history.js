// Practice history kept in the browser (IndexedDB), so it survives refreshes and
// closed tabs without an account or a server. It stays on this device; export and
// import move it between browsers.

import { run as runIn } from './db'

const EXPORT_VERSION = 1

const run = (mode, work) => runIn('attempts', mode, work)

// Keep only what the History page and insights need; audio is never stored.
export function saveAttempt({ word, accent, target_phonemes, predicted_phonemes, score, wer, errors }) {
  return run('readwrite', (store) => store.add({
    word, accent, target_phonemes, predicted_phonemes, score, wer, errors,
    timestamp: new Date().toISOString(),
  }))
}

// Newest first.
export async function getHistory() {
  const rows = await run('readonly', (store) => store.getAll())
  return rows.sort((a, b) => b.id - a.id)
}

export const clearHistory = () => run('readwrite', (store) => store.clear())

export function countToday(attempts, now = new Date()) {
  const today = now.toDateString()
  return attempts.filter((attempt) => new Date(attempt.timestamp).toDateString() === today).length
}

export function exportHistory(attempts) {
  return JSON.stringify({ app: 'phonemepro', version: EXPORT_VERSION, attempts }, null, 1)
}

const isAttempt = (value) =>
  value && typeof value.word === 'string' && typeof value.score === 'number' && typeof value.timestamp === 'string'
  && Array.isArray(value.target_phonemes) && Array.isArray(value.predicted_phonemes) && Array.isArray(value.errors)

// Adds attempts from an export file, skipping ones already present. Returns how many were added.
export async function importHistory(text) {
  let parsed
  try {
    parsed = JSON.parse(text)
  } catch {
    throw new Error('That file is not a PhonemePro history export.')
  }
  if (parsed?.app !== 'phonemepro' || !Array.isArray(parsed.attempts) || !parsed.attempts.every(isAttempt)) {
    throw new Error('That file is not a PhonemePro history export.')
  }

  const existing = new Set((await getHistory()).map((attempt) => `${attempt.timestamp}|${attempt.word}`))
  const fresh = parsed.attempts.filter((attempt) => !existing.has(`${attempt.timestamp}|${attempt.word}`))
  await run('readwrite', (store) => {
    // Oldest first so ids keep the original order.
    for (const attempt of [...fresh].sort((a, b) => a.timestamp.localeCompare(b.timestamp))) {
      const { id: _id, ...row } = attempt
      store.add(row)
    }
  })
  return fresh.length
}

export function toCsv(attempts) {
  const quote = (value) => `"${String(value).replace(/"/g, '""')}"`
  const header = ['word', 'accent', 'score', 'wer', 'target_phonemes', 'predicted_phonemes', 'errors', 'timestamp']
  const rows = attempts.map((a) => [
    a.word, a.accent ?? 'ga', a.score, a.wer, a.target_phonemes.join(' '), a.predicted_phonemes.join(' '),
    a.errors.length, a.timestamp,
  ])
  return [header, ...rows].map((row) => row.map(quote).join(',')).join('\n')
}
