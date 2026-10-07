import { API_BASE } from '../config'

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

const OFFLINE_MESSAGE = 'The backend is not reachable. Start it, then try again.'

async function request(path, options) {
  let response
  try {
    response = await fetch(`${API_BASE}${path}`, options)
  } catch {
    throw new ApiError(OFFLINE_MESSAGE, 0)
  }

  if (!response.ok) {
    let detail = null
    try {
      detail = (await response.json()).detail
    } catch {
      // Non-JSON error body: fall through to the generic message.
    }
    throw new ApiError(typeof detail === 'string' ? detail : 'Something went wrong. Try again.', response.status)
  }
  return response.json()
}

export const getWords = () => request('/api/words')
export const getPhonemes = (text) => request(`/api/phonemes?text=${encodeURIComponent(text)}`)
export const getTodayCount = () => request('/api/history/today-count')
export const getHistory = (limit = 200) => request(`/api/history?limit=${limit}`)
export const getInsights = () => request('/api/insights')
export const getModelInfo = () => request('/api/model-info')
export const clearRealHistory = () => request('/api/history?source=real', { method: 'DELETE' })
export const historyExportUrl = `${API_BASE}/api/history/export`

export function analyze(wavBlob, word) {
  const form = new FormData()
  form.append('audio', wavBlob, 'recording.wav')
  form.append('word', word.word)
  form.append('phonemes', JSON.stringify(word.phonemes))
  return request('/api/analyze', { method: 'POST', body: form })
}
