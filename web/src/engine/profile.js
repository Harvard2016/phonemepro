// What a visitor chose to tell the app about themselves. Everything is optional,
// kept in this browser (localStorage), and can be changed or withdrawn at any time.
import { isCountryCode } from '../lib/countries'

const KEY = 'phonemepro.profile'
export const MAX_REGION_CHARS = 60

export const EMPTY_PROFILE = {
  seen: false, // the welcome screen has been shown and closed
  country: null, // ISO 3166-1 alpha-2, or null
  region: '', // free text: a region or city
  contribute: false, // "Help it learn". Off unless the visitor turns it on.
  contributor: null, // random id for this browser, made when contributing is first turned on
}

// Collapse whitespace and drop control characters; the field is free text.
export const cleanRegion = (text) =>
  // eslint-disable-next-line no-control-regex
  String(text ?? '').replace(/[\u0000-\u001f\u007f]/g, ' ').replace(/\s+/g, ' ').trim().slice(0, MAX_REGION_CHARS)

// Coerce anything read from storage into a well-formed profile.
export function normalizeProfile(value) {
  const source = value && typeof value === 'object' ? value : {}
  const contribute = source.contribute === true
  const contributor = typeof source.contributor === 'string' ? source.contributor : null
  return {
    seen: source.seen === true,
    country: typeof source.country === 'string' && isCountryCode(source.country) ? source.country : null,
    region: cleanRegion(source.region),
    contribute,
    contributor: contribute ? contributor ?? crypto.randomUUID() : contributor,
  }
}

export function loadProfile() {
  try {
    return normalizeProfile(JSON.parse(localStorage.getItem(KEY)))
  } catch {
    return { ...EMPTY_PROFILE }
  }
}

export function storeProfile(profile) {
  const normalized = normalizeProfile(profile)
  try {
    localStorage.setItem(KEY, JSON.stringify(normalized))
  } catch {
    // Private browsing: the profile lasts only until the tab closes.
  }
  return normalized
}
