// What a visitor chose to tell the app about themselves. Everything is optional,
// kept in this browser (localStorage), and can be changed or withdrawn at any time.
import { isCountryCode } from '../lib/countries'
import { isRegion } from '../lib/regions'

const KEY = 'phonemepro.profile'

export const EMPTY_PROFILE = {
  seen: false, // the welcome screen has been shown and closed
  country: null, // ISO 3166-1 alpha-2, or null
  region: null, // a code from lib/regions.js for the chosen country, or null. Never typed text.
  contribute: false, // "Help it learn". Off unless the visitor turns it on.
  contributor: null, // random id for this browser, made when contributing is first turned on
}

// Coerce anything read from storage into a well-formed profile. A region that is not
// one of the country's fixed codes is dropped, which also clears text typed into the
// free-text field this replaced.
export function normalizeProfile(value) {
  const source = value && typeof value === 'object' ? value : {}
  const contribute = source.contribute === true
  const contributor = typeof source.contributor === 'string' ? source.contributor : null
  const country = typeof source.country === 'string' && isCountryCode(source.country) ? source.country : null
  return {
    seen: source.seen === true,
    country,
    region: isRegion(country, source.region) ? source.region : null,
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
