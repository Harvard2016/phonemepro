import { describe, expect, it } from 'vitest'
import { loadVoices, pickVoice } from './voices'

const voice = (name, lang, localService = true) => ({ name, lang, localService })

// English voices as Chrome lists them on a Mac, in its order.
const MAC = [
  voice('Albert', 'en-US'), voice('Bad News', 'en-US'), voice('Bells', 'en-US'), voice('Daniel', 'en-GB'),
  voice('Eddy (English (United Kingdom))', 'en-GB'), voice('Eddy (English (United States))', 'en-US'),
  voice('Fred', 'en-US'), voice('Karen', 'en-AU'), voice('Kathy', 'en-US'), voice('Rishi', 'en-IN'),
  voice('Samantha', 'en-US'), voice('Zarvox', 'en-US'),
  voice('Google US English', 'en-US', false), voice('Google UK English Female', 'en-GB', false),
]

describe('pickVoice', () => {
  it('skips novelty voices that come first in the list', () => {
    expect(pickVoice(MAC, 'ga').name).toBe('Samantha')
    expect(pickVoice(MAC, 'rp').name).toBe('Daniel')
    expect(pickVoice(MAC, 'au').name).toBe('Karen')
  })

  it('never borrows a voice from another accent', () => {
    const noAustralian = MAC.filter((v) => v.lang !== 'en-AU')
    expect(pickVoice(noAustralian, 'au')).toBeNull()
    expect(pickVoice([], 'rp')).toBeNull()
  })

  it('returns null when only unsuitable voices match', () => {
    expect(pickVoice([voice('Albert', 'en-US'), voice('Eddy (English (United States))', 'en-US')], 'ga')).toBeNull()
  })

  it('prefers a voice on the device to one that sends text to a server', () => {
    const voices = [voice('Google UK English Female', 'en-GB', false), voice('Some Local Voice', 'en-GB')]
    expect(pickVoice(voices, 'rp').name).toBe('Some Local Voice')
    expect(pickVoice(voices.slice(0, 1), 'rp').name).toBe('Google UK English Female')
  })

  it('accepts underscore locales and unknown but ordinary voices', () => {
    expect(pickVoice([voice('English Australia', 'en_AU')], 'au').name).toBe('English Australia')
  })
})

describe('loadVoices', () => {
  it('waits for the list when the first call is empty', async () => {
    let listener
    let voices = []
    const synth = {
      getVoices: () => voices,
      addEventListener: (_, fn) => { listener = fn },
      removeEventListener: () => {},
    }
    const pending = loadVoices(synth, 5000)
    voices = MAC
    listener()
    expect(await pending).toHaveLength(MAC.length)
  })

  it('gives up after the timeout', async () => {
    const synth = { getVoices: () => [], addEventListener: () => {}, removeEventListener: () => {} }
    expect(await loadVoices(synth, 10)).toEqual([])
  })
})
