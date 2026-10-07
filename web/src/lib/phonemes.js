// ARPABET helpers and the IPA transcription shown next to each headword.

const IPA = {
  AA: 'ɑ', AE: 'æ', AH: 'ʌ', AO: 'ɔ', AW: 'aʊ', AY: 'aɪ', EH: 'ɛ', ER: 'ɝ', EY: 'eɪ',
  IH: 'ɪ', IY: 'i', OW: 'oʊ', OY: 'ɔɪ', UH: 'ʊ', UW: 'u',
  B: 'b', CH: 'tʃ', D: 'd', DH: 'ð', F: 'f', G: 'ɡ', HH: 'h', JH: 'dʒ', K: 'k', L: 'l',
  M: 'm', N: 'n', NG: 'ŋ', P: 'p', R: 'ɹ', S: 's', SH: 'ʃ', T: 't', TH: 'θ', V: 'v',
  W: 'w', Y: 'j', Z: 'z', ZH: 'ʒ',
}

// Unstressed AH and ER reduce to schwa forms.
const REDUCED = { AH: 'ə', ER: 'ɚ' }

// Consonant clusters that can begin an English syllable, used to place stress marks.
const ONSETS = new Set([
  'P L', 'P R', 'B L', 'B R', 'T R', 'D R', 'K L', 'K R', 'K W', 'G L', 'G R', 'F L', 'F R',
  'TH R', 'SH R', 'S P', 'S T', 'S K', 'S M', 'S N', 'S L', 'S W', 'T W', 'D W', 'K Y', 'M Y',
  'B Y', 'P Y', 'F Y', 'HH Y', 'V Y', 'S P R', 'S P L', 'S T R', 'S K R', 'S K W', 'S K Y',
])

export const basePhoneme = (phoneme) => phoneme.replace(/[0-2]$/, '')

export const stressOf = (phoneme) => {
  const match = phoneme.match(/[0-2]$/)
  return match ? match[0] : null
}

export const isVowel = (phoneme) => stressOf(phoneme) !== null

export function toIpa(phoneme) {
  const base = basePhoneme(phoneme)
  if (stressOf(phoneme) === '0' && REDUCED[base]) return REDUCED[base]
  return IPA[base] ?? base.toLowerCase()
}

// Index at which the syllable containing the vowel at `vowelIndex` starts.
function syllableStart(phonemes, vowelIndex) {
  let runStart = vowelIndex
  while (runStart > 0 && !isVowel(phonemes[runStart - 1])) runStart -= 1
  if (runStart === 0) return 0

  const run = phonemes.slice(runStart, vowelIndex).map(basePhoneme)
  for (let take = run.length; take > 1; take -= 1) {
    if (ONSETS.has(run.slice(run.length - take).join(' '))) return vowelIndex - take
  }
  // A single consonant starts the syllable, except NG which never does.
  if (run.length > 0 && run[run.length - 1] !== 'NG') return vowelIndex - 1
  return vowelIndex
}

export function wordToIpa(phonemes) {
  const vowelCount = phonemes.filter(isVowel).length
  const marks = {}
  if (vowelCount > 1) {
    phonemes.forEach((phoneme, index) => {
      const stress = stressOf(phoneme)
      if (stress === '1') marks[syllableStart(phonemes, index)] = 'ˈ'
      if (stress === '2') marks[syllableStart(phonemes, index)] = 'ˌ'
    })
  }
  return phonemes.map((phoneme, index) => (marks[index] ?? '') + toIpa(phoneme)).join('')
}
