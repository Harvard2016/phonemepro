// Whether a heard sound is the same broad class as its target: any vowel for a vowel, or
// a consonant from the same accent-swap group ("tink" for "think").
//
// Mirrors pronunciation/sound_match.py; both read sound_groups.json and are pinned to
// tests/fixtures/sound_match_cases.json. This share is recorded with each kept take but
// does not decide whether a take is kept: measured on accented speech and on wrong words
// (Docs/model/match_threshold.json), it let through far more wrong words than the exact
// share did for the same number of genuine takes.
import { align, roundHalfEven, stripStress, UNSCORABLE_PHONEMES } from './scoring'
import groups from './sound_groups.json'

const VOWELS = new Set(groups.vowels)
const PARTNERS = new Map()
for (const group of groups.groups) {
  for (const sound of group) {
    if (!PARTNERS.has(sound)) PARTNERS.set(sound, new Set())
    group.forEach((partner) => PARTNERS.get(sound).add(partner))
  }
}

// True when `heard` is an acceptable stand-in for `target` (stress ignored).
export function sameClass(target, heard) {
  const wanted = stripStress(target)
  const got = stripStress(heard)
  if (wanted === got) return true
  if (VOWELS.has(wanted) && VOWELS.has(got)) return true
  return PARTNERS.get(wanted)?.has(got) ?? false
}

// Percentage of target sounds heard as written or as a sound of the same class.
export function soundsSameClass(target, heard) {
  if (target.length === 0) return 0
  const ref = target.map(stripStress)
  const hyp = heard.map(stripStress)
  let attempted = 0
  for (const op of align(ref, hyp).ops) {
    if (op.kind === 'insert') continue
    // The model cannot hear these, so they are given the benefit of the doubt.
    if (UNSCORABLE_PHONEMES.has(ref[op.refIndex])) attempted += 1
    else if (op.kind !== 'delete' && sameClass(ref[op.refIndex], hyp[op.hypIndex])) attempted += 1
  }
  return roundHalfEven((attempted * 100) / ref.length)
}
