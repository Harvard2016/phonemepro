// The guided label check (pages/CheckPage.jsx, development only).
//
// Each step sets up its own profile, tells the tester what to say, runs the take through
// the same cleanup, model, scoring and keeping rules as the app, and compares what
// happened with what should have happened. Everything expected is in STEPS below.
import { exportContributions, EXPORT_VERSION, HOME_ACCENT, routeOf } from '../engine/contributions'
import { SENTENCES } from '../engine/naturalVoice'
import { isCountryCode } from '../lib/countries'
import { isRegion, regionName } from '../lib/regions'

const SHARING_OFF = '"Help it learn" is off'

// kind: 'take' records and is judged, 'info' records and only reports, 'export' checks what was kept.
// `expect` keys are compared with the outcome of the take; see `LABELS` for what each means.
export const STEPS = [
  {
    id: 1, kind: 'take', title: 'From Britain, practising American',
    profile: { country: 'GB', region: null, contribute: true }, accent: 'ga', word: 'water',
    say: 'Say “water”', how: 'Once, clearly, at your normal volume.',
    expect: {
      kept: true, kind: 'attempt', joins: ['GB>ga'], natural_voice: false, matches_home_accent: false,
      country: 'GB', region: null, target_accent: 'ga',
    },
  },
  {
    id: 2, kind: 'take', title: 'From Britain, practising British',
    profile: { country: 'GB', region: null, contribute: true }, accent: 'rp', word: 'water',
    say: 'Say “water”', how: 'Once, clearly. Practising your own country’s standard accent is still an attempt.',
    expect: {
      kept: true, kind: 'attempt', joins: ['GB>rp'], natural_voice: false, matches_home_accent: true,
      country: 'GB', region: null, target_accent: 'rp',
    },
  },
  {
    id: 3, kind: 'take', title: 'From the US South, practising American',
    profile: { country: 'US', region: 'S', contribute: true }, accent: 'ga', word: 'better',
    say: 'Say “better”', how: 'Once, clearly. It must not join the US or US-S native totals.',
    expect: {
      kept: true, kind: 'attempt', joins: ['US>ga'], natural_voice: false, matches_home_accent: true,
      country: 'US', region: 'S', target_accent: 'ga',
    },
  },
  {
    id: 4, kind: 'take', title: 'From Scotland, in a normal voice', naturalVoice: true,
    profile: { country: 'GB', region: 'SCT', contribute: true }, accent: null, word: SENTENCES[0],
    say: `Read: “${SENTENCES[0][0].toUpperCase()}${SENTENCES[0].slice(1)}.”`, how: 'In your own voice, not an accent you are practising.',
    expect: {
      kept: true, kind: 'native', joins: ['GB', 'GB-SCT'], natural_voice: true, matches_home_accent: false,
      country: 'GB', region: 'SCT', target_accent: null,
    },
  },
  {
    id: 5, kind: 'take', title: 'Silence',
    profile: { country: 'GB', region: null, contribute: true }, accent: 'ga', word: 'water',
    say: 'Stay silent', how: 'Hold space for two seconds and say nothing.',
    expect: { kept: false, reason: 'no speech', retake_notice: true },
  },
  {
    id: 6, kind: 'take', title: 'The wrong word',
    profile: { country: 'GB', region: null, contribute: true }, accent: 'ga', word: 'water',
    say: 'Say “banana”', how: 'The target word is “water”. Say “banana” instead, clearly.',
    expect: { kept: false, failed_includes: 'matched' },
  },
  {
    id: 7, kind: 'take', title: 'A soft first sound',
    profile: { country: 'GB', region: null, contribute: true }, accent: 'ga', word: 'fish',
    say: 'Say “fish”', how: 'Start speaking the moment you press space. The first sound must survive.',
    expect: { first_sound: 'F' },
  },
  {
    id: 8, kind: 'info', title: 'A noisy room',
    profile: { country: 'GB', region: null, contribute: true }, accent: 'ga', word: 'water',
    say: 'Say “water” over noise', how: 'Play music or run a fan nearby, then say it. Nothing here passes or fails.',
    report: ['snr_db', 'kept', 'reason', 'retake_notice'],
  },
  {
    id: 9, kind: 'take', title: 'Sharing switched off',
    profile: { country: 'GB', region: null, contribute: false }, accent: 'ga', word: 'water',
    say: 'Say “water”', how: 'Once, clearly. It should be scored and nothing should be kept.',
    expect: { scored: true, kept: false, reason: SHARING_OFF, stored_added: 0 },
  },
  {
    id: 10, kind: 'export', title: 'What an export would contain',
    say: 'No recording', how: 'Checks everything kept during this session against the ingest rules.',
  },
]

export const LABELS = {
  kept: 'kept',
  kind: 'counts as',
  joins: 'joins totals',
  natural_voice: 'natural_voice',
  matches_home_accent: 'matches_home_accent',
  country: 'country',
  region: 'region',
  target_accent: 'target_accent',
  reason: 'reason not kept',
  retake_notice: 'retake notice shown',
  failed_includes: 'failed check includes',
  first_sound: 'first sound heard',
  scored: 'scored',
  stored_added: 'takes added to the store',
  snr_db: 'signal-to-noise (dB)',
}

const show = (value) => {
  if (value === null || value === undefined) return 'none'
  if (Array.isArray(value)) return value.length ? value.join(', ') : 'none'
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  return String(value)
}

const same = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null)

// A short description of the profile a step runs under.
export function describeSetup(step, accentNames = {}) {
  if (!step.profile) return 'Uses what the earlier steps kept'
  const { country, region, contribute } = step.profile
  return [
    `Profile ${country}${region ? ` · ${regionName(country, region)}` : ''}`,
    `sharing ${contribute ? 'on' : 'off'}`,
    step.naturalVoice ? 'normal voice' : `target ${accentNames[step.accent] ?? step.accent}`,
  ].join(' · ')
}

// Compare what happened with what the table expects.
// Returns { status: 'pass' | 'fail' | 'info', rows: [{ label, expected, actual, pass }] }.
export function evaluateStep(step, outcome) {
  if (step.kind === 'info') {
    return {
      status: 'info',
      rows: step.report.map((key) => ({ label: LABELS[key], expected: 'not judged', actual: show(outcome[key]), pass: null })),
    }
  }
  const rows = Object.entries(step.expect).map(([key, expected]) => {
    if (key === 'failed_includes') {
      const failed = outcome.failed_checks ?? []
      return { label: LABELS[key], expected, actual: show(failed), pass: failed.includes(expected) }
    }
    return { label: LABELS[key], expected: show(expected), actual: show(outcome[key]), pass: same(outcome[key], expected) }
  })
  return { status: rows.every((row) => row.pass) ? 'pass' : 'fail', rows }
}

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
const FEATURES = 256
const MAX_ABS_FEATURE = 60
const AUDIO_KEYS = /samples|wav|audio|blob|recording/i

// The reason ingest would refuse a take, or null. Mirrors `check_take` in scripts/learn_regions.py.
export function refusalOf(take, modelVersion) {
  if (!take || typeof take !== 'object' || Array.isArray(take)) return 'not a take'
  if (typeof take.id !== 'string' || !UUID.test(take.id)) return 'bad take id'
  if (typeof take.contributor !== 'string' || !UUID.test(take.contributor)) return 'bad contributor id'
  if (typeof take.country !== 'string' || !isCountryCode(take.country)) return 'bad country'
  if (take.region !== null && !isRegion(take.country, take.region)) return 'bad region'
  if (typeof take.natural_voice !== 'boolean' || typeof take.matches_home_accent !== 'boolean') return 'bad labels'
  if (take.natural_voice ? take.target_accent !== null : !['ga', 'rp', 'au'].includes(take.target_accent)) return 'bad target accent'
  if (take.matches_home_accent !== (!take.natural_voice && HOME_ACCENT[take.country] === take.target_accent)) return 'labels disagree'
  if (take.model_version !== modelVersion) return 'made with a different model version'
  if (!Array.isArray(take.features) || take.features.length !== FEATURES) return 'wrong feature length'
  if (!take.features.every((value) => typeof value === 'number' && Number.isFinite(value))) return 'non-finite feature'
  if (take.features.some((value) => Math.abs(value) > MAX_ABS_FEATURE)) return 'feature out of range'
  return null
}

// Step 10. Export everything the session kept, read it back as ingest would, and confirm each
// take from steps 1 to 4 is routed the way the table predicted.
// `kept` maps a step id to the id of the take that step stored.
export function checkExport(takes, kept, modelVersion) {
  const parsed = JSON.parse(exportContributions(takes))
  const exported = Array.isArray(parsed.takes) ? parsed.takes : []
  const refusals = exported.map((take) => refusalOf(take, modelVersion)).filter(Boolean)
  const audioKeys = exported.flatMap((take) => [...Object.keys(take), ...Object.keys(take.quality ?? {})]).filter((key) => AUDIO_KEYS.test(key))

  const rows = [
    {
      label: 'export format', expected: `phonemepro contributions v${EXPORT_VERSION}`,
      actual: `${parsed.app} ${parsed.kind} v${parsed.version}`,
      pass: parsed.app === 'phonemepro' && parsed.kind === 'contributions' && parsed.version === EXPORT_VERSION,
    },
    { label: 'takes exported', expected: String(takes.length), actual: String(exported.length), pass: exported.length === takes.length },
    { label: 'takes ingest would refuse', expected: 'none', actual: show(refusals), pass: refusals.length === 0 },
    { label: 'audio in the export', expected: 'none', actual: show(audioKeys), pass: audioKeys.length === 0 },
  ]
  const routed = STEPS.filter((step) => step.expect?.joins)
  for (const step of routed) {
    const take = exported.find((candidate) => candidate.id === kept[step.id])
    rows.push({
      label: `step ${step.id} joins`,
      expected: show(step.expect.joins),
      actual: take ? show(routeOf(take).keys) : 'not kept in this session',
      pass: take ? same(routeOf(take).keys, step.expect.joins) && routeOf(take).kind === step.expect.kind : null,
    })
  }
  const judged = rows.slice(4).filter((row) => row.pass !== null)
  const status = judged.length === 0 ? 'skipped' : rows.every((row) => row.pass !== false) ? 'pass' : 'fail'
  return { status, rows }
}

// Plain text of the whole run, to paste somewhere else.
export function formatResults(results, { modelVersion = 'unknown', when = new Date() } = {}) {
  const lines = [`PhonemePro label check · ${when.toISOString()} · model ${modelVersion}`, '']
  for (const step of STEPS) {
    const result = results[step.id]
    const status = result ? result.evaluation.status.toUpperCase() : 'SKIPPED'
    lines.push(`${String(step.id).padStart(2)}. ${status.padEnd(7)} ${step.title}`)
    if (!result) continue
    for (const row of result.evaluation.rows) {
      const mark = row.pass === null ? ' ' : row.pass ? '✓' : '✕'
      lines.push(`      ${mark} ${row.label}: expected ${row.expected}, got ${row.actual}`)
    }
    if (result.outcome?.detail) lines.push(`        ${result.outcome.detail}`)
  }
  const counts = STEPS.reduce((tally, step) => {
    const status = results[step.id]?.evaluation.status ?? 'skipped'
    return { ...tally, [status]: (tally[status] ?? 0) + 1 }
  }, {})
  lines.push('', `Passed ${counts.pass ?? 0}, failed ${counts.fail ?? 0}, informational ${counts.info ?? 0}, skipped ${counts.skipped ?? 0}.`)
  return lines.join('\n')
}
