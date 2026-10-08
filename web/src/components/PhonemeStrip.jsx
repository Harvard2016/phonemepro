import { motion } from 'motion/react'
import { toIpa } from '../lib/phonemes'

const STATUS_LABEL = { correct: 'correct', error: 'substituted', missed: 'missed', unscored: 'not scored' }

// Target phonemes set like a line of type, marked up the way a proofreader would:
// substitutions struck through in red with what was heard written below, omissions left hollow.
// With `onPlay`, a sound that was heard becomes a button that plays its slice of the take.
function PhonemeStrip({ results, animate = true, compact = false, accent = 'ga', onPlay }) {
  return (
    <ol className={`strip${compact ? ' strip--compact' : ''}`}>
      {results.map((result, index) => (
        <motion.li
          key={`${result.phoneme}-${index}`}
          className={`strip__cell strip__cell--${result.status}`}
          initial={animate ? { opacity: 0, y: 14 } : false}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: animate ? 0.15 + index * 0.05 : 0 }}
        >
          <span className="strip__arpabet">{result.phoneme}</span>
          {onPlay && result.start !== undefined ? (
            <button
              type="button"
              className="strip__ipa strip__play ipa"
              onClick={() => onPlay(result.start, result.end)}
              aria-label={`Play how you said ${result.phoneme}`}
            >
              {toIpa(result.phoneme, accent)}
            </button>
          ) : (
            <span className="strip__ipa ipa">{toIpa(result.phoneme, accent)}</span>
          )}
          {!compact && (
            <span className="strip__note">
              {result.status === 'error' && result.heard && (
                <>heard <span className="ipa">{toIpa(result.heard, accent)}</span></>
              )}
              {result.status === 'missed' && 'missed'}
              {result.status === 'unscored' && 'not scored'}
              {result.status === 'correct' && result.stress_match === false && 'stress'}
            </span>
          )}
          <span className="visually-hidden">{STATUS_LABEL[result.status]}</span>
        </motion.li>
      ))}
    </ol>
  )
}

export default PhonemeStrip
