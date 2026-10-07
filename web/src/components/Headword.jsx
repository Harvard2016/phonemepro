import { motion } from 'motion/react'
import { wordToIpa } from '../lib/phonemes'

const EASE = [0.2, 0.8, 0.2, 1]

// The dictionary entry: an oversized headword that rises letter by letter, with its IPA beneath.
function Headword({ word }) {
  const parts = word.word.split(' ')
  const isPhrase = parts.length > 1
  const letters = [...word.word]
  const longest = Math.max(...parts.map((part) => part.length), 4)
  let position = 0

  return (
    <div className={`headword${isPhrase ? ' headword--phrase' : ''}`} style={{ '--len': longest }}>
      <h1 className="headword__word" aria-label={word.word}>
        {parts.map((part, partIndex) => (
          <span key={`${word.word}-${partIndex}`} className="headword__part">
            {[...part].map((letter) => {
              const index = position
              position += 1
              return (
                <span key={index} className="headword__clip" aria-hidden="true">
                  <motion.span
                    className="headword__char"
                    initial={{ y: '110%' }}
                    animate={{ y: 0 }}
                    transition={{ duration: 0.7, delay: index * (isPhrase ? 0.018 : 0.035), ease: EASE }}
                  >
                    {letter}
                  </motion.span>
                </span>
              )
            })}
          </span>
        ))}
      </h1>
      <motion.p
        key={word.word}
        className="headword__ipa"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.25 + Math.min(letters.length, 14) * 0.035 }}
      >
        <span className="ipa">/{wordToIpa(word.phonemes)}/</span>
        <span className="headword__arpabet">{word.phonemes.join(' ')}</span>
      </motion.p>
    </div>
  )
}

export default Headword
