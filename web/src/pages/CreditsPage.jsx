import { useApp } from '../state'
import { LoadingState } from '../components/States'

const TEAM = ['Arshpreet Singh Sandhu', 'Prakhar Verma', 'Abhisar Anand', 'Harshit Sharma']

const SOURCES = [
  ['SpeechOcean762', 'Learner speech with human pronunciation ratings. Trains and tests the phoneme model and the score head.', 'CC BY 4.0', 'https://www.openslr.org/101/'],
  ['VCTK Corpus 0.92', 'Native speakers with labelled accents. Trains and tests the accent head.', 'CC BY 4.0', 'https://datashare.ed.ac.uk/handle/10283/3443'],
  ['wav2vec 2.0 base', 'The pretrained speech encoder the model is fine-tuned from.', 'Apache 2.0', 'https://huggingface.co/facebook/wav2vec2-base'],
  ['CMUdict', 'American pronunciations.', 'BSD-style', 'http://www.speech.cs.cmu.edu/cgi-bin/cmudict'],
  ['Piper voices', 'The synthetic voices for text with no human recording: Joe (American, CC0), Cori (British, public domain) and VCTK speaker p326 (Australian, CC BY 4.0).', 'per voice', 'https://huggingface.co/rhasspy/piper-voices'],
  ['eSpeak NG', 'Turns typed text into the sounds the synthetic voices read, through the phonemizer package.', 'GPL 3.0', 'https://github.com/espeak-ng/espeak-ng'],
  ['Britfone 3.0.1', 'British pronunciations. Australian entries are derived from these by rule.', 'MIT', 'https://github.com/JoseLlarena/Britfone'],
]

function CreditsPage() {
  const { lexicon } = useApp()
  if (!lexicon) return <LoadingState label="Gathering credits" />

  const recordings = Object.entries(lexicon.credits).flatMap(([accent, words]) =>
    Object.entries(words).map(([word, credit]) => ({ accent, word, ...credit })))

  return (
    <div className="page">
      <header className="page__head">
        <p className="label">Credits</p>
        <h1 className="page__title">Who made this</h1>
        <p className="page__lede">
          PhonemePro began as a team project by {TEAM.slice(0, -1).join(', ')} and {TEAM[TEAM.length - 1]}.
          Harshit carried it on from there: the accent work, the in-browser model and this interface.
        </p>
      </header>

      <section aria-label="Data and models">
        <h2 className="section-title">Data and models</h2>
        <dl className="spec spec--wide">
          {SOURCES.map(([name, use, license, url]) => (
            <div key={name}>
              <dt><a href={url} target="_blank" rel="noreferrer">{name}</a> · {license}</dt>
              <dd>{use}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section aria-label="Voices">
        <h2 className="section-title">Voices</h2>
        <p className="prose">
          The {recordings.length} human reference recordings come from volunteers on Wikimedia Commons. Each was
          converted to MP3, trimmed and loudness-normalized; the originals and their licences are linked below.
          Recordings under a ShareAlike licence remain under that licence.
        </p>
        <ul className="credits">
          {recordings.map((item) => (
            <li key={`${item.accent}-${item.word}`}>
              <a href={item.page} target="_blank" rel="noreferrer">{item.word}</a>
              <span className="tag">{lexicon.accents[item.accent]?.name ?? item.accent}</span>
              <span className="credits__by">
                {item.author} · {item.license_url
                  ? <a href={item.license_url} target="_blank" rel="noreferrer">{item.license}</a>
                  : item.license}
              </span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}

export default CreditsPage
