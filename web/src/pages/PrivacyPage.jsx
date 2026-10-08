import { Link } from 'react-router-dom'
import { MAX_STORED } from '../engine/contributions'

// INTERIM. Written by the developer, not reviewed by a lawyer. Before sharing is
// switched on, a lawyer should check at least:
//   - whether the 256-number summary is biometric or otherwise special-category data
//     (GDPR Art. 9, Illinois BIPA, Texas CUBI, Washington MHMDA) and what consent that needs;
//   - the legal basis and the wording, timing and record of consent for "Help it learn";
//   - country plus free-text region or city, which can single out a person in a small place;
//   - children: no age check exists (COPPA, GDPR Art. 8);
//   - who the controller is, the contact address, and how access and erasure requests are met
//     once summaries have been folded into totals and cannot be picked back out;
//   - transfers between countries, retention periods, and whether an impact assessment is needed;
//   - the claims "not audio" and "cannot be played back": true of the format, but the
//     summary may still be characteristic of a voice.
const CONTACT = 'CONTACT ADDRESS TO BE ADDED'

function PrivacyPage() {
  return (
    <div className="page">
      <header className="page__head">
        <p className="label label--accent">Privacy · interim notice</p>
        <h1 className="page__title">What happens to your voice</h1>
        <p className="page__lede">
          This is a temporary notice in plain language. It has not been reviewed by a lawyer and will be
          replaced before anything on this page changes.
        </p>
      </header>

      <p className="notice notice--static">
        The short version: PhonemePro has no server to send your voice to. Recordings are analysed in your
        browser and then dropped. Sharing is not switched on.
      </p>

      <section aria-label="Details">
        <h2 className="section-title">Your recordings</h2>
        <p className="prose">
          When you record a take, the audio stays in this tab so you can play it back, and is gone when you
          move on or close the tab. It is never saved and never uploaded. The model that listens to it runs
          on your device.
        </p>

        <h2 className="section-title">Your practice history</h2>
        <p className="prose">
          Words, scores and the sounds heard are kept in this browser so the History page works. You can
          export or erase them on <Link to="/history">History</Link>.
        </p>

        <h2 className="section-title">If you turn on “Help it learn”</h2>
        <p className="prose">It is off unless you turn it on. When it is on, this browser keeps, for each clear take:</p>
        <ul className="prose">
          <li>a summary of the take as 256 numbers, worked out by the model;</li>
          <li>the labels you gave: your country and, if you typed one, your region or city;</li>
          <li>which accent you were practising, or that you were speaking in your normal voice;</li>
          <li>the word, the score, how clean the take was, and which version of the model made the summary;</li>
          <li>a random id for the take and a random id for this browser. Neither is your name or an account.</li>
        </ul>
        <p className="prose">
          The summary is not audio and cannot be played back. It may still be characteristic of your voice,
          in the way a description of a face is not a photograph but is still about one person. Treat it as
          personal information. At most {MAX_STORED} takes are kept per browser.
        </p>

        <h2 className="section-title">Where it is</h2>
        <p className="prose">
          On this device, in this browser, and nowhere else for now. Nothing is sent to PhonemePro or to
          anyone else. There is no upload. If sharing is ever added it will be a separate choice, asked
          again, with a new notice.
        </p>

        <h2 className="section-title">Seeing and deleting it</h2>
        <p className="prose">
          <Link to="/data">Your data</Link> lists every stored take with all of its labels, lets you save a
          copy, and has a button that deletes everything. “Your profile” at the foot of every page lets you
          change your answers, turn the switch off, or withdraw. Clearing this site’s data in your browser
          removes it too.
        </p>

        <h2 className="section-title">Contact</h2>
        <p className="prose">Questions about this notice: {CONTACT}.</p>
      </section>
    </div>
  )
}

export default PrivacyPage
