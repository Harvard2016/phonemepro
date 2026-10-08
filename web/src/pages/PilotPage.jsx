import { Link } from 'react-router-dom'
import { SENTENCES } from '../engine/naturalVoice'
import { useApp } from '../state'

// For people who were asked to take part in the pilot. It is not linked from the footer:
// the person running the pilot sends this address to those they invite.
//
// The app still has no upload. Taking part means saving a file from "Your data" and sending
// it yourself, so nothing leaves the device unless the visitor does it by hand.
function PilotPage() {
  const { openWelcome } = useApp()
  return (
    <div className="page">
      <header className="page__head">
        <p className="label label--accent">Pilot · by invitation</p>
        <h1 className="page__title">Help it learn accents</h1>
        <p className="page__lede">
          You were asked to take part in a small test. It takes about five minutes. Nothing is sent
          automatically: at the end you save one file and send it yourself, or decide not to.
        </p>
      </header>

      <section aria-label="What to do">
        <h2 className="section-title">What to do</h2>
        <ol className="prose pilot__steps">
          <li>
            <button type="button" className="text-button" onClick={openWelcome}>Open your profile</button>.
            Choose your country, and a region if one is offered. Turn on “Help it learn”.
          </li>
          <li>
            On the same screen, read the {SENTENCES.length} sentences in your normal voice, not an accent you
            are practising. These are the takes that matter most. Then save.
          </li>
          <li>
            On <Link to="/">Practice</Link>, pick any accent and say ten or so words. Use a quiet room.
          </li>
          <li>
            Open <Link to="/data">Your data</Link>. Look through what was kept, then press “Export to a JSON file”.
          </li>
          <li>Send that file to the person who invited you, the same way they sent you this page.</li>
        </ol>
      </section>

      <section aria-label="What you are sending">
        <h2 className="section-title">What is in the file</h2>
        <p className="prose">
          For each clear take: a summary of 256 numbers worked out by the model, your country and region,
          which accent you were practising or that it was your normal voice, the word, the score, and a random
          id for your browser. No audio, no name, no email. The summary cannot be played back, but it comes
          from your voice and may still be characteristic of it, so treat it as personal information.
          Your data shows every value before you send anything.
        </p>

        <h2 className="section-title">What happens to it</h2>
        <p className="prose">
          The file is kept privately by the person running the pilot and is not published or passed on.
          Its takes are added to running totals. Normal-voice takes count towards how your country and region
          sound. Practice takes count only as attempts at the accent you chose.
        </p>

        <h2 className="section-title">Changing your mind</h2>
        <p className="prose">
          Before sending: do nothing, or delete everything on Your data. After sending: ask the person who
          invited you. During the pilot the totals are rebuilt from the files held, so deleting your file
          removes your takes from them completely. <Link to="/privacy">Privacy notice</Link>.
        </p>
      </section>
    </div>
  )
}

export default PilotPage
