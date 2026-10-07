import { useEffect, useState } from 'react'
import { motion } from 'motion/react'
import LineChart from '../components/LineChart'
import { LoadingState, MessageState } from '../components/States'
import { getModelInfo } from '../lib/api'
import { toIpa } from '../lib/phonemes'

const MODE_LABEL = {
  multitask: 'Multitask model with learned scoring',
  finetuned: 'Fine-tuned phoneme model',
  'asr-fallback': 'Public ASR fallback',
}

const percent = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`
const orDash = (value) => (value === null || value === undefined ? 'not recorded' : value)

function ModelInfoPage() {
  const [info, setInfo] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getModelInfo().then(setInfo).catch((failure) => setError(failure.message))
  }, [])

  if (error) return <MessageState title="Model details are unavailable">{error}</MessageState>
  if (!info) return <LoadingState label="Reading the lab notes" />

  const { evaluation, training, config, runtime, dataset } = info
  const trainLoss = info.log_history.filter((e) => e.loss !== undefined).map((e) => ({ x: e.step, y: e.loss }))
  const evalLoss = info.log_history.filter((e) => e.eval_loss !== undefined).map((e) => ({ x: e.step, y: e.eval_loss }))
  const isFallback = runtime.mode === 'asr-fallback'
  const detection = evaluation?.mispronunciation_detection
  const learned = evaluation?.learned_score

  return (
    <div className="page">
      <header className="page__head">
        <p className="label">Model report</p>
        <h1 className="page__title">How well does it hear?</h1>
        <p className="page__lede">
          A wav2vec 2.0 encoder fine-tuned to transcribe speech into ARPABET phonemes, measured on{' '}
          {evaluation ? `${evaluation.utterances.toLocaleString()} test-split utterances from ${evaluation.speakers} speakers` : 'the test split'}.
        </p>
        <p className={`status${isFallback ? ' status--warn' : ''}`}>
          <span className="status__dot" aria-hidden="true" />
          Running: {MODE_LABEL[runtime.mode] ?? runtime.mode}
          <code>{runtime.model_source}</code>
        </p>
      </header>

      {isFallback && (
        <p className="notice notice--static">
          The fine-tuned weights are not installed, so scores come from a general speech recognizer and are rough.
          The figures below describe the fine-tuned model, not what is running now.
        </p>
      )}

      {evaluation ? (
        <dl className="figures">
          <div className="figure">
            <dt className="label">Phoneme error rate</dt>
            <dd className="figure__value">{percent(evaluation.per)}</dd>
            <dd className="figure__note">stress ignored · {percent(evaluation.per_with_stress)} with stress</dd>
          </div>
          <div className="figure">
            <dt className="label">Score vs human raters</dt>
            <dd className="figure__value">r = {(learned && runtime.mode === 'multitask'
              ? learned.test_correlation_app_score
              : evaluation.score_correlation.app_score_vs_human_accuracy).toFixed(2)}
            </dd>
            <dd className="figure__note">
              {learned && runtime.mode === 'multitask'
                ? `Pearson, per utterance · ${learned.test_correlation_alignment_score.toFixed(2)} without the learned score head`
                : 'Pearson, per utterance'}
            </dd>
          </div>
          <div className="figure">
            <dt className="label">Mispronunciations caught</dt>
            <dd className="figure__value">{percent(detection.recall, 0)}</dd>
            <dd className="figure__note">recall · precision {percent(detection.precision, 0)}</dd>
          </div>
          <div className="figure">
            <dt className="label">False alarms</dt>
            <dd className="figure__value">{percent(detection.false_alarm_rate, 0)}</dd>
            <dd className="figure__note">of well-pronounced sounds flagged</dd>
          </div>
        </dl>
      ) : (
        <p className="focus__empty">
          No evaluation on file. Run <code>python -m scripts.evaluate infer</code> then <code>report</code>.
        </p>
      )}

      {evaluation && (
        <p className="caveat">
          Reference phonemes are the dictionary pronunciation, and the speakers are learners, so part of the error
          rate is real mispronunciation rather than model error. The model flags far more sounds than human raters
          do: treat a red mark as a prompt to listen again, not a verdict.
        </p>
      )}

      <div className="spread">
        <section className="spread__main" aria-label="Training loss">
          <h2 className="section-title">Training</h2>
          {trainLoss.length > 1 ? (
            <LineChart
              logY
              xLabel="Step"
              series={[
                { name: 'Train loss', tone: 'ink', points: trainLoss },
                ...(evalLoss.length > 1 ? [{ name: 'Eval loss', tone: 'accent', points: evalLoss }] : []),
              ]}
            />
          ) : (
            <p className="focus__empty">
              No training log next to the model weights. Place <code>trainer_state.json</code> in the model folder.
            </p>
          )}
        </section>

        <section className="spread__side" aria-label="Configuration">
          <h2 className="section-title">Run</h2>
          <dl className="spec">
            <div><dt>Steps</dt><dd>{orDash(training.max_steps?.toLocaleString())}</dd></div>
            <div><dt>Epochs</dt><dd>{orDash(training.epochs)}</dd></div>
            <div><dt>Batch size</dt><dd>{orDash(training.batch_size)}</dd></div>
            <div><dt>Peak learning rate</dt><dd>{training.learning_rate ? training.learning_rate.toExponential(1) : 'not recorded'}</dd></div>
            <div><dt>Optimizer</dt><dd>{orDash(training.optimizer)}</dd></div>
            <div><dt>Final eval loss</dt><dd>{orDash(training.final_eval_loss)}</dd></div>
            <div><dt>Dataset</dt><dd>{dataset.name}</dd></div>
            <div><dt>Train / test</dt><dd>{dataset.train_samples.toLocaleString()} / {dataset.test_samples.toLocaleString()}</dd></div>
          </dl>
        </section>
      </div>

      <section aria-label="Pipeline">
        <h2 className="section-title">From voice to verdict</h2>
        <ol className="pipeline">
          {[
            ['Audio', '16 kHz mono'],
            ['wav2vec 2.0', `${orDash(config.num_hidden_layers)} layers · ${orDash(config.hidden_size)} wide`],
            ['CTC head', `${orDash(config.vocab_size)} symbols`],
            ['Alignment', 'edit distance to target'],
            ['Score', runtime.mode === 'multitask' ? 'alignment + learned head' : '0 to 10'],
          ].map(([name, detail], index) => (
            <motion.li
              key={name}
              className="pipeline__step"
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.2 }}
              transition={{ duration: 0.5, delay: index * 0.08 }}
            >
              <span className="pipeline__index">{index + 1}</span>
              <span className="pipeline__name">{name}</span>
              <span className="pipeline__detail">{detail}</span>
            </motion.li>
          ))}
        </ol>
      </section>

      {evaluation && (
        <div className="spread">
          <section className="spread__main" aria-label="Error rate by phoneme">
            <h2 className="section-title">Error rate by sound</h2>
            <ul className="periodic">
              {evaluation.per_phoneme.map((item) => (
                <li
                  key={item.phoneme}
                  className={`periodic__cell${item.error_rate >= 0.5 ? ' periodic__cell--deep' : ''}`}
                  style={{ '--rate': item.error_rate }}
                  title={`${item.phoneme}: ${percent(item.error_rate)} of ${item.count.toLocaleString()} occurrences`}
                >
                  <span className="periodic__ipa ipa">{toIpa(item.phoneme)}</span>
                  <span className="periodic__name">{item.phoneme}</span>
                  <span className="periodic__rate">{Math.round(item.error_rate * 100)}</span>
                </li>
              ))}
            </ul>
            <p className="caveat">
              Percent of occurrences not matched. ZH and OY appear about 20 times each in the test set and the model
              never produces them, so practice does not count those two sounds against you.
            </p>
          </section>

          <section className="spread__side" aria-label="Most common confusions">
            <h2 className="section-title">Most confused</h2>
            <ol className="bars">
              {evaluation.top_confusions.slice(0, 10).map((item) => (
                <li key={`${item.expected}-${item.heard}`} className="bars__row">
                  <span className="bars__name">
                    <span className="ipa">{toIpa(item.expected)}</span> heard as <span className="ipa">{toIpa(item.heard)}</span>
                  </span>
                  <span className="meter" aria-hidden="true">
                    <span
                      className="meter__fill"
                      style={{ transform: `scaleX(${item.count / evaluation.top_confusions[0].count})` }}
                    />
                  </span>
                  <span className="bars__value">{item.count}</span>
                </li>
              ))}
            </ol>
          </section>
        </div>
      )}
    </div>
  )
}

export default ModelInfoPage
