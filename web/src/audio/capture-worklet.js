// Runs on the audio thread and forwards raw microphone samples to the page,
// one 128-sample block at a time, so a take can be cut with sample accuracy.
class CaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0]?.[0]
    if (channel) this.port.postMessage(channel.slice(0))
    return true
  }
}

registerProcessor('capture', CaptureProcessor)
