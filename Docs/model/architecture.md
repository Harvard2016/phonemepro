# Model Architecture Decision

**Date:** 2026-02-17  
**Status:** Approved

## Selected Architecture: Fine-Tuned wav2vec 2.0 (CTC)

We have selected a **Fine-Tuned wav2vec 2.0 Base model with a CTC Head** for the phoneme-level mispronunciation detection task.

### Architecture Diagram

```mermaid
graph TD
    A[Audio Input (16kHz)] --> B[wav2vec 2.0 Base Encoder]
    B --> C[Linear Projection (768 -> 256)]
    C --> D[Dropout (0.1)]
    D --> E[CTC Head / Phoneme Classifier]
    E --> F[Predicted Phoneme Sequence]
    F --> G[Alignment & Error Detection]
    G --> H[Feedback Generation]
```

### Rationale

1.  **Data Scarcity**: The Speechocean762 dataset contains only ~2,500 training utterances. Training a model from scratch (e.g., Conformer, custom CNN-RNN) would require significantly more data. wav2vec 2.0 provides robust pre-trained speech representations that work well with limited fine-tuning data.
2.  **Phoneme-Level Focus**: wav2vec 2.0's pre-training objective (contrastive learning on quantized latent speech representations) is naturally aligned with learning distinct phonetic units.
3.  **Simplicity & Efficiency**:
    *   **Single-Task Learning**: We avoid the complexity of balancing multiple loss functions (e.g., score regression vs. phoneme classification).
    *   **Implicit Scoring**: Overall pronunciation scores can be derived directly from the phoneme-level accuracy (e.g., percentage of correct phonemes), rendering a separate regression head redundant.
    *   **Latency**: The "Base" model (approx. 95M parameters) fits well within the <2s latency requirement.
4.  **Comparison to Alternatives**:
    *   *Multi-Task (Score + Phoneme)*: Rejected due to data scarcity. The competing objectives (smooth global score vs. sharp local phonemes) often degrade performance on small datasets.
    *   *Whisper*: Rejected. Whisper is an Encoder-Decoder model optimized for ASR (text generation), making forced alignment and granular phoneme timing extraction more complex and less precise than a CTC-based encoder-only approach.

### Implementation Details

*   **Base Model**: `facebook/wav2vec2-base`
*   **Vocabulary**: ARPABET (39 phonemes) to match Speechocean762 annotations.
*   **Loss Function**: CTC Loss.
*   **Training Strategy**:
    *   Freeze the feature extractor (CNN layers).
    *   Freeze the Transformer encoder for the first few epochs (optional, to stabilize the head).
    *   Fine-tune the Transformer layers and the CTC head.
*   **Inference**:
    *   Greedy decoding or Beam search.
    *   Needleman-Wunsch alignment between predicted and canonical phoneme sequences to identify substitutions, insertions, and deletions.
