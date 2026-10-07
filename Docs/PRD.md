
# Product Requirements Document (PRD)
## Phoneme-Level Mispronunciation Detection System
### Dataset: Speechocean762

---

## 1. Product Overview

This project aims to build a phoneme-level pronunciation assessment and feedback system for non-native English learners.  
The system will analyze user-recorded speech, detect mispronounced phonemes, and generate corrective feedback.

The system is trained primarily using the Speechocean762 dataset.

---

## 2. Problem Statement

Many English learners:

- Mispronounce specific phonemes
- Do not know which sound is incorrect
- Do not receive actionable feedback
- Receive only overall scores from existing tools

This system addresses the gap by providing phoneme-level mispronunciation detection.

---

## 3. Goals

### Primary Goals

- Detect mispronounced phonemes in prompted speech
- Provide phoneme-level corrective feedback
- Output overall pronunciation score (0–100)

### Secondary Goals

- Maintain inference latency under 2 seconds
- Ensure speaker-independent evaluation
- Use pre-trained speech encoders for efficiency

---

## 4. Target Users

- Non-native English learners
- Students preparing for proficiency exams
- Self-learners using web or desktop applications

---

## 5. Scope

### In Scope

- Prompted single-word pronunciation
- Phoneme-level mispronunciation detection
- Overall pronunciation scoring
- Rule-based corrective feedback
- Model trained on Speechocean762

### Out of Scope

- Free-form conversational correction
- Accent classification
- Real-time conversation assistance
- Native-level articulatory coaching system

---

## 6. Dataset Context (Speechocean762)

### Dataset Summary

- 5,000 read-aloud English sentences
- 250 Mandarin-speaking learners
- 16kHz WAV recordings
- Multi-level annotation (phoneme, word, sentence)
- 2,500 train / 2,500 test split (speaker-independent)

### Label Types

Phoneme-level:
- Score ∈ {0,1,2} (correct / accented / incorrect)

Word-level:
- Accuracy (0–10)
- Stress score
- Total score

Sentence-level:
- Accuracy
- Fluency
- Prosody
- Completeness
- Overall score

---

## 7. Functional Requirements

The system shall:

1. Accept a prompted English word.
2. Record user speech via microphone.
3. Extract acoustic features using a pre-trained speech encoder.
4. Perform phoneme-level alignment (CTC or forced alignment).
5. Compare predicted phoneme sequence with canonical sequence.
6. Detect phoneme-level mispronunciations.
7. Generate corrective feedback.
8. Output an overall pronunciation score.

---

## 8. Model Architecture

### Option A: CTC-Based Phoneme Detection

Audio  
→ Pretrained wav2vec2 Encoder  
→ CTC Phoneme Decoder  
→ Predicted Phoneme Sequence  
→ Compare with Canonical Sequence  
→ Detect Substitution / Deletion / Insertion  

### Option B: Multi-Task Model

Shared Encoder: wav2vec2 / HuBERT  
Head 1: Regression (overall pronunciation score)  
Head 2: Phoneme classification (correct vs incorrect)  

Recommended approach: Multi-task fine-tuning for both scoring and phoneme detection.

---

## 9. Training Strategy

- Fine-tune pretrained self-supervised speech model
- Use Cross-Entropy loss for phoneme classification
- Use Mean Squared Error (MSE) for score regression
- Apply class weighting to address label imbalance
- Ensure no speaker overlap between train/test

---

## 10. Evaluation Metrics

- Phoneme-level Precision / Recall / F1
- Phone Error Rate (PER)
- Pearson correlation (score prediction)
- Root Mean Squared Error (RMSE)
- Inference latency

---

## 11. Feedback Generation Logic

If phoneme confidence < threshold:

- Highlight phoneme
- Identify likely substitution (if detectable)
- Map to rule-based corrective suggestion

Example:

Target: B AE M B UW Z AH L  
User:   B AE M B UW S AH L  

Detected: Z → S substitution  
Feedback: “The /z/ sound should be voiced. Try buzzing your throat.”

---

## 12. System Architecture

User Input  
→ Audio Capture  
→ Feature Extraction  
→ Phoneme Alignment  
→ Error Detection  
→ Score Computation  
→ Feedback Generation  
→ UI Output  

---

## 13. Success Criteria

The system is successful if:

- Phoneme-level F1 exceeds baseline
- Score correlation is statistically significant
- Latency < 2 seconds
- Demo highlights mispronounced phonemes correctly

---

## 14. Risks & Mitigation

Risk: Imbalanced score distribution  
Mitigation: Class weighting and augmentation  

Risk: Overfitting to Mandarin accent  
Mitigation: Regularization and careful validation  

Risk: Alignment errors  
Mitigation: Experiment with CTC vs forced alignment  

---

## 15. Deliverables

- Trained phoneme-level mispronunciation detection model
- Evaluation report with metrics
- Demo interface
- GitHub repository with documentation
