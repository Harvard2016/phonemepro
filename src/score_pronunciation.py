
import torch
import jiwer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from dataset import Speechocean762Dataset
import sys

def score_pronunciation(model_path="results", num_samples=5):
    print(f"Loading model from {model_path}...")
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    try:
        ds = Speechocean762Dataset("data/speechocean762", subset="test")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("\n" + "="*60)
    print("PRONUNCIATION SCORING DEMO")
    print("="*60)

    for i in range(min(num_samples, len(ds))):
        item = ds[i]
        audio = item["speech"]
        target_phonemes = item["phonemes"] # List of strings
        
        # Predict
        input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values
        with torch.no_grad():
            logits = model(input_values).logits
        
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)[0]
        pred_phonemes = pred_str.split(" ")
        
        # Clean up empty strings
        pred_phonemes = [p for p in pred_phonemes if p.strip()]
        
        ref_str = " ".join(target_phonemes)
        hyp_str = " ".join(pred_phonemes)
        
        print(f"\nSample {i}:")
        print(f"  Canonical (Expected): {ref_str}")
        print(f"  Recognized (Actual):   {hyp_str}")
        
        # Scoring with jiwer
        output = jiwer.process_words(ref_str, hyp_str)
        wer = output.wer
        
        # Simple correctness score (0-10)
        # correctness = (1 - WER) * 10, clamped to 0
        score = max(0, (1 - wer) * 10)
        
        print(f"  - Word Error Rate (WER): {wer:.2f}")
        print(f"  - Estimated Score (0-10): {score:.1f}")
        
        print("  - Differences:")
        
        # Access alignment chunks
        # alignment is typically a list of AlignmentChunk objects or similar
        # Depending on version, fields might vary. We use standard attributes.
        
        for align_chunk in output.alignments[0]:
            # align_chunk has properties: type, ref_start_idx, ref_end_idx, hyp_start_idx, hyp_end_idx
            op = align_chunk.type
            
            if op == 'equal':
                continue
                
            ref_start = align_chunk.ref_start_idx
            ref_end = align_chunk.ref_end_idx
            hyp_start = align_chunk.hyp_start_idx
            hyp_end = align_chunk.hyp_end_idx
            
            ref_segment = target_phonemes[ref_start:ref_end]
            hyp_segment = pred_phonemes[hyp_start:hyp_end]
            
            if op == 'substitute':
                print(f"    [SUB] Expected '{' '.join(ref_segment)}' -> Heard '{' '.join(hyp_segment)}'")
            elif op == 'delete':
                print(f"    [DEL] Missed '{' '.join(ref_segment)}'")
            elif op == 'insert':
                print(f"    [INS] Inserted '{' '.join(hyp_segment)}'")

    print("\n" + "="*60)

if __name__ == "__main__":
    score_pronunciation()
