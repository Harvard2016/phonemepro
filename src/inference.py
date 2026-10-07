
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from dataset import Speechocean762Dataset
import argparse

def run_inference(num_samples=5):
    print("Loading model and processor from 'results/'...")
    model_path = "results"
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Loading test dataset...")
    ds = Speechocean762Dataset("data/speechocean762", subset="test")
    
    model.eval()
    
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    
    # Check a few random samples
    indices = range(min(num_samples, len(ds)))
    
    for i in indices:
        item = ds[i]
        audio = item["speech"]
        target_phonemes = item["phonemes"]
        
        # 1. Process Audio
        input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values
        
        # 2. Forward Pass
        with torch.no_grad():
            logits = model(input_values).logits
            
        # 3. Decode
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)[0]
        
        target_str = " ".join(target_phonemes)

        print(f"\nSample {i}:")
        print(f"  Target:    {target_str}")
        print(f"  Predicted: {pred_str}")
        
    print("\n" + "="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to run")
    args = parser.parse_args()
    run_inference(args.num_samples)
