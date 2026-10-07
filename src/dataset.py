import json
import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Speechocean762Dataset(Dataset):
    def __init__(self, data_root, subset='train'):
        self.data_root = data_root
        self.subset = subset
        
        # Load metadata
        resource_dir = os.path.join(data_root, 'resource')
        with open(os.path.join(resource_dir, 'scores.json'), 'r') as f:
            self.metadata = json.load(f)
            
        self.wav_paths = []
        self.entries = []
        
        # Path to the wav.scp file for the specific subset
        scp_path = os.path.join(data_root, subset, 'wav.scp')
        
        if not os.path.exists(scp_path):
            raise FileNotFoundError(f"wav.scp not found at {scp_path}")
            
        with open(scp_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                utt_id = parts[0]
                # The path in wav.scp is like "WAVE/SPEAKER0001/000010011.WAV"
                rel_path = parts[1]
                wav_path = os.path.join(data_root, rel_path)
                
                # Verify file exists
                if os.path.exists(wav_path) and utt_id in self.metadata:
                    self.wav_paths.append(wav_path)
                    self.entries.append(self.metadata[utt_id])
                elif not os.path.exists(wav_path):
                    print(f"Warning: Audio file not found: {wav_path}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        entry = self.entries[idx]
        
        # Load Audio
        # Speechocean762 is 16kHz
        speech, sr = librosa.load(wav_path, sr=16000)
        
        # Extract phonemes and labels
        phonemes = []
        labels = []
        
        if 'words' in entry:
            for word in entry['words']:
                if 'phones' in word:
                    # phones is a list of strings: ["B", "EH0", "R"]
                    phones_list = word['phones']
                    # phones-accuracy is a list of floats
                    accuracies = word.get('phones-accuracy', [])
                    
                    if len(phones_list) == len(accuracies):
                        phonemes.extend(phones_list)
                        # Ensure labels are integers for CTC token IDs (but here they are scores? Wait!)
                        # IMPORTANT: dataset.py returns 'labels' as accuracy scores (0.0, 1.0, 2.0).
                        # train.py prepares the dataset and overwrites 'labels' with token IDs.
                        # The issue is likely that map() sees the original 'labels' column (floats) 
                        # and enforces that schema even if we try to overwrite it with ints.
                        
                        # But we are removing columns! 
                        # Wait, the ValueError comes from tokenizer.pad().
                        # That means the input to tokenizer.pad() contains floats.
                        # tokenizer.pad() takes `label_features` which comes from `feature["labels"]`.
                        
                        # So `feature["labels"]` in data_collator is a float.
                        # This means `prepare_dataset` is producing floats OR `map` is casting them back.
                        
                        labels.extend(accuracies)
                    else:
                        # Fallback if lengths don't match (shouldn't happen in clean data)
                        phonemes.extend(phones_list)
                        labels.extend([0.0] * len(phones_list))
        
        # Extract the overall accuracy score (0-10), scale down to 0-1 for stability
        score = entry.get('accuracy', 0.0) / 10.0
        
        return {
            "speech": speech,
            "phonemes": phonemes,
            "labels": labels,
            "text": entry['text'],
            "score": score
        }

if __name__ == "__main__":
    # Test the dataloader
    dataset = Speechocean762Dataset("data/speechocean762", subset="train")
    print(f"Loaded {len(dataset)} items")
    sample = dataset[0]
    print("Sample text:", sample['text'])
    print("Phonemes:", sample['phonemes'])
    print("Overall Score:", sample['score'])
