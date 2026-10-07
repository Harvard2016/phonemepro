import json
import os
from dataset import Speechocean762Dataset
from tqdm import tqdm

def create_vocab(data_root, output_file):
    dataset = Speechocean762Dataset(data_root, subset='train')
    all_phonemes = set()
    
    print("Collecting phonemes...")
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        phonemes = item['phonemes']
        all_phonemes.update(phonemes)
        
    # Add special tokens
    # CTC usually needs <pad> and maybe | for word boundary if we were doing char-level
    # For phonemes, we treat each phoneme as a token.
    # Wav2Vec2CTCTokenizer usually handles this.
    
    vocab_list = sorted(list(all_phonemes))
    vocab_dict = {v: int(k) for k, v in enumerate(vocab_list)}
    
    # Add special tokens starting from the next index
    # Standard Wav2Vec2 special tokens
    vocab_dict["[PAD]"] = len(vocab_dict)
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["<s>"] = len(vocab_dict)
    vocab_dict["</s>"] = len(vocab_dict)
        
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(vocab_dict, f, indent=2)
        
    print(f"Vocab saved to {output_file} with {len(vocab_dict)} tokens")

if __name__ == "__main__":
    create_vocab("data/speechocean762", "vocab.json")
