import torch
import json
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

def get_model_and_processor(vocab_path, model_name="facebook/wav2vec2-base"):
    # Load vocab
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
        
    # debug: ensure special tokens are in vocab
    # The tokenizer expects a json file where keys are tokens and values are IDs.
    # We already created that.
    
    # Create tokenizer
    # We need to manually construct it because we aren't loading from a pretrained tokenizer repo
    # but creating a new one from our vocab.
    # Wav2Vec2CTCTokenizer arguments: vocab_file, unk_token, pad_token, word_delimiter_token, etc.
    
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path, 
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )
    
    # Create feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    
    # Create processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    # Create model
    # We need to specifying the vocab size and ignore mismatch in head
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name, 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(vocab),
        ignore_mismatched_sizes=True
    )
    
    # Freeze feature extractor (optional, but good for small data)
    model.freeze_feature_encoder()
    
    return model, processor

if __name__ == "__main__":
    # Test model loading
    model, processor = get_model_and_processor("vocab.json")
    print("Model loaded successfully")
    print("Vocab size:", model.config.vocab_size)
    print("Example tokenize:", processor.tokenizer("W IY0".split())) # Expect input splits?
    # Actually tokenizer expects a string if it's char level, or list of words? 
    # For phonemes, we treat them as "words" essentially.
    # If we pass "W IY0", it might split by space and tokenize.
