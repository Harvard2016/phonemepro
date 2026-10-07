import os
import json
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import Dataset
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from src.dataset import Speechocean762Dataset

# --- Configuration ---
DATA_ROOT = 'data/speechocean762'
VOCAB_PATH = 'vocab.json'
MODEL_ID = 'facebook/wav2vec2-base'
OUTPUT_DIR = 'results/finetuned-phoneme-model'

# --- 1. Vocabulary & Processor ---
def load_vocab(path):
    with open(path, 'r') as f:
        return json.load(f)

vocab_dict = load_vocab(VOCAB_PATH)
# Create a tokenizer
tokenizer = Wav2Vec2CTCTokenizer(VOCAB_PATH, unk_token='[UNK]', pad_token='[PAD]', word_delimiter_token='|')
# Feature extractor
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
# Processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# --- 2. Dataset Wrapper ---
class PhonemeDataset(Dataset):
    def __init__(self, original_dataset, processor, vocab_dict):
        self.dataset = original_dataset
        self.processor = processor
        self.vocab = vocab_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        speech = item['speech']
        phonemes = item['phonemes']
        
        # 1. Process Audio
        input_values = self.processor(speech, sampling_rate=16000).input_values[0]
        
        # 2. Process Labels (Phonemes -> IDs)
        # Handle unknown phonemes with [UNK]
        label_ids = [self.vocab.get(p.upper(), self.vocab.get('[UNK]', 67)) for p in phonemes]
        
        return {
            'input_values': input_values,
            'labels': label_ids
        }

# --- 3. Data Collator ---
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_values': feature['input_values']} for feature in features]
        label_features = [{'input_ids': feature['labels']} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors='pt',
        )
        
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors='pt',
        )

        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch['labels'] = labels
        return batch

# --- Main Training Function ---
def main():
    print(f'Loading dataset from {DATA_ROOT}...')
    train_ds_raw = Speechocean762Dataset(DATA_ROOT, subset='train')
    test_ds_raw = Speechocean762Dataset(DATA_ROOT, subset='test')
    
    # Reload vocab inside main? No, global is fine for script.
    vocab_dict = load_vocab(VOCAB_PATH)

    train_ds = PhonemeDataset(train_ds_raw, processor, vocab_dict)
    test_ds = PhonemeDataset(test_ds_raw, processor, vocab_dict)
    
    print(f'Train size: {len(train_ds)}')
    print(f'Test size: {len(test_ds)}')
    
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    print(f'Loading base model: {MODEL_ID}')
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID, 
        ctc_loss_reduction='mean', 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
        mask_time_prob=0.05,
        mask_feature_prob=0.05
    )
    
    model.freeze_feature_encoder()
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        use_cpu=os.getenv('TRAIN_USE_CPU', '').lower() in {'1', 'true', 'yes'},
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        eval_strategy='steps',
        num_train_epochs=30,
        fp16=False,
        save_steps=200,
        eval_steps=200,
        logging_steps=50,
        learning_rate=3e-5,
        warmup_steps=100,
        save_total_limit=2,
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=None,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=processor.feature_extractor,
    )
    
    print('Starting training...')
    trainer.train()
    
    print('Training complete. Saving model...')
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f'Model saved to {OUTPUT_DIR}')

if __name__ == '__main__':
    main()
