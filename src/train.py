import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, Wav2Vec2Processor
from dataset import Speechocean762Dataset
from model import get_model_and_processor
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import jiwer
from datasets import Dataset as HFDataset



@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        # map() creates the dataset with "labels" (token IDs)
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad audio
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Replace padding with -100 to ignore loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # Ensure labels are long (int64) for torch cross_entropy / ctc_loss
        batch["labels"] = labels.long()
        
        return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = jiwer.wer(label_str, pred_str)
    return {"per": wer}

if __name__ == "__main__":
    # Setup
    data_root = "data/speechocean762"
    output_dir = "results"
    
    # Load Model & Processor
    model, processor = get_model_and_processor("vocab.json", "facebook/wav2vec2-base")
    
    # Load Datasets
    # We need to map the dataset output to format expected by data collator
    # dataset[i] returns {'speech': np.array, 'phonemes': list, ...}
    # We need 'input_values' and 'labels' (token IDs)
    
    def prepare_dataset(batch):
        audio = batch["speech"]
        # Feature extraction
        # input_values is [seq_len]
        input_values = processor(audio, sampling_rate=16000).input_values[0]
        
        # Tokenize phonemes
        # phonemes is a list of strings e.g. ['W', 'IY0']
        with processor.as_target_processor():
             # processor(..., is_split_into_words=True) handles a single list of words.
             labels = processor(batch["phonemes"], is_split_into_words=True).input_ids
             
             # Ensure labels are python integers
             token_ids = [int(l) for l in labels]
             
        return {"input_values": input_values, "labels": token_ids} # standard key name
        
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        # map() creates the dataset with "target_ids"
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad audio
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Replace padding with -100 to ignore loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # Ensure labels are long (int64) for torch cross_entropy / ctc_loss
        batch["labels"] = labels.long()
        
        return batch

# Define generator at top level for pickling
def data_generator(dataset, processor):
    for i in range(len(dataset)):
        item = dataset[i]
        audio = item["speech"]
        phonemes = item["phonemes"]
        
        # Feature extraction
        # This happens on-the-fly now!
        input_values = processor(audio, sampling_rate=16000).input_values[0]
        
        # Tokenize phonemes
        with processor.as_target_processor():
             labels = processor(phonemes, is_split_into_words=True).input_ids
             labels = [int(l) for l in labels]
        
        # Filter short audio or long labels
        # Wav2Vec2 downsampling factor is 320.
        # CTC loss requires input_length >= target_length (labels).
        # input_length = audio_length / 320.
        
        # Simple check, skip if too short
        if len(input_values) < len(labels) * 320:
            print(f"Skipping item {i}: inputs too short for labels ({len(input_values)} vs {len(labels)})")
            continue
            
        try:
            yield {
                "input_values": input_values.tolist(), # Ensure list for Sequence
                "labels": labels
            }
        except Exception as e:
            print(f"Error yielding item {i}: {e}")
            print(f"Input shape: {input_values.shape}, Labels len: {len(labels)}")
            raise e

if __name__ == "__main__":
    # Setup
    data_root = "data/speechocean762"
    output_dir = "results"
    
    # Load Model & Processor
    model, processor = get_model_and_processor("vocab.json", "facebook/wav2vec2-base")
    
    # Load Datasets
    from datasets import Dataset as HFDataset, Features, Sequence, Value, Array2D
    
    print("Loading and preprocessing datasets (Lazy)...")
    
    # Initialize raw datasets (lightweight)
    train_ds_raw = Speechocean762Dataset(data_root, subset="train")
    test_ds_raw = Speechocean762Dataset(data_root, subset="test")
    
    # Create HF Datasets using generator + strict schema
    # This avoids loading everything into RAM instantly
    print("Creating streaming-like datasets...")
    train_ds = HFDataset.from_generator(
        data_generator, 
        gen_kwargs={"dataset": train_ds_raw, "processor": processor},
        streaming=True
    )
    test_ds = HFDataset.from_generator(
        data_generator, 
        gen_kwargs={"dataset": test_ds_raw, "processor": processor},
        streaming=True
    )
    
    print(f"Created streaming datasets. Training for fixed steps.")
    
    # Data Collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    # Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        # group_by_length=True, # Not supported for streaming
        per_device_train_batch_size=8, # Small batch size for base model
        eval_strategy="steps", # Updated from evaluation_strategy
        max_steps=3200, # Approx 10 epochs for 2500 samples/8 batch size
        fp16=False, # Disable fp16 for stability on MPS
        no_cuda=True, # Force CPU for debugging NaN
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        learning_rate=3e-5, # Lower learning rate for stability
        warmup_steps=300,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=processor.tokenizer, # Correct tokenizer for padding
    )
    
    import gc
    
    try:
        print("Starting training...")
        trainer.train()
        
        # Save final model
        print(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
        
        # Evaluate
        metrics = trainer.evaluate()
        print("Eval metrics:", metrics)
        
    except Exception as e:
        print(f"An error occurred during training: {e}")
        # raise e # Don't raise to ensure cleanup print happens? No, finally runs anyway.
        raise e
        
    finally:
        # Cleanup memory
        print("Cleaning up memory...")
        # Use simple global check or try/except
        try:
             del model
             del processor
             del trainer
             del train_ds
             del test_ds
        except:
             pass
             
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        # For MPS
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("Memory cleanup complete.")
