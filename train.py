import os
import argparse
import json
import logging
import sys
import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
    
    # Data, model, and output directories
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--eval_dir', type=str, default=os.environ['SM_CHANNEL_EVAL'])
    parser.add_argument('--num_labels', type=int, default=2)  # Adjust based on your task
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load datasets
    train_dataset = load_dataset('json', data_files=os.path.join(args.training_dir, 'train.jsonl'))['train']
    eval_dataset = load_dataset('json', data_files=os.path.join(args.eval_dir, 'eval.jsonl'))['train']
    
    # Load pretrained model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=args.num_labels
    )
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    
    # Data collator for batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    eval_result = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_result}")
    
    # Save model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save special tokens file
    special_tokens_map = tokenizer.special_tokens_map
    with open(os.path.join(args.output_dir, 'special_tokens_map.json'), 'w') as f:
        json.dump(special_tokens_map, f)

if __name__ == "__main__":
    main()