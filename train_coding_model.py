#!/usr/bin/env python3
"""
Fine-tuning script for coding models
Supports LoRA, QLoRA, and full fine-tuning
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from datetime import datetime

def load_config():
    """Load training configuration"""
    with open("training_config.json", "r") as f:
        return json.load(f)

def setup_model_and_tokenizer(config):
    """Setup model and tokenizer"""
    print(f"Loading model: {config['model_name']}")
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=True
    )
    
    # Setup LoRA if enabled
    if config.get('use_lora', False):
        print("Setting up LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['lora_rank'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            target_modules=["c_attn", "c_proj", "c_fc"],  # GPT-2 specific modules
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Ensure model is in training mode
        model.train()
        for param in model.parameters():
            if param.requires_grad:
                print(f"Parameter requires grad: {param.requires_grad}")
                break
    
    return model, tokenizer

def prepare_dataset(config, tokenizer):
    """Prepare training dataset"""
    print("Loading dataset...")
    
    # Load our local datasets
    from datasets import Dataset
    import json
    
    all_data = []
    
    # Load from our prepared datasets
    dataset_paths = [
        "data/prompts/ambiguous_coding.jsonl",
        "data/github_code/github_code.jsonl",
        "data/humaneval/humaneval.jsonl", 
        "data/code_review/code_review.jsonl",
        "data/documentation/documentation.jsonl"
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"Loading {path}...")
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # Normalize to a common 'content' field for tokenization
                        if 'content' not in data:
                            if 'prompt' in data:
                                data = {**data, 'content': data['prompt']}
                            else:
                                # Skip if neither 'content' nor 'prompt' exist
                                continue
                        all_data.append(data)
    
    print(f"Loaded {len(all_data)} samples total")
    
    # Limit dataset size if specified
    if config.get('dataset_size'):
        all_data = all_data[:config['dataset_size']]
        print(f"Limited to {len(all_data)} samples")
    
    # Create dataset
    dataset = Dataset.from_list(all_data)
    
    def tokenize_function(examples):
        # Tokenize code samples
        return tokenizer(
            examples['content'],
            truncation=True,
            padding=True,
            max_length=config['max_length'],
            return_tensors=None  # Don't return tensors here, let the trainer handle it
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def main():
    """Main training function"""
    config = load_config()
    
    # Setup wandb for experiment tracking
    wandb.init(
        project="ai-stack-coding-training",
        name=f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config=config
    )
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Prepare dataset
    dataset = prepare_dataset(config, tokenizer)
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        num_train_epochs=config['num_epochs'],
        warmup_steps=config['warmup_steps'],
        save_steps=config['save_steps'],
        eval_steps=config['eval_steps'],
        logging_steps=config['logging_steps'],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=config.get('mixed_precision') == 'fp16',
        gradient_checkpointing=config.get('gradient_checkpointing', False),
        dataloader_num_workers=config.get('dataloader_num_workers', 4),
        remove_unused_columns=config.get('remove_unused_columns', False),
        report_to="wandb",
        run_name=f"coding-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Start training
    print("🚀 Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(config['output_dir'])
    
    print("✅ Training completed!")
    wandb.finish()

if __name__ == "__main__":
    main()
