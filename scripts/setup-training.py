#!/usr/bin/env python3
"""
Training Setup Script for AI Stack
Configures training infrastructure for coding-specific models
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result

def check_gpu_memory():
    """Check available GPU memory for training"""
    try:
        result = run_command("nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits")
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                total, free = map(int, line.split(', '))
                gpus.append({'total': total, 'free': free})
        return gpus
    except Exception as e:
        print(f"Could not check GPU memory: {e}")
        return []

def setup_training_environment():
    """Set up training environment with required packages"""
    print("🔧 Setting up training environment...")
    
    # Install training packages
    packages = [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",  # For LoRA
        "bitsandbytes",  # For quantization
        "trl",  # For RLHF
        "wandb",  # For experiment tracking
        "tensorboard",
        "scikit-learn",
        "pandas",
        "numpy"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        run_command(f"pip install {package}")

def create_training_config():
    """Create training configuration files"""
    print("📝 Creating training configuration...")
    
    # Training configuration
    config = {
        "model_name": "microsoft/CodeLlama-7b-Python-hf",
        "dataset_name": "bigcode/the-stack-dedup",
        "max_length": 2048,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "warmup_steps": 100,
        "save_steps": 500,
        "eval_steps": 500,
        "logging_steps": 100,
        "output_dir": "./models/fine-tuned",
        "use_lora": True,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "mixed_precision": "fp16",
        "gradient_checkpointing": True,
        "dataloader_num_workers": 4,
        "remove_unused_columns": False
    }
    
    import json
    with open("training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Training configuration saved to training_config.json")

def create_training_script():
    """Create the main training script"""
    print("📜 Creating training script...")
    
    training_script = '''#!/usr/bin/env python3
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
        trust_remote_code=True
    )
    
    # Setup LoRA if enabled
    if config.get('use_lora', False):
        print("Setting up LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['lora_rank'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

def prepare_dataset(config, tokenizer):
    """Prepare training dataset"""
    print("Loading dataset...")
    
    # Load coding dataset
    dataset = load_dataset(config['dataset_name'], split="train[:10000]")  # Use subset for testing
    
    def tokenize_function(examples):
        # Tokenize code samples
        return tokenizer(
            examples['content'],
            truncation=True,
            padding=False,
            max_length=config['max_length'],
            return_tensors="pt"
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
        evaluation_strategy="steps",
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
'''
    
    with open("train_coding_model.py", "w", encoding="utf-8") as f:
        f.write(training_script)
    
    print("✅ Training script saved to train_coding_model.py")

def create_evaluation_script():
    """Create evaluation script for trained models"""
    print("📊 Creating evaluation script...")
    
    eval_script = '''#!/usr/bin/env python3
"""
Evaluation script for coding models
Tests model performance on coding tasks
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
from datetime import datetime

def evaluate_model(model_path, test_dataset="openai_humaneval"):
    """Evaluate model on coding tasks"""
    print(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load test dataset
    dataset = load_dataset(test_dataset, split="test")
    
    results = []
    
    for i, example in enumerate(dataset):
        print(f"Evaluating example {i+1}/{len(dataset)}")
        
        # Generate code completion
        prompt = example['prompt']
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_code = generated_code[len(prompt):]
        
        results.append({
            'task_id': example['task_id'],
            'prompt': prompt,
            'generated_code': generated_code,
            'test': example['test']
        })
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    results_file = f"evaluation_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Evaluation results saved to {results_file}")
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python evaluate_model.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    evaluate_model(model_path)
'''
    
    with open("evaluate_model.py", "w", encoding="utf-8") as f:
        f.write(eval_script)
    
    print("✅ Evaluation script saved to evaluate_model.py")

def create_optimization_script():
    """Create model optimization script"""
    print("⚡ Creating optimization script...")
    
    opt_script = '''#!/usr/bin/env python3
"""
Model optimization script
Quantizes and optimizes models for deployment
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

def quantize_model(model_path, output_path, quantization_type="4bit"):
    """Quantize model for deployment"""
    print(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if quantization_type == "4bit":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
    elif quantization_type == "8bit":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # Save quantized model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ Quantized model saved to {output_path}")

def merge_lora_weights(model_path, output_path):
    """Merge LoRA weights into base model"""
    print(f"Loading LoRA model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Merge LoRA weights
    if hasattr(model, 'merge_and_unload'):
        model = model.merge_and_unload()
    
    # Save merged model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ Merged model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize model for deployment")
    parser.add_argument("--model_path", required=True, help="Path to model")
    parser.add_argument("--output_path", required=True, help="Output path")
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--merge_lora", action="store_true", help="Merge LoRA weights")
    
    args = parser.parse_args()
    
    if args.merge_lora:
        merge_lora_weights(args.model_path, args.output_path)
    else:
        quantize_model(args.model_path, args.output_path, args.quantization)
'''
    
    with open("optimize_model.py", "w", encoding="utf-8") as f:
        f.write(opt_script)
    
    print("✅ Optimization script saved to optimize_model.py")

def main():
    parser = argparse.ArgumentParser(description='Setup training infrastructure for coding models')
    parser.add_argument('--install-packages', action='store_true', help='Install required packages')
    parser.add_argument('--create-config', action='store_true', help='Create training configuration')
    parser.add_argument('--create-scripts', action='store_true', help='Create training scripts')
    parser.add_argument('--all', action='store_true', help='Do everything')
    
    args = parser.parse_args()
    
    if args.all or args.install_packages:
        setup_training_environment()
    
    if args.all or args.create_config:
        create_training_config()
    
    if args.all or args.create_scripts:
        create_training_script()
        create_evaluation_script()
        create_optimization_script()
    
    if not any([args.install_packages, args.create_config, args.create_scripts, args.all]):
        print("No action specified. Use --help for options.")
        return
    
    # Check GPU setup
    gpus = check_gpu_memory()
    if gpus:
        total_memory = sum(gpu['total'] for gpu in gpus)
        print(f"🖥️  Detected {len(gpus)} GPU(s) with {total_memory/1024:.1f}GB total memory")
        print("✅ Ready for training!")
    
    print("\n🎉 Training infrastructure setup complete!")
    print("\nNext steps:")
    print("1. Prepare your coding dataset")
    print("2. Run: python train_coding_model.py")
    print("3. Evaluate: python evaluate_model.py <model_path>")
    print("4. Optimize: python optimize_model.py --model_path <path> --output_path <output>")

if __name__ == "__main__":
    main()
