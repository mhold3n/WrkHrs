#!/usr/bin/env python3
"""
Complete Training Workflow for Coding Models
End-to-end training pipeline for coding-specific AI models
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime

def run_command(cmd, check=True, capture_output=True):
    """Run a command and return the result"""
    print(f"🔄 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
    if check and result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        if not capture_output:
            sys.exit(1)
    return result

def check_system_requirements():
    """Check if system meets training requirements"""
    print("🔍 Checking system requirements...")
    
    # Check GPU availability
    result = run_command("nvidia-smi", check=False)
    if result.returncode != 0:
        print("❌ NVIDIA GPU not detected. Training requires GPU.")
        return False
    
    # Check CUDA
    result = run_command("nvcc --version", check=False, capture_output=True)
    if result.returncode != 0:
        print("⚠️  CUDA toolkit not detected. Some features may not work.")
        # Check if PyTorch can see CUDA
        result = run_command('python -c "import torch; print(torch.cuda.is_available())"', check=False, capture_output=True)
        if result.returncode == 0 and "True" in result.stdout:
            print("✅ PyTorch CUDA support is available.")
        else:
            print("⚠️  PyTorch CUDA support not available.")
    
    # Check Python packages
    required_packages = ["torch", "transformers", "datasets", "accelerate", "peft"]
    for package in required_packages:
        result = run_command(f'python -c "import {package}"', check=False, capture_output=True)
        if result.returncode != 0:
            print(f"❌ Missing package: {package}")
            print(f"Error: {result.stderr}")
            return False
    
    print("✅ System requirements met!")
    return True

def setup_training_environment():
    """Set up training environment"""
    print("🔧 Setting up training environment...")
    
    # Create directories
    directories = [
        "data/raw",
        "data/processed", 
        "models/fine-tuned",
        "logs/training",
        "checkpoints",
        "results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Install training packages
    packages = [
        "torch",
        "transformers>=4.30.0",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
        "trl",
        "wandb",
        "tensorboard",
        "scikit-learn",
        "pandas",
        "numpy"
    ]
    
    for package in packages:
        print(f"📦 Installing {package}...")
        run_command(f"pip install {package}", check=False)
    
    print("✅ Training environment setup complete!")

def prepare_datasets():
    """Prepare training datasets"""
    print("📚 Preparing datasets...")
    
    # Run dataset preparation script
    result = run_command("python scripts/prepare_coding_dataset.py --all", check=False, capture_output=True)
    if result.returncode != 0:
        print("❌ Dataset preparation failed")
        print(f"Error: {result.stderr}")
        return False
    
    print("✅ Datasets prepared successfully!")
    return True

def create_training_config(base_model="microsoft/CodeLlama-7b-Python-hf", 
                          dataset_size=10000,
                          batch_size=4,
                          learning_rate=2e-4,
                          epochs=3):
    """Create training configuration"""
    print("📝 Creating training configuration...")
    
    config = {
        "model_name": base_model,
        "dataset_name": "custom_coding_dataset",
        "max_length": 2048,
        "batch_size": batch_size,
        "gradient_accumulation_steps": 4,
        "learning_rate": learning_rate,
        "num_epochs": epochs,
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
        "remove_unused_columns": False,
        "dataset_size": dataset_size,
        "training_data_paths": [
            "data/prompts/ambiguous_coding.jsonl",
            "data/github_code/github_code.jsonl",
            "data/humaneval/humaneval.jsonl",
            "data/code_review/code_review.jsonl",
            "data/documentation/documentation.jsonl"
        ]
    }
    
    with open("training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Training configuration created!")
    return config

def start_training(config):
    """Start the training process"""
    print("🚀 Starting training...")
    
    # Create training script if it doesn't exist
    if not Path("train_coding_model.py").exists():
        run_command("python scripts/setup-training.py --create-scripts")
    
    # Start training
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = f"logs/training/training_{timestamp}.log"
    
    print(f"📊 Training logs will be saved to: {log_file}")
    print("🔄 Starting training process...")
    
    # Run training
    result = run_command(f"python train_coding_model.py", check=False, capture_output=True)
    
    if result.returncode == 0:
        print("✅ Training completed successfully!")
        return True
    else:
        print("❌ Training failed!")
        print(f"Check logs at: {log_file}")
        return False

def evaluate_model(model_path):
    """Evaluate the trained model"""
    print("📊 Evaluating trained model...")
    
    # Create evaluation script if it doesn't exist
    if not Path("evaluate_model.py").exists():
        run_command("python scripts/setup-training.py --create-scripts")
    
    # Run evaluation
    result = run_command(f"python evaluate_model.py {model_path}", check=False)
    
    if result.returncode == 0:
        print("✅ Model evaluation completed!")
        return True
    else:
        print("❌ Model evaluation failed!")
        return False

def optimize_model(model_path, output_path):
    """Optimize model for deployment"""
    print("⚡ Optimizing model for deployment...")
    
    # Create optimization script if it doesn't exist
    if not Path("optimize_model.py").exists():
        run_command("python scripts/setup-training.py --create-scripts")
    
    # Run optimization
    result = run_command(f"python optimize_model.py --model_path {model_path} --output_path {output_path} --quantization 4bit", check=False)
    
    if result.returncode == 0:
        print("✅ Model optimization completed!")
        return True
    else:
        print("❌ Model optimization failed!")
        return False

def integrate_with_ai_stack(model_path, model_name="custom-coding-model"):
    """Integrate trained model with AI stack"""
    print("🔌 Integrating model with AI stack...")
    
    # Run integration script
    result = run_command(f"python scripts/integrate_trained_model.py --model_path {model_path} --model_name {model_name}", check=False)
    
    if result.returncode == 0:
        print("✅ Model integration completed!")
        return True
    else:
        print("❌ Model integration failed!")
        return False

def main():
    parser = argparse.ArgumentParser(description='Complete training workflow for coding models')
    parser.add_argument('--base-model', default='microsoft/CodeLlama-7b-Python-hf', help='Base model for fine-tuning')
    parser.add_argument('--dataset-size', type=int, default=10000, help='Size of training dataset')
    parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--model-name', default='custom-coding-model', help='Name for the trained model')
    parser.add_argument('--skip-dataset-prep', action='store_true', help='Skip dataset preparation')
    parser.add_argument('--skip-training', action='store_true', help='Skip training (use existing model)')
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip model evaluation')
    parser.add_argument('--skip-optimization', action='store_true', help='Skip model optimization')
    parser.add_argument('--skip-integration', action='store_true', help='Skip AI stack integration')
    parser.add_argument('--existing-model', help='Path to existing trained model')
    
    args = parser.parse_args()
    
    print("🎯 Starting complete training workflow for coding models")
    print(f"📊 Base model: {args.base_model}")
    print(f"📚 Dataset size: {args.dataset_size}")
    print(f"🔧 Batch size: {args.batch_size}")
    print(f"📈 Learning rate: {args.learning_rate}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"🏷️  Model name: {args.model_name}")
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met. Exiting.")
        return
    
    # Setup training environment
    setup_training_environment()
    
    # Prepare datasets
    if not args.skip_dataset_prep:
        if not prepare_datasets():
            print("❌ Dataset preparation failed. Exiting.")
            return
    
    # Create training configuration
    config = create_training_config(
        base_model=args.base_model,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )
    
    # Determine model path
    if args.existing_model:
        model_path = args.existing_model
        print(f"📁 Using existing model: {model_path}")
    else:
        model_path = config["output_dir"]
    
    # Start training
    if not args.skip_training and not args.existing_model:
        if not start_training(config):
            print("❌ Training failed. Exiting.")
            return
    
    # Evaluate model
    if not args.skip_evaluation:
        if not evaluate_model(model_path):
            print("⚠️  Model evaluation failed, but continuing...")
    
    # Optimize model
    if not args.skip_optimization:
        optimized_path = f"{model_path}_optimized"
        if not optimize_model(model_path, optimized_path):
            print("⚠️  Model optimization failed, but continuing...")
        else:
            model_path = optimized_path
    
    # Integrate with AI stack
    if not args.skip_integration:
        if not integrate_with_ai_stack(model_path, args.model_name):
            print("⚠️  AI stack integration failed, but continuing...")
    
    print("\n🎉 Training workflow completed!")
    print(f"📊 Trained model: {model_path}")
    print(f"🏷️  Model name: {args.model_name}")
    print("\nNext steps:")
    print("1. Review the trained model and evaluation results")
    print("2. Deploy the model: ./deploy_trained_model.sh")
    print("3. Test the model with coding tasks")
    print("4. Monitor performance and iterate")

if __name__ == "__main__":
    main()
