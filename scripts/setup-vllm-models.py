#!/usr/bin/env python3
"""
vLLM Model Setup Script
Downloads and configures models for vLLM multi-GPU inference
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
    """Check available GPU memory"""
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

def get_recommended_models(gpu_memory_gb):
    """Get recommended models based on GPU memory"""
    models = {
        'small': {
            'name': 'microsoft/DialoGPT-medium',
            'size_gb': 0.5,
            'description': 'Small conversational model, good for testing'
        },
        'medium': {
            'name': 'meta-llama/Llama-2-7b-chat-hf',
            'size_gb': 13,
            'description': '7B parameter model, good balance of performance and size'
        },
        'large': {
            'name': 'meta-llama/Llama-2-13b-chat-hf',
            'size_gb': 26,
            'description': '13B parameter model, requires tensor parallelism'
        },
        'xlarge': {
            'name': 'meta-llama/Llama-2-70b-chat-hf',
            'size_gb': 140,
            'description': '70B parameter model, requires pipeline parallelism'
        }
    }
    
    # Filter models based on available memory
    available_models = {}
    for size, model_info in models.items():
        if model_info['size_gb'] <= gpu_memory_gb * 0.8:  # 80% of available memory
            available_models[size] = model_info
    
    return available_models

def download_model(model_name, models_dir):
    """Download a model using huggingface-hub"""
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)
    
    print(f"Downloading model: {model_name}")
    print(f"Destination: {models_path}")
    
    # Use huggingface-hub to download
    cmd = f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{model_name}', local_dir='{models_path}/{model_name.replace('/', '--')}', local_dir_use_symlinks=False)\""
    
    try:
        run_command(cmd)
        print(f"✅ Model {model_name} downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False

def update_env_file(model_name):
    """Update .env file with the new model"""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        return False
    
    # Read current content
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Update VLLM_MODEL
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('VLLM_MODEL='):
            lines[i] = f'VLLM_MODEL={model_name}'
            break
    
    # Write back
    with open(env_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Updated .env file with model: {model_name}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Setup vLLM models for multi-GPU inference')
    parser.add_argument('--model', help='Specific model to download')
    parser.add_argument('--list', action='store_true', help='List recommended models')
    parser.add_argument('--models-dir', default='./models', help='Directory to store models')
    parser.add_argument('--auto', action='store_true', help='Automatically select best model')
    
    args = parser.parse_args()
    
    # Check GPU memory
    gpus = check_gpu_memory()
    if not gpus:
        print("❌ No GPUs detected or nvidia-smi not available")
        sys.exit(1)
    
    total_memory = sum(gpu['total'] for gpu in gpus)
    total_memory_gb = total_memory / 1024
    
    print(f"🖥️  Detected {len(gpus)} GPU(s) with {total_memory_gb:.1f}GB total memory")
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu['total']/1024:.1f}GB total, {gpu['free']/1024:.1f}GB free")
    
    # Get recommended models
    available_models = get_recommended_models(total_memory_gb)
    
    if args.list:
        print("\n📋 Available models for your hardware:")
        for size, model_info in available_models.items():
            print(f"   {size.upper()}: {model_info['name']}")
            print(f"      Size: {model_info['size_gb']}GB")
            print(f"      Description: {model_info['description']}")
            print()
        return
    
    # Select model
    if args.model:
        model_name = args.model
    elif args.auto:
        # Select the largest model that fits
        if available_models:
            largest = max(available_models.items(), key=lambda x: x[1]['size_gb'])
            model_name = largest[1]['name']
            print(f"🤖 Auto-selected: {model_name}")
        else:
            print("❌ No suitable models found for your hardware")
            sys.exit(1)
    else:
        print("\n📋 Available models:")
        for i, (size, model_info) in enumerate(available_models.items(), 1):
            print(f"   {i}. {size.upper()}: {model_info['name']} ({model_info['size_gb']}GB)")
        
        try:
            choice = int(input("\nSelect model (number): ")) - 1
            model_name = list(available_models.values())[choice]['name']
        except (ValueError, IndexError):
            print("❌ Invalid selection")
            sys.exit(1)
    
    # Download model
    if download_model(model_name, args.models_dir):
        # Update .env file
        update_env_file(model_name)
        
        print(f"\n🎉 Setup complete!")
        print(f"   Model: {model_name}")
        print(f"   Location: {args.models_dir}")
        print(f"   Configuration: Updated .env file")
        print(f"\n🚀 To start vLLM with this model:")
        print(f"   docker compose -f compose/docker-compose.base.yml -f compose/docker-compose.prod.yml up -d llm-runner")
    else:
        print("❌ Setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
