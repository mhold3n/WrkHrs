#!/usr/bin/env python3
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
