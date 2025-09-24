#!/usr/bin/env python3
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
