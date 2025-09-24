#!/usr/bin/env python3
"""
Coding Dataset Preparation Script
Prepares datasets for training coding-specific models
"""

import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset
import re

def clean_code(code):
    """Clean and normalize code"""
    # Remove excessive whitespace
    code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)
    # Remove trailing whitespace
    code = re.sub(r'[ \t]+$', '', code, flags=re.MULTILINE)
    return code.strip()

def prepare_github_dataset(output_path="data/github_code"):
    """Prepare GitHub code dataset"""
    print("📚 Preparing GitHub code dataset...")
    
    # Use a publicly available coding dataset instead
    try:
        # Try to load a smaller, public dataset
        dataset = load_dataset("codeparrot/github-code", split="train[:5000]")
    except Exception as e:
        print(f"⚠️  Could not load codeparrot/github-code: {e}")
        print("📝 Creating synthetic coding examples...")
        
        # Create synthetic coding examples
        synthetic_examples = [
            {
                'content': '''
def fibonacci(n):
    """Calculate the nth Fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    """Calculate factorial of n"""
    if n <= 1:
        return 1
    return n * factorial(n-1)

def binary_search(arr, target):
    """Binary search implementation"""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
''',
                'language': 'python',
                'size': 500,
                'source': 'synthetic'
            },
            {
                'content': '''
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

class Queue:
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        self.items.insert(0, item)
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items.pop()
    
    def is_empty(self):
        return len(self.items) == 0
''',
                'language': 'python',
                'size': 600,
                'source': 'synthetic'
            }
        ]
        
        processed_data = []
        for example in synthetic_examples:
            processed_data.append({
                'content': clean_code(example['content']),
                'language': example['language'],
                'size': example['size'],
                'source': example['source']
            })
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save processed data
        with open(f"{output_path}/github_code.jsonl", "w", encoding="utf-8") as f:
            for item in processed_data:
                f.write(json.dumps(item) + "\n")
        
        print(f"✅ Synthetic GitHub dataset saved to {output_path}/github_code.jsonl")
        return len(processed_data)
    
    processed_data = []
    
    for item in dataset:
        if item.get('content') and len(item['content']) > 100:
            # Clean the code
            cleaned_code = clean_code(item['content'])
            
            # Extract language if available
            language = item.get('language', 'unknown')
            
            processed_data.append({
                'content': cleaned_code,
                'language': language,
                'size': len(cleaned_code),
                'source': 'github'
            })
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save processed data
    with open(f"{output_path}/github_code.jsonl", "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"✅ GitHub dataset saved to {output_path}/github_code.jsonl")
    return len(processed_data)

def prepare_humaneval_dataset(output_path="data/humaneval"):
    """Prepare HumanEval dataset"""
    print("🧠 Preparing HumanEval dataset...")
    
    dataset = load_dataset("openai_humaneval", split="test")
    
    processed_data = []
    
    for item in dataset:
        # Create training examples from HumanEval
        prompt = item['prompt']
        solution = item['canonical_solution']
        
        # Full function with solution
        full_code = prompt + solution
        
        processed_data.append({
            'content': clean_code(full_code),
            'language': 'python',
            'task_id': item['task_id'],
            'test': item['test'],
            'source': 'humaneval'
        })
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save processed data
    with open(f"{output_path}/humaneval.jsonl", "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"✅ HumanEval dataset saved to {output_path}/humaneval.jsonl")
    return len(processed_data)

def prepare_code_review_dataset(output_path="data/code_review"):
    """Prepare code review dataset"""
    print("🔍 Preparing code review dataset...")
    
    # This would typically load from a code review dataset
    # For now, we'll create a synthetic example
    code_review_examples = [
        {
            'code': '''
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
''',
            'review': '''
This function has exponential time complexity O(2^n). 
Consider using memoization or iterative approach for better performance.
''',
            'improved_code': '''
def calculate_fibonacci(n):
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
'''
        }
    ]
    
    processed_data = []
    
    for example in code_review_examples:
        processed_data.append({
            'content': clean_code(example['code']),
            'review': example['review'],
            'improved_code': clean_code(example['improved_code']),
            'language': 'python',
            'source': 'code_review'
        })
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save processed data
    with open(f"{output_path}/code_review.jsonl", "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"✅ Code review dataset saved to {output_path}/code_review.jsonl")
    return len(processed_data)

def prepare_documentation_dataset(output_path="data/documentation"):
    """Prepare documentation dataset"""
    print("📖 Preparing documentation dataset...")
    
    # Example documentation data
    doc_examples = [
        {
            'code': '''
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
''',
            'documentation': '''
"""
Merge Sort Implementation

This module implements the merge sort algorithm, a divide-and-conquer
sorting algorithm with O(n log n) time complexity.

Functions:
    merge_sort(arr): Sorts an array using merge sort algorithm
    merge(left, right): Merges two sorted arrays into one sorted array

Args:
    arr (list): The array to be sorted

Returns:
    list: A new sorted array

Example:
    >>> arr = [64, 34, 25, 12, 22, 11, 90]
    >>> sorted_arr = merge_sort(arr)
    >>> print(sorted_arr)
    [11, 12, 22, 25, 34, 64, 90]
"""
'''
        }
    ]
    
    processed_data = []
    
    for example in doc_examples:
        processed_data.append({
            'content': clean_code(example['code']),
            'documentation': example['documentation'],
            'language': 'python',
            'source': 'documentation'
        })
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save processed data
    with open(f"{output_path}/documentation.jsonl", "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"✅ Documentation dataset saved to {output_path}/documentation.jsonl")
    return len(processed_data)

def create_training_prompts(dataset_path, output_path="data/training_prompts"):
    """Create training prompts for different coding tasks"""
    print("📝 Creating training prompts...")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Code completion prompts
    completion_prompts = [
        {
            'prompt': 'def fibonacci(n):\n    """Calculate the nth Fibonacci number"""\n    ',
            'completion': 'if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)',
            'task': 'code_completion'
        },
        {
            'prompt': 'class Stack:\n    def __init__(self):\n        self.items = []\n    \n    def push(self, item):\n        ',
            'completion': 'self.items.append(item)\n    \n    def pop(self):\n        if self.is_empty():\n            raise IndexError("Stack is empty")\n        return self.items.pop()\n    \n    def is_empty(self):\n        return len(self.items) == 0',
            'task': 'code_completion'
        }
    ]
    
    # Code review prompts
    review_prompts = [
        {
            'prompt': 'Review this code for potential issues:\n\ndef divide(a, b):\n    return a / b',
            'completion': 'This code has a potential division by zero error. Consider adding a check:\n\ndef divide(a, b):\n    if b == 0:\n        raise ValueError("Cannot divide by zero")\n    return a / b',
            'task': 'code_review'
        }
    ]
    
    # Documentation prompts
    doc_prompts = [
        {
            'prompt': 'Add documentation to this function:\n\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1',
            'completion': 'def binary_search(arr, target):\n    """\n    Perform binary search on a sorted array.\n    \n    Args:\n        arr (list): Sorted array to search in\n        target: Value to search for\n    \n    Returns:\n        int: Index of target if found, -1 otherwise\n    \n    Time Complexity: O(log n)\n    Space Complexity: O(1)\n    """\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1',
            'task': 'documentation'
        }
    ]
    
    all_prompts = completion_prompts + review_prompts + doc_prompts
    
    # Save prompts
    with open(f"{output_path}/training_prompts.jsonl", "w", encoding="utf-8") as f:
        for prompt in all_prompts:
            f.write(json.dumps(prompt) + "\n")
    
    print(f"✅ Training prompts saved to {output_path}/training_prompts.jsonl")
    return len(all_prompts)

def prepare_shadow_prompts(input_path="data/prompts/ambiguous_coding.jsonl"):
    """Validate and stage shadow prompts used for subtext inference training"""
    print("🕶️ Preparing shadow prompts dataset...")
    count = 0
    if not os.path.exists(input_path):
        print(f"⚠️  Shadow prompts file not found at {input_path}")
        return count
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    # Basic validation: length 200-500 chars preferred
                    text = obj.get("prompt", "")
                    if 100 <= len(text) <= 800:
                        count += 1
        print(f"✅ Shadow prompts validated: {count}")
        return count
    except Exception as e:
        print(f"❌ Failed to read shadow prompts: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description='Prepare coding datasets for training')
    parser.add_argument('--github', action='store_true', help='Prepare GitHub dataset')
    parser.add_argument('--humaneval', action='store_true', help='Prepare HumanEval dataset')
    parser.add_argument('--code-review', action='store_true', help='Prepare code review dataset')
    parser.add_argument('--documentation', action='store_true', help='Prepare documentation dataset')
    parser.add_argument('--prompts', action='store_true', help='Create training prompts')
    parser.add_argument('--shadow-prompts', action='store_true', help='Validate shadow prompts dataset')
    parser.add_argument('--all', action='store_true', help='Prepare all datasets')
    
    args = parser.parse_args()
    
    total_samples = 0
    
    if args.all or args.github:
        total_samples += prepare_github_dataset()
    
    if args.all or args.humaneval:
        total_samples += prepare_humaneval_dataset()
    
    if args.all or args.code_review:
        total_samples += prepare_code_review_dataset()
    
    if args.all or args.documentation:
        total_samples += prepare_documentation_dataset()
    
    if args.all or args.prompts:
        total_samples += create_training_prompts("data/")

    if args.all or args.shadow_prompts:
        total_samples += prepare_shadow_prompts()
    
    if not any([args.github, args.humaneval, args.code_review, args.documentation, args.prompts, args.all]):
        print("No action specified. Use --help for options.")
        return
    
    print(f"\n🎉 Dataset preparation complete!")
    print(f"📊 Total samples prepared: {total_samples}")
    print("\nNext steps:")
    print("1. Review the prepared datasets in the data/ directory")
    print("2. Run: python train_coding_model.py")
    print("3. Monitor training progress with wandb")

if __name__ == "__main__":
    main()
