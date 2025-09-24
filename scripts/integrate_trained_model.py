#!/usr/bin/env python3
"""
Trained Model Integration Script
Integrates fine-tuned models with the AI stack
"""

import os
import json
import shutil
import argparse
from pathlib import Path

def update_vllm_config(model_path, model_name="custom-coding-model"):
    """Update vLLM configuration for the trained model"""
    print(f"🔧 Updating vLLM configuration for {model_path}")
    
    # Update .env file
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Update VLLM_MODEL
        content = content.replace(
            f"VLLM_MODEL={os.environ.get('VLLM_MODEL', '')}",
            f"VLLM_MODEL={model_path}"
        )
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated .env file with model path: {model_path}")
    
    # Update docker-compose.prod.yml
    compose_file = Path("compose/docker-compose.prod.yml")
    if compose_file.exists():
        with open(compose_file, 'r') as f:
            content = f.read()
        
        # Update model in command
        content = content.replace(
            '--model ${VLLM_MODEL}',
            f'--model {model_path}'
        )
        
        with open(compose_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated docker-compose.prod.yml with model path: {model_path}")

def create_model_metadata(model_path, model_info):
    """Create model metadata file"""
    print("📝 Creating model metadata...")
    
    metadata = {
        "model_name": model_info.get("name", "custom-coding-model"),
        "model_path": model_path,
        "model_type": model_info.get("type", "causal-lm"),
        "base_model": model_info.get("base_model", "unknown"),
        "training_data": model_info.get("training_data", []),
        "training_config": model_info.get("training_config", {}),
        "performance_metrics": model_info.get("metrics", {}),
        "created_at": model_info.get("created_at", "unknown"),
        "description": model_info.get("description", "Fine-tuned coding model"),
        "languages": model_info.get("languages", ["python", "javascript", "typescript"]),
        "tasks": model_info.get("tasks", ["code_completion", "code_review", "documentation"])
    }
    
    metadata_file = Path(model_path) / "model_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model metadata saved to {metadata_file}")

def create_coding_plugin(model_path):
    """Create a coding-specific plugin for the tool registry"""
    print("🔌 Creating coding plugin...")
    
    plugin_code = f'''#!/usr/bin/env python3
"""
Coding Assistant Plugin
Provides coding-specific tools and capabilities
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, List
import re

class CodingAssistantPlugin:
    """Plugin for coding assistance tasks"""
    
    def __init__(self, model_path="{model_path}"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print(f"✅ Loaded coding model from {{self.model_path}}")
        except Exception as e:
            print(f"❌ Failed to load model: {{e}}")
            self.model = None
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Return tool information"""
        return {{
            "name": "coding_assistant",
            "description": "AI-powered coding assistant for code completion, review, and documentation",
            "parameters": {{
                "task": {{
                    "type": "string",
                    "description": "Type of coding task",
                    "enum": ["completion", "review", "documentation", "refactor", "debug"]
                }},
                "code": {{
                    "type": "string",
                    "description": "Code to process"
                }},
                "language": {{
                    "type": "string",
                    "description": "Programming language",
                    "enum": ["python", "javascript", "typescript", "java", "cpp", "go"]
                }},
                "context": {{
                    "type": "string",
                    "description": "Additional context or requirements"
                }}
            }},
            "category": "coding",
            "version": "1.0.0"
        }}
    
    def execute_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute coding assistance task"""
        if not self.model:
            return {{"error": "Model not loaded", "status": "error"}}
        
        task = parameters.get("task", "completion")
        code = parameters.get("code", "")
        language = parameters.get("language", "python")
        context = parameters.get("context", "")
        
        try:
            if task == "completion":
                result = self._complete_code(code, language, context)
            elif task == "review":
                result = self._review_code(code, language, context)
            elif task == "documentation":
                result = self._generate_documentation(code, language, context)
            elif task == "refactor":
                result = self._refactor_code(code, language, context)
            elif task == "debug":
                result = self._debug_code(code, language, context)
            else:
                result = {{"error": f"Unknown task: {{task}}", "status": "error"}}
            
            return {{"result": result, "status": "success"}}
        
        except Exception as e:
            return {{"error": str(e), "status": "error"}}
    
    def _complete_code(self, code: str, language: str, context: str) -> str:
        """Complete code using the model"""
        prompt = f"# {language}\\n# Context: {context}\\n{code}\\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated[len(prompt):]
        
        return completion.strip()
    
    def _review_code(self, code: str, language: str, context: str) -> str:
        """Review code for issues and improvements"""
        prompt = f"Review this {language} code for potential issues and improvements:\\n\\n{code}\\n\\nReview:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        review = generated[len(prompt):]
        
        return review.strip()
    
    def _generate_documentation(self, code: str, language: str, context: str) -> str:
        """Generate documentation for code"""
        prompt = f"Generate documentation for this {language} code:\\n\\n{code}\\n\\nDocumentation:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        documentation = generated[len(prompt):]
        
        return documentation.strip()
    
    def _refactor_code(self, code: str, language: str, context: str) -> str:
        """Refactor code for better quality"""
        prompt = f"Refactor this {language} code for better quality and performance:\\n\\n{code}\\n\\nRefactored:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        refactored = generated[len(prompt):]
        
        return refactored.strip()
    
    def _debug_code(self, code: str, language: str, context: str) -> str:
        """Debug code and suggest fixes"""
        prompt = f"Debug this {language} code and suggest fixes:\\n\\n{code}\\n\\nDebug suggestions:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        debug_suggestions = generated[len(prompt):]
        
        return debug_suggestions.strip()

# Plugin instance
plugin = CodingAssistantPlugin()
'''
    
    # Save plugin to plugins directory
    plugins_dir = Path("plugins")
    plugins_dir.mkdir(exist_ok=True)
    
    plugin_file = plugins_dir / "coding_assistant_plugin.py"
    with open(plugin_file, 'w', encoding='utf-8') as f:
        f.write(plugin_code)
    
    print(f"✅ Coding plugin saved to {plugin_file}")

def create_deployment_script(model_path):
    """Create deployment script for the trained model"""
    print("🚀 Creating deployment script...")
    
    deploy_script = f'''#!/bin/bash
"""
Deployment script for trained coding model
"""

set -e

MODEL_PATH="{model_path}"
MODEL_NAME="custom-coding-model"

echo "🚀 Deploying trained model: $MODEL_NAME"

# Stop existing services
echo "⏹️  Stopping existing services..."
docker compose -f compose/docker-compose.base.yml -f compose/docker-compose.prod.yml down

# Update environment variables
echo "🔧 Updating environment variables..."
export VLLM_MODEL="$MODEL_PATH"
export VLLM_GPU_MEMORY_UTILIZATION="0.85"
export VLLM_MAX_MODEL_LEN="4096"

# Start vLLM service with trained model
echo "🔄 Starting vLLM service with trained model..."
docker compose -f compose/docker-compose.base.yml -f compose/docker-compose.prod.yml up -d llm-runner

# Wait for model to load
echo "⏳ Waiting for model to load..."
sleep 60

# Check if vLLM is healthy
echo "🏥 Checking vLLM health..."
for i in {{1..10}}; do
    if curl -f http://localhost:8001/health > /dev/null 2>&1; then
        echo "✅ vLLM is healthy!"
        break
    else
        echo "⏳ Waiting for vLLM to be ready... ($i/10)"
        sleep 30
    fi
done

# Start other services
echo "🔄 Starting other services..."
docker compose -f compose/docker-compose.base.yml -f compose/docker-compose.dev.yml up -d

# Test the deployment
echo "🧪 Testing deployment..."
curl -X POST http://localhost:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: change-this-secret-key-in-production" \\
  -d '{{
    "messages": [
      {{"role": "user", "content": "Complete this Python function: def fibonacci(n):"}}
    ],
    "model": "custom-coding-model",
    "temperature": 0.1,
    "max_tokens": 200
  }}'

echo "🎉 Deployment complete!"
echo "📊 Model: $MODEL_NAME"
echo "🔗 API: http://localhost:8080"
echo "📈 Monitoring: http://localhost:8001"
'''
    
    deploy_file = Path("deploy_trained_model.sh")
    with open(deploy_file, 'w', encoding='utf-8') as f:
        f.write(deploy_script)
    
    # Make executable
    os.chmod(deploy_file, 0o755)
    
    print(f"✅ Deployment script saved to {deploy_file}")

def main():
    parser = argparse.ArgumentParser(description='Integrate trained model with AI stack')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--model_name', default='custom-coding-model', help='Name for the model')
    parser.add_argument('--base_model', default='unknown', help='Base model used for training')
    parser.add_argument('--description', default='Fine-tuned coding model', help='Model description')
    parser.add_argument('--languages', nargs='+', default=['python', 'javascript', 'typescript'], help='Supported languages')
    parser.add_argument('--tasks', nargs='+', default=['code_completion', 'code_review', 'documentation'], help='Supported tasks')
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model_path).exists():
        print(f"❌ Model path does not exist: {args.model_path}")
        return
    
    # Create model info
    model_info = {
        "name": args.model_name,
        "type": "causal-lm",
        "base_model": args.base_model,
        "description": args.description,
        "languages": args.languages,
        "tasks": args.tasks,
        "created_at": "2024-01-01",
        "training_data": ["github", "humaneval", "code_review"],
        "training_config": {
            "learning_rate": 2e-4,
            "batch_size": 4,
            "epochs": 3,
            "lora_rank": 16
        },
        "metrics": {
            "perplexity": "unknown",
            "bleu_score": "unknown"
        }
    }
    
    # Update configurations
    update_vllm_config(args.model_path, args.model_name)
    
    # Create metadata
    create_model_metadata(args.model_path, model_info)
    
    # Create plugin
    create_coding_plugin(args.model_path)
    
    # Create deployment script
    create_deployment_script(args.model_path)
    
    print(f"\n🎉 Model integration complete!")
    print(f"📊 Model: {args.model_name}")
    print(f"📁 Path: {args.model_path}")
    print(f"🔌 Plugin: plugins/coding_assistant_plugin.py")
    print(f"🚀 Deploy: ./deploy_trained_model.sh")
    
    print("\nNext steps:")
    print("1. Review the updated configuration files")
    print("2. Run: ./deploy_trained_model.sh")
    print("3. Test the deployed model with coding tasks")

if __name__ == "__main__":
    main()
