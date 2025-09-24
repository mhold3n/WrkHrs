# Quick Start Guide: Training Coding Models

## 🚀 Quick Start (5 minutes)

### 1. Setup Training Environment
```bash
# Install training packages and create scripts
python scripts/setup-training.py --all
```

### 2. Prepare Datasets
```bash
# Download and prepare coding datasets
python scripts/prepare_coding_dataset.py --all
```

### 3. Start Training
```bash
# Run complete training workflow
python scripts/train_coding_workflow.py --base-model microsoft/CodeLlama-7b-Python-hf --epochs 1
```

### 4. Deploy Trained Model
```bash
# Deploy to your AI stack
./deploy_trained_model.sh
```

## 📊 Training Options

### Quick Training (1-2 hours)
```bash
python scripts/train_coding_workflow.py \
  --base-model microsoft/CodeLlama-7b-Python-hf \
  --dataset-size 5000 \
  --batch-size 2 \
  --epochs 1 \
  --learning-rate 5e-4
```

### Full Training (6-12 hours)
```bash
python scripts/train_coding_workflow.py \
  --base-model microsoft/CodeLlama-7b-Python-hf \
  --dataset-size 50000 \
  --batch-size 4 \
  --epochs 3 \
  --learning-rate 2e-4
```

### Custom Training
```bash
python scripts/train_coding_workflow.py \
  --base-model your-model \
  --dataset-size 10000 \
  --batch-size 4 \
  --epochs 2 \
  --learning-rate 1e-4 \
  --model-name my-coding-assistant
```

## 🎯 Training for Specific Tasks

### Code Completion
- **Base Model**: `microsoft/CodeLlama-7b-Python-hf`
- **Dataset**: GitHub code + HumanEval
- **Focus**: Function completion, class methods

### Code Review
- **Base Model**: `microsoft/CodeLlama-7b-Instruct-hf`
- **Dataset**: Code review examples
- **Focus**: Bug detection, optimization suggestions

### Documentation Generation
- **Base Model**: `microsoft/CodeLlama-7b-Instruct-hf`
- **Dataset**: Code + documentation pairs
- **Focus**: Docstring generation, API documentation

## 🔧 Hardware Optimization

### Your Setup (RTX 4070 Ti + RTX 2080)
```bash
# Optimized for your 20GB VRAM
python scripts/train_coding_workflow.py \
  --batch-size 4 \
  --learning-rate 2e-4 \
  --base-model microsoft/CodeLlama-7b-Python-hf
```

### Memory Distribution
- **RTX 4070 Ti (12GB)**: Model weights + training data
- **RTX 2080 (8GB)**: Gradient computation + optimizer states

## 📈 Monitoring Training

### Weights & Biases
```bash
# Training metrics are automatically logged to wandb
# View at: https://wandb.ai/your-username/ai-stack-coding-training
```

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir logs/training
# View at: http://localhost:6006
```

## 🧪 Testing Trained Models

### Code Completion Test
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-this-secret-key-in-production" \
  -d '{
    "messages": [
      {"role": "user", "content": "Complete this Python function: def fibonacci(n):"}
    ],
    "model": "custom-coding-model",
    "temperature": 0.1,
    "max_tokens": 200
  }'
```

### Code Review Test
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-this-secret-key-in-production" \
  -d '{
    "messages": [
      {"role": "user", "content": "Review this code: def divide(a, b): return a / b"}
    ],
    "model": "custom-coding-model",
    "temperature": 0.1,
    "max_tokens": 200
  }'
```

## 🔄 Iterative Improvement

### 1. Train Initial Model
```bash
python scripts/train_coding_workflow.py --epochs 1
```

### 2. Evaluate Performance
```bash
python evaluate_model.py models/fine-tuned
```

### 3. Improve Dataset
- Add more domain-specific code
- Include more examples of target tasks
- Balance different programming languages

### 4. Retrain with Improvements
```bash
python scripts/train_coding_workflow.py --epochs 2 --learning-rate 1e-4
```

## 🚨 Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python scripts/train_coding_workflow.py --batch-size 2

# Use gradient accumulation
# Edit training_config.json: "gradient_accumulation_steps": 8
```

### Slow Training
```bash
# Use mixed precision
# Edit training_config.json: "mixed_precision": "fp16"

# Enable gradient checkpointing
# Edit training_config.json: "gradient_checkpointing": true
```

### Poor Performance
```bash
# Increase dataset size
python scripts/train_coding_workflow.py --dataset-size 20000

# Train for more epochs
python scripts/train_coding_workflow.py --epochs 5

# Adjust learning rate
python scripts/train_coding_workflow.py --learning-rate 1e-4
```

## 📚 Advanced Features

### Multi-GPU Training
```bash
# Automatically uses both GPUs with tensor parallelism
python scripts/train_coding_workflow.py --base-model microsoft/CodeLlama-7b-Python-hf
```

### LoRA Fine-tuning
```bash
# Efficient fine-tuning with LoRA (default)
# Edit training_config.json: "use_lora": true
```

### Quantized Training
```bash
# 4-bit quantization for memory efficiency
python optimize_model.py --model_path models/fine-tuned --quantization 4bit
```

## 🎯 Next Steps

1. **Start with Quick Training** to test the pipeline
2. **Evaluate Results** and identify improvement areas
3. **Iterate on Dataset** with domain-specific data
4. **Scale Up Training** with larger datasets and more epochs
5. **Deploy and Monitor** in production environment

Your AI stack is now ready for coding-specific model training! 🚀
