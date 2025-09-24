# vLLM Multi-GPU Setup Guide

## Overview

This guide explains how to configure vLLM for multi-GPU inference on your hardware setup:
- **RTX 4070 Ti** (12GB VRAM) - Primary GPU
- **RTX 2080** (8GB VRAM) - Secondary GPU
- **Total VRAM**: 20GB across both GPUs

## Key vLLM Features for Multi-GPU

### 1. Tensor Parallelism
- **What it does**: Splits model layers across multiple GPUs
- **Configuration**: `--tensor-parallel-size 2`
- **Best for**: Large models that don't fit on single GPU
- **Your setup**: Perfect for 7B-13B models

### 2. Pipeline Parallelism
- **What it does**: Splits model into stages across GPUs
- **Configuration**: `--pipeline-parallel-size 2`
- **Best for**: Very large models (70B+)
- **Your setup**: Not needed for most models

### 3. Memory Optimization
- **GPU Memory Utilization**: `--gpu-memory-utilization 0.85`
- **Max Model Length**: `--max-model-len 8192`
- **Data Type**: `--dtype auto` (automatically selects best precision)

## Model Recommendations

Based on your 20GB total VRAM:

| Model | Size | Memory Usage | Tensor Parallel | Performance |
|-------|------|--------------|-----------------|-------------|
| **Llama-2-7b-chat** | 13GB | ~11GB | 2 GPUs | ⭐⭐⭐⭐⭐ |
| **Llama-2-13b-chat** | 26GB | ~22GB | 2 GPUs | ⭐⭐⭐⭐ |
| **Mistral-7B-Instruct** | 13GB | ~11GB | 2 GPUs | ⭐⭐⭐⭐⭐ |
| **CodeLlama-7b** | 13GB | ~11GB | 2 GPUs | ⭐⭐⭐⭐ |

## Configuration Files

### Production Docker Compose (`compose/docker-compose.prod.yml`)
```yaml
llm-runner:
  image: vllm/vllm-openai:latest
  ports: [ "8001:8000" ]
  environment:
    - NVIDIA_VISIBLE_DEVICES=all
    - CUDA_VISIBLE_DEVICES=all
    - VLLM_GPU_MEMORY_UTILIZATION=0.85
    - VLLM_MAX_MODEL_LEN=8192
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
  command: [
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "${VLLM_MODEL}",
    "--trust-remote-code",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--tensor-parallel-size", "2",
    "--gpu-memory-utilization", "0.85",
    "--max-model-len", "8192",
    "--dtype", "auto",
    "--enforce-eager"
  ]
```

### Environment Variables (`.env`)
```bash
# vLLM Configuration
LLM_BACKEND=vllm
VLLM_MODEL=meta-llama/Llama-2-7b-chat-hf
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_MAX_MODEL_LEN=8192

# GPU Configuration
ENABLE_GPU=true
NVIDIA_VISIBLE_DEVICES=all
```

## Setup Steps

### 1. Download Models
```bash
# Use the setup script
python scripts/setup-vllm-models.py --auto

# Or manually specify a model
python scripts/setup-vllm-models.py --model meta-llama/Llama-2-7b-chat-hf
```

### 2. Start vLLM Service
```bash
# Start production environment with vLLM
docker compose -f compose/docker-compose.base.yml -f compose/docker-compose.prod.yml up -d llm-runner
```

### 3. Verify Multi-GPU Usage
```bash
# Check GPU utilization
nvidia-smi

# Test vLLM endpoint
curl http://localhost:8001/health
```

## Performance Optimization

### Memory Settings
- **GPU Memory Utilization**: 0.85 (85% of available VRAM)
- **Max Model Length**: 8192 tokens
- **Data Type**: Auto (FP16/BF16 for efficiency)

### Parallelism Settings
- **Tensor Parallel Size**: 2 (for your 2 GPUs)
- **Pipeline Parallel Size**: 1 (not needed for most models)

### Advanced Options
```bash
# For maximum throughput
--max-num-seqs 256
--max-num-batched-tokens 8192

# For low latency
--max-num-seqs 1
--max-num-batched-tokens 1024

# For memory efficiency
--swap-space 4
--cpu-offload-gb 4
```

## Monitoring and Debugging

### GPU Monitoring
```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Detailed GPU info
nvidia-smi -q
```

### vLLM Logs
```bash
# Check vLLM container logs
docker logs ai-stack-llm-runner-1 -f

# Check for tensor parallelism
docker logs ai-stack-llm-runner-1 | grep -i "tensor"
```

### Performance Testing
```bash
# Test inference speed
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "prompt": "Hello, how are you?",
    "max_tokens": 100
  }'
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `--gpu-memory-utilization` to 0.7
   - Use smaller model
   - Reduce `--max-model-len`

2. **Tensor Parallelism Not Working**
   - Check `nvidia-smi` shows both GPUs
   - Verify `--tensor-parallel-size 2`
   - Check Docker has access to all GPUs

3. **Slow Performance**
   - Increase `--max-num-seqs`
   - Use `--dtype float16`
   - Check GPU utilization with `nvidia-smi`

### GPU Memory Distribution
With tensor parallelism, memory usage will be:
- **GPU 0 (RTX 4070 Ti)**: ~11GB for model layers
- **GPU 1 (RTX 2080)**: ~11GB for model layers
- **Remaining**: Available for inference cache

## Scaling to More GPUs

When you add more GPUs:

1. **Update tensor parallel size**:
   ```bash
   --tensor-parallel-size 4  # For 4 GPUs
   ```

2. **Adjust memory utilization**:
   ```bash
   --gpu-memory-utilization 0.9  # Can use more with more GPUs
   ```

3. **Consider pipeline parallelism** for very large models:
   ```bash
   --pipeline-parallel-size 2
   --tensor-parallel-size 2
   ```

## Next Steps

1. **Test with different models** to find optimal performance
2. **Benchmark throughput** with your specific use cases
3. **Monitor GPU utilization** to ensure both GPUs are being used
4. **Scale up** by adding more GPUs as needed

Your setup is perfectly configured for high-performance multi-GPU inference with vLLM! 🚀
