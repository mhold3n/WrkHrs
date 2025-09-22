"""
LLM Backend Integration for AI Stack Orchestrator
Supports Ollama and vLLM backends with unified interface
"""

import os
import logging
import requests
import asyncio
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMBackendError(Exception):
    """Custom exception for LLM backend errors"""
    pass


class LLMBackend(ABC):
    """Abstract base class for LLM backends"""
    
    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url
        self.timeout = timeout
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is healthy"""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[str]:
        """List available models"""
        pass
    
    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling"""
        try:
            kwargs.setdefault('timeout', self.timeout)
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            raise LLMBackendError(f"{self.name} request timed out")
        except requests.exceptions.ConnectionError:
            raise LLMBackendError(f"Cannot connect to {self.name} at {self.base_url}")
        except requests.exceptions.HTTPError as e:
            raise LLMBackendError(f"{self.name} HTTP error: {e}")
        except Exception as e:
            raise LLMBackendError(f"{self.name} unexpected error: {e}")


class OllamaBackend(LLMBackend):
    """Ollama LLM backend implementation"""
    
    def __init__(self, base_url: str = "http://llm-runner:11434", timeout: int = 60):
        super().__init__(base_url, timeout)
        self.model = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct")
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion using Ollama API"""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "num_predict": kwargs.get("max_tokens", 1000),
                }
            }
            
            logger.debug(f"Sending Ollama request: {payload}")
            
            response = self._make_request(
                "POST", 
                f"{self.base_url}/api/chat", 
                json=payload
            )
            
            result = response.json()
            logger.debug(f"Ollama response: {result}")
            
            # Transform Ollama response to OpenAI format
            content = result.get("message", {}).get("content", "")
            
            return {
                "id": f"ollama-{datetime.utcnow().timestamp()}",
                "object": "chat.completion",
                "created": int(datetime.utcnow().timestamp()),
                "model": self.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Ollama chat completion error: {e}")
            raise LLMBackendError(f"Ollama chat completion failed: {e}")
    
    async def health_check(self) -> bool:
        """Check Ollama health"""
        try:
            response = self._make_request("GET", f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False
    
    async def list_models(self) -> List[str]:
        """List Ollama models"""
        try:
            response = self._make_request("GET", f"{self.base_url}/api/tags")
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return models
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    async def pull_model(self, model_name: str = None) -> bool:
        """Pull/download a model"""
        model_to_pull = model_name or self.model
        try:
            payload = {"name": model_to_pull}
            response = self._make_request(
                "POST", 
                f"{self.base_url}/api/pull", 
                json=payload
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull Ollama model {model_to_pull}: {e}")
            return False


class VLLMBackend(LLMBackend):
    """vLLM OpenAI-compatible backend implementation"""
    
    def __init__(self, base_url: str = "http://llm-runner:8000", timeout: int = 60):
        super().__init__(base_url, timeout)
        self.model = os.getenv("VLLM_MODEL", "default")
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion using vLLM OpenAI-compatible API"""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000),
                "stream": False
            }
            
            logger.debug(f"Sending vLLM request: {payload}")
            
            response = self._make_request(
                "POST", 
                f"{self.base_url}/v1/chat/completions", 
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            result = response.json()
            logger.debug(f"vLLM response: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"vLLM chat completion error: {e}")
            raise LLMBackendError(f"vLLM chat completion failed: {e}")
    
    async def health_check(self) -> bool:
        """Check vLLM health"""
        try:
            response = self._make_request("GET", f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            # Try alternative health endpoint
            try:
                response = self._make_request("GET", f"{self.base_url}/v1/models")
                return response.status_code == 200
            except Exception as e:
                logger.warning(f"vLLM health check failed: {e}")
                return False
    
    async def list_models(self) -> List[str]:
        """List vLLM models"""
        try:
            response = self._make_request("GET", f"{self.base_url}/v1/models")
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            return models
        except Exception as e:
            logger.error(f"Failed to list vLLM models: {e}")
            return []


class LLMManager:
    """Manager class for LLM backends"""
    
    def __init__(self):
        self.backend_type = os.getenv("LLM_BACKEND", "ollama").lower()
        self.backend = self._create_backend()
        logger.info(f"Initialized LLM Manager with backend: {self.backend_type}")
    
    def _create_backend(self) -> LLMBackend:
        """Create appropriate backend based on configuration"""
        if self.backend_type == "ollama":
            return OllamaBackend()
        elif self.backend_type == "vllm":
            return VLLMBackend()
        else:
            logger.warning(f"Unknown backend type: {self.backend_type}, defaulting to Ollama")
            return OllamaBackend()
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion using configured backend"""
        return await self.backend.chat_completion(messages, **kwargs)
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        start_time = datetime.utcnow()
        
        try:
            is_healthy = await self.backend.health_check()
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            health_data = {
                "backend": self.backend_type,
                "healthy": is_healthy,
                "response_time": response_time,
                "timestamp": start_time.isoformat()
            }
            
            if is_healthy:
                try:
                    models = await self.backend.list_models()
                    health_data["available_models"] = models
                    health_data["model_count"] = len(models)
                except Exception as e:
                    logger.warning(f"Could not fetch models during health check: {e}")
                    health_data["models_error"] = str(e)
            
            return health_data
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "backend": self.backend_type,
                "healthy": False,
                "error": str(e),
                "response_time": (datetime.utcnow() - start_time).total_seconds(),
                "timestamp": start_time.isoformat()
            }
    
    async def list_models(self) -> List[str]:
        """List available models"""
        return await self.backend.list_models()
    
    async def switch_backend(self, backend_type: str) -> bool:
        """Switch to different backend"""
        try:
            old_backend = self.backend_type
            self.backend_type = backend_type.lower()
            self.backend = self._create_backend()
            
            # Test new backend
            if await self.backend.health_check():
                logger.info(f"Successfully switched from {old_backend} to {self.backend_type}")
                return True
            else:
                # Revert on failure
                self.backend_type = old_backend
                self.backend = self._create_backend()
                logger.error(f"Failed to switch to {backend_type}, reverted to {old_backend}")
                return False
                
        except Exception as e:
            logger.error(f"Error switching backend: {e}")
            return False
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get backend information"""
        return {
            "type": self.backend_type,
            "base_url": self.backend.base_url,
            "timeout": self.backend.timeout,
            "name": self.backend.name
        }


# Global LLM manager instance
llm_manager = LLMManager()
