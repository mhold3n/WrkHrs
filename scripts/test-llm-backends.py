#!/usr/bin/env python3
"""
LLM Backend Integration Testing Script
Tests both Ollama and vLLM backends for AI Stack
"""

import asyncio
import sys
import os
import requests
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory to path to import llm_backends
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'orchestrator'))

try:
    from llm_backends import OllamaBackend, VLLMBackend, LLMManager, LLMBackendError
except ImportError as e:
    print(f"❌ Failed to import LLM backends: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


class LLMTester:
    """LLM Backend Testing Class"""
    
    def __init__(self):
        self.test_results = {}
        self.test_messages = [
            {"role": "user", "content": "Hello! Please respond with exactly 'Test successful' and nothing else."}
        ]
    
    def print_status(self, message: str, status: str = "info"):
        """Print colored status message"""
        color_map = {
            "info": Colors.BLUE,
            "success": Colors.GREEN,
            "warning": Colors.YELLOW,
            "error": Colors.RED,
            "test": Colors.PURPLE
        }
        color = color_map.get(status, Colors.NC)
        print(f"{color}{message}{Colors.NC}")
    
    def print_test_header(self, test_name: str):
        """Print test section header"""
        print(f"\n{Colors.CYAN}{'='*60}{Colors.NC}")
        print(f"{Colors.CYAN}Testing: {test_name}{Colors.NC}")
        print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
    
    async def test_backend_health(self, backend, backend_name: str) -> bool:
        """Test backend health check"""
        self.print_status(f"🔍 Testing {backend_name} health check...", "test")
        
        try:
            is_healthy = await backend.health_check()
            if is_healthy:
                self.print_status(f"✅ {backend_name} health check passed", "success")
                return True
            else:
                self.print_status(f"❌ {backend_name} health check failed", "error")
                return False
        except Exception as e:
            self.print_status(f"❌ {backend_name} health check error: {e}", "error")
            return False
    
    async def test_backend_models(self, backend, backend_name: str) -> List[str]:
        """Test backend model listing"""
        self.print_status(f"📋 Testing {backend_name} model listing...", "test")
        
        try:
            models = await backend.list_models()
            if models:
                self.print_status(f"✅ {backend_name} found {len(models)} models", "success")
                for model in models[:3]:  # Show first 3 models
                    print(f"   • {model}")
                if len(models) > 3:
                    print(f"   ... and {len(models) - 3} more")
                return models
            else:
                self.print_status(f"⚠️  {backend_name} returned no models", "warning")
                return []
        except Exception as e:
            self.print_status(f"❌ {backend_name} model listing error: {e}", "error")
            return []
    
    async def test_backend_chat(self, backend, backend_name: str) -> Dict[str, Any]:
        """Test backend chat completion"""
        self.print_status(f"💬 Testing {backend_name} chat completion...", "test")
        
        try:
            start_time = time.time()
            result = await backend.chat_completion(self.test_messages, max_tokens=50)
            response_time = time.time() - start_time
            
            if result and "choices" in result:
                choices = result.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    self.print_status(f"✅ {backend_name} chat completion successful", "success")
                    print(f"   Response: '{content}'")
                    print(f"   Response time: {response_time:.2f}s")
                    
                    return {
                        "success": True,
                        "response": content,
                        "response_time": response_time,
                        "full_result": result
                    }
                else:
                    self.print_status(f"❌ {backend_name} returned empty choices", "error")
                    return {"success": False, "error": "Empty choices"}
            else:
                self.print_status(f"❌ {backend_name} invalid response format", "error")
                return {"success": False, "error": "Invalid response format"}
                
        except LLMBackendError as e:
            self.print_status(f"❌ {backend_name} backend error: {e}", "error")
            return {"success": False, "error": str(e)}
        except Exception as e:
            self.print_status(f"❌ {backend_name} unexpected error: {e}", "error")
            return {"success": False, "error": str(e)}
    
    async def test_ollama_backend(self) -> Dict[str, Any]:
        """Test Ollama backend"""
        self.print_test_header("Ollama Backend")
        
        # Test different connection URLs
        urls_to_try = [
            "http://llm-runner:11434",  # Docker compose
            "http://localhost:11434",   # Local development
        ]
        
        results = {"tested": False, "healthy": False, "models": [], "chat": {}}
        
        for url in urls_to_try:
            self.print_status(f"🔗 Trying Ollama at {url}...", "info")
            
            try:
                backend = OllamaBackend(base_url=url, timeout=10)
                
                # Test health
                healthy = await self.test_backend_health(backend, f"Ollama ({url})")
                if not healthy:
                    continue
                
                results["tested"] = True
                results["healthy"] = True
                results["url"] = url
                
                # Test models
                models = await self.test_backend_models(backend, f"Ollama ({url})")
                results["models"] = models
                
                # Test chat (only if models are available)
                if models:
                    chat_result = await self.test_backend_chat(backend, f"Ollama ({url})")
                    results["chat"] = chat_result
                else:
                    self.print_status("⚠️  Skipping chat test - no models available", "warning")
                    self.print_status("💡 Try running: make pull-models", "info")
                
                break  # Success, no need to try other URLs
                
            except Exception as e:
                self.print_status(f"❌ Failed to connect to {url}: {e}", "error")
                continue
        
        if not results["tested"]:
            self.print_status("❌ Could not connect to any Ollama instance", "error")
            self.print_status("💡 Make sure Ollama is running and accessible", "info")
        
        return results
    
    async def test_vllm_backend(self) -> Dict[str, Any]:
        """Test vLLM backend"""
        self.print_test_header("vLLM Backend")
        
        # Test different connection URLs
        urls_to_try = [
            "http://llm-runner:8000",   # Docker compose
            "http://localhost:8001",    # Local development (prod port)
        ]
        
        results = {"tested": False, "healthy": False, "models": [], "chat": {}}
        
        for url in urls_to_try:
            self.print_status(f"🔗 Trying vLLM at {url}...", "info")
            
            try:
                backend = VLLMBackend(base_url=url, timeout=10)
                
                # Test health
                healthy = await self.test_backend_health(backend, f"vLLM ({url})")
                if not healthy:
                    continue
                
                results["tested"] = True
                results["healthy"] = True
                results["url"] = url
                
                # Test models
                models = await self.test_backend_models(backend, f"vLLM ({url})")
                results["models"] = models
                
                # Test chat
                chat_result = await self.test_backend_chat(backend, f"vLLM ({url})")
                results["chat"] = chat_result
                
                break  # Success, no need to try other URLs
                
            except Exception as e:
                self.print_status(f"❌ Failed to connect to {url}: {e}", "error")
                continue
        
        if not results["tested"]:
            self.print_status("❌ Could not connect to any vLLM instance", "error")
            self.print_status("💡 Make sure vLLM is running in production mode", "info")
        
        return results
    
    async def test_llm_manager(self) -> Dict[str, Any]:
        """Test LLM Manager functionality"""
        self.print_test_header("LLM Manager")
        
        results = {"tested": False, "manager_healthy": False, "switch_test": {}}
        
        try:
            # Create manager with environment settings
            manager = LLMManager()
            self.print_status(f"📊 LLM Manager initialized with backend: {manager.backend_type}", "info")
            
            # Test current backend health
            health_info = await manager.health_check()
            self.print_status(f"🔍 Testing manager health check...", "test")
            
            if health_info.get("healthy", False):
                self.print_status("✅ LLM Manager health check passed", "success")
                results["manager_healthy"] = True
            else:
                self.print_status(f"❌ LLM Manager health check failed: {health_info.get('error', 'Unknown')}", "error")
            
            results["tested"] = True
            results["current_backend"] = manager.get_backend_info()
            results["health_info"] = health_info
            
            # Test backend switching if both backends are available
            self.print_status("🔄 Testing backend switching...", "test")
            current_backend = manager.backend_type
            target_backend = "vllm" if current_backend == "ollama" else "ollama"
            
            switch_success = await manager.switch_backend(target_backend)
            if switch_success:
                self.print_status(f"✅ Successfully switched to {target_backend}", "success")
                
                # Switch back
                switch_back = await manager.switch_backend(current_backend)
                if switch_back:
                    self.print_status(f"✅ Successfully switched back to {current_backend}", "success")
                    results["switch_test"] = {"success": True}
                else:
                    self.print_status(f"❌ Failed to switch back to {current_backend}", "error")
                    results["switch_test"] = {"success": False, "error": "Failed to switch back"}
            else:
                self.print_status(f"❌ Failed to switch to {target_backend}", "error")
                results["switch_test"] = {"success": False, "error": f"Failed to switch to {target_backend}"}
            
        except Exception as e:
            self.print_status(f"❌ LLM Manager test error: {e}", "error")
            results["error"] = str(e)
        
        return results
    
    async def test_orchestrator_endpoints(self) -> Dict[str, Any]:
        """Test orchestrator LLM endpoints"""
        self.print_test_header("Orchestrator LLM Endpoints")
        
        results = {"tested": False, "endpoints": {}}
        
        # Test different orchestrator URLs
        urls_to_try = [
            "http://orchestrator:8000",  # Docker compose
            "http://localhost:8081",     # Local development
        ]
        
        for base_url in urls_to_try:
            self.print_status(f"🔗 Trying orchestrator at {base_url}...", "info")
            
            try:
                # Test health endpoint
                health_url = f"{base_url}/health"
                response = requests.get(health_url, timeout=5)
                
                if response.status_code == 200:
                    self.print_status(f"✅ Orchestrator health check passed", "success")
                    results["tested"] = True
                    results["base_url"] = base_url
                    
                    # Test LLM info endpoint
                    try:
                        llm_info_url = f"{base_url}/llm/info"
                        llm_response = requests.get(llm_info_url, timeout=10)
                        
                        if llm_response.status_code == 200:
                            llm_data = llm_response.json()
                            self.print_status("✅ LLM info endpoint working", "success")
                            results["endpoints"]["llm_info"] = {"success": True, "data": llm_data}
                            
                            backend_type = llm_data.get("backend_info", {}).get("type", "unknown")
                            is_healthy = llm_data.get("health", {}).get("healthy", False)
                            print(f"   Backend: {backend_type}")
                            print(f"   Healthy: {is_healthy}")
                        else:
                            self.print_status(f"❌ LLM info endpoint failed: {llm_response.status_code}", "error")
                            results["endpoints"]["llm_info"] = {"success": False, "status": llm_response.status_code}
                    except Exception as e:
                        self.print_status(f"❌ LLM info endpoint error: {e}", "error")
                        results["endpoints"]["llm_info"] = {"success": False, "error": str(e)}
                    
                    # Test LLM test endpoint
                    try:
                        llm_test_url = f"{base_url}/llm/test"
                        test_response = requests.post(llm_test_url, timeout=30)
                        
                        if test_response.status_code == 200:
                            test_data = test_response.json()
                            if test_data.get("success", False):
                                self.print_status("✅ LLM test endpoint working", "success")
                                response_text = test_data.get("test_response", "")
                                print(f"   Test response: '{response_text}'")
                                results["endpoints"]["llm_test"] = {"success": True, "data": test_data}
                            else:
                                self.print_status(f"❌ LLM test failed: {test_data.get('error', 'Unknown')}", "error")
                                results["endpoints"]["llm_test"] = {"success": False, "error": test_data.get('error')}
                        else:
                            self.print_status(f"❌ LLM test endpoint failed: {test_response.status_code}", "error")
                            results["endpoints"]["llm_test"] = {"success": False, "status": test_response.status_code}
                    except Exception as e:
                        self.print_status(f"❌ LLM test endpoint error: {e}", "error")
                        results["endpoints"]["llm_test"] = {"success": False, "error": str(e)}
                    
                    break  # Success, no need to try other URLs
                
            except Exception as e:
                self.print_status(f"❌ Failed to connect to {base_url}: {e}", "error")
                continue
        
        if not results["tested"]:
            self.print_status("❌ Could not connect to orchestrator", "error")
            self.print_status("💡 Make sure the orchestrator service is running", "info")
        
        return results
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{Colors.CYAN}{'='*60}{Colors.NC}")
        print(f"{Colors.CYAN}TEST SUMMARY{Colors.NC}")
        print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, result in self.test_results.items():
            total_tests += 1
            if result.get("tested", False):
                if test_name == "ollama":
                    success = result.get("healthy", False) and result.get("chat", {}).get("success", False)
                elif test_name == "vllm":
                    success = result.get("healthy", False) and result.get("chat", {}).get("success", False)
                elif test_name == "manager":
                    success = result.get("manager_healthy", False)
                elif test_name == "orchestrator":
                    success = result.get("tested", False) and result.get("endpoints", {}).get("llm_test", {}).get("success", False)
                else:
                    success = True
                
                if success:
                    passed_tests += 1
                    self.print_status(f"✅ {test_name.upper()}: PASSED", "success")
                else:
                    self.print_status(f"❌ {test_name.upper()}: FAILED", "error")
            else:
                self.print_status(f"⚠️  {test_name.upper()}: NOT TESTED", "warning")
        
        print(f"\n{Colors.BLUE}Overall: {passed_tests}/{total_tests} tests passed{Colors.NC}")
        
        if passed_tests == total_tests:
            self.print_status("🎉 ALL TESTS PASSED! LLM integration is working correctly.", "success")
        elif passed_tests > 0:
            self.print_status(f"⚠️  PARTIAL SUCCESS: {passed_tests}/{total_tests} backends working.", "warning")
        else:
            self.print_status("❌ ALL TESTS FAILED: LLM integration needs attention.", "error")
        
        # Print recommendations
        print(f"\n{Colors.YELLOW}Recommendations:{Colors.NC}")
        if not self.test_results.get("ollama", {}).get("tested", False):
            print("• Install and start Ollama for development")
        if not self.test_results.get("vllm", {}).get("tested", False):
            print("• Set up vLLM for production GPU inference")
        if not self.test_results.get("orchestrator", {}).get("tested", False):
            print("• Make sure orchestrator service is running")
        
        print(f"\n{Colors.BLUE}Next steps:{Colors.NC}")
        print("• Run: make up-dev (to start development environment)")
        print("• Run: make pull-models (to download Ollama models)")
        print("• Run: make health (to check all services)")
    
    async def run_all_tests(self):
        """Run all LLM backend tests"""
        print(f"{Colors.BLUE}{'='*60}{Colors.NC}")
        print(f"{Colors.BLUE}LLM Backend Integration Test Suite{Colors.NC}")
        print(f"{Colors.BLUE}{'='*60}{Colors.NC}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test individual backends
        self.test_results["ollama"] = await self.test_ollama_backend()
        self.test_results["vllm"] = await self.test_vllm_backend()
        
        # Test LLM manager
        self.test_results["manager"] = await self.test_llm_manager()
        
        # Test orchestrator endpoints
        self.test_results["orchestrator"] = await self.test_orchestrator_endpoints()
        
        # Print summary
        self.print_summary()


async def main():
    """Main test function"""
    tester = LLMTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.NC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Test suite error: {e}{Colors.NC}")
        sys.exit(1)
