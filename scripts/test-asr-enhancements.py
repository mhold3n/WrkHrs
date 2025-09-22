#!/usr/bin/env python3
"""
ASR Enhancement Testing Script
Tests URL and base64 audio transcription functionality
"""

import asyncio
import sys
import os
import requests
import json
import base64
import tempfile
import time
from typing import Dict, Any, Optional
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


class ASRTester:
    """ASR Enhancement Testing Class"""
    
    def __init__(self, base_url: str = "http://localhost:8084"):
        self.base_url = base_url
        self.test_results = {}
    
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
    
    def check_asr_health(self) -> bool:
        """Check if ASR service is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                model_loaded = health_data.get("model_loaded", False)
                
                if model_loaded:
                    self.print_status("✅ ASR service is healthy and model is loaded", "success")
                    model_size = health_data.get("model_size", "unknown")
                    device = health_data.get("device", "unknown")
                    print(f"   Model: {model_size}, Device: {device}")
                    return True
                else:
                    self.print_status("❌ ASR service is running but model not loaded", "error")
                    return False
            else:
                self.print_status(f"❌ ASR service health check failed: {response.status_code}", "error")
                return False
        except Exception as e:
            self.print_status(f"❌ Could not connect to ASR service: {e}", "error")
            return False
    
    def create_test_audio_base64(self) -> str:
        """Create a minimal test audio file encoded as base64"""
        # Create a very simple WAV file (1 second of silence)
        # WAV header for 1 second of silence at 8kHz, 16-bit mono
        wav_header = b'RIFF'  # ChunkID
        wav_header += (36).to_bytes(4, 'little')  # ChunkSize  
        wav_header += b'WAVE'  # Format
        wav_header += b'fmt '  # Subchunk1ID
        wav_header += (16).to_bytes(4, 'little')  # Subchunk1Size
        wav_header += (1).to_bytes(2, 'little')   # AudioFormat (PCM)
        wav_header += (1).to_bytes(2, 'little')   # NumChannels (mono)
        wav_header += (8000).to_bytes(4, 'little')  # SampleRate
        wav_header += (16000).to_bytes(4, 'little')  # ByteRate
        wav_header += (2).to_bytes(2, 'little')   # BlockAlign
        wav_header += (16).to_bytes(2, 'little')  # BitsPerSample
        wav_header += b'data'  # Subchunk2ID
        wav_header += (0).to_bytes(4, 'little')   # Subchunk2Size (empty)
        
        return base64.b64encode(wav_header).decode('utf-8')
    
    def test_base64_transcription(self) -> Dict[str, Any]:
        """Test base64 audio transcription"""
        self.print_test_header("Base64 Audio Transcription")
        
        try:
            # Create test audio data
            base64_audio = self.create_test_audio_base64()
            
            self.print_status("🔍 Testing base64 audio transcription...", "test")
            
            # Test with data URL format
            data_url = f"data:audio/wav;base64,{base64_audio}"
            
            payload = {
                "audio_data": data_url,
                "language": None,
                "extract_technical": True
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/transcribe",
                json=payload,
                timeout=60
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                self.print_status("✅ Base64 transcription completed", "success")
                print(f"   Response time: {response_time:.2f}s")
                print(f"   Transcript: '{result.get('transcript', 'N/A')}'")
                print(f"   Segments: {len(result.get('segments', []))}")
                print(f"   Technical segments: {len(result.get('technical_segments', []))}")
                
                return {
                    "success": True,
                    "response_time": response_time,
                    "transcript": result.get('transcript', ''),
                    "segments": len(result.get('segments', [])),
                    "technical_segments": len(result.get('technical_segments', []))
                }
            else:
                error_detail = response.json().get('detail', 'Unknown error') if response.headers.get('content-type', '').startswith('application/json') else response.text
                self.print_status(f"❌ Base64 transcription failed: {response.status_code}", "error")
                print(f"   Error: {error_detail}")
                
                return {
                    "success": False,
                    "error": error_detail,
                    "status_code": response.status_code
                }
                
        except Exception as e:
            self.print_status(f"❌ Base64 transcription test error: {e}", "error")
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_url_transcription(self) -> Dict[str, Any]:
        """Test URL audio transcription"""
        self.print_test_header("URL Audio Transcription")
        
        # Test URLs (these are example URLs - in real testing you'd use actual audio URLs)
        test_urls = [
            {
                "url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav",
                "description": "Small WAV file"
            },
            {
                "url": "https://file-examples.com/storage/fe2b51d0fb66cbe072e26bf/2017/11/file_example_MP3_700KB.mp3", 
                "description": "Sample MP3 file"
            }
        ]
        
        results = {}
        
        for test_case in test_urls:
            url = test_case["url"]
            description = test_case["description"]
            
            self.print_status(f"🔍 Testing URL transcription: {description}", "test")
            print(f"   URL: {url}")
            
            try:
                payload = {
                    "audio_url": url,
                    "language": None,
                    "extract_technical": True
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/transcribe",
                    json=payload,
                    timeout=120  # Longer timeout for downloads
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    self.print_status(f"✅ URL transcription completed: {description}", "success")
                    print(f"   Response time: {response_time:.2f}s")
                    print(f"   Transcript: '{result.get('transcript', 'N/A')}'")
                    print(f"   Audio duration: {result.get('audio_duration', 0):.2f}s")
                    
                    results[description] = {
                        "success": True,
                        "url": url,
                        "response_time": response_time,
                        "transcript": result.get('transcript', ''),
                        "audio_duration": result.get('audio_duration', 0)
                    }
                else:
                    error_detail = response.json().get('detail', 'Unknown error') if response.headers.get('content-type', '').startswith('application/json') else response.text
                    self.print_status(f"❌ URL transcription failed: {description}", "error")
                    print(f"   Status: {response.status_code}")
                    print(f"   Error: {error_detail}")
                    
                    results[description] = {
                        "success": False,
                        "url": url,
                        "error": error_detail,
                        "status_code": response.status_code
                    }
                    
            except Exception as e:
                self.print_status(f"❌ URL transcription error: {description}", "error")
                print(f"   Exception: {e}")
                
                results[description] = {
                    "success": False,
                    "url": url,
                    "error": str(e)
                }
        
        return results
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling for invalid inputs"""
        self.print_test_header("Error Handling")
        
        test_cases = [
            {
                "name": "Invalid URL",
                "payload": {"audio_url": "not-a-valid-url"},
                "expected_status": 400
            },
            {
                "name": "Invalid base64",
                "payload": {"audio_data": "invalid-base64-data"},
                "expected_status": 400
            },
            {
                "name": "No audio input",
                "payload": {"language": "en"},
                "expected_status": 400
            },
            {
                "name": "Empty base64",
                "payload": {"audio_data": ""},
                "expected_status": 400
            }
        ]
        
        results = {}
        
        for test_case in test_cases:
            name = test_case["name"]
            payload = test_case["payload"]
            expected_status = test_case["expected_status"]
            
            self.print_status(f"🔍 Testing error case: {name}", "test")
            
            try:
                response = requests.post(
                    f"{self.base_url}/transcribe",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == expected_status:
                    self.print_status(f"✅ Error handling correct: {name}", "success")
                    results[name] = {"success": True, "status": response.status_code}
                else:
                    self.print_status(f"⚠️  Unexpected status for {name}: {response.status_code} (expected {expected_status})", "warning")
                    results[name] = {"success": False, "status": response.status_code, "expected": expected_status}
                    
            except Exception as e:
                self.print_status(f"❌ Error test failed: {name}", "error")
                results[name] = {"success": False, "error": str(e)}
        
        return results
    
    def test_technical_analysis(self) -> Dict[str, Any]:
        """Test technical text analysis endpoint"""
        self.print_test_header("Technical Analysis")
        
        test_texts = [
            {
                "text": "The steel beam has a Young's modulus of 200 GPa and yield strength of 250 MPa",
                "expected_technical": True
            },
            {
                "text": "Hello, how are you today? The weather is nice.",
                "expected_technical": False
            },
            {
                "text": "The chemical reaction produces methane and carbon dioxide with a catalyst",
                "expected_technical": True
            }
        ]
        
        results = {}
        
        for i, test_case in enumerate(test_texts):
            text = test_case["text"]
            expected_technical = test_case["expected_technical"]
            
            self.print_status(f"🔍 Testing technical analysis {i+1}...", "test")
            
            try:
                response = requests.post(
                    f"{self.base_url}/technical/analyze",
                    params={"text": text},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    is_technical = result.get("is_technical", False)
                    score = result.get("technical_score", 0)
                    
                    if is_technical == expected_technical:
                        self.print_status(f"✅ Technical analysis correct", "success")
                    else:
                        self.print_status(f"⚠️  Technical analysis mismatch", "warning")
                    
                    print(f"   Text: '{text[:50]}...'")
                    print(f"   Score: {score:.3f}")
                    print(f"   Is technical: {is_technical} (expected: {expected_technical})")
                    print(f"   Keywords: {len(result.get('matching_keywords', []))}")
                    
                    results[f"test_{i+1}"] = {
                        "success": True,
                        "text": text,
                        "score": score,
                        "is_technical": is_technical,
                        "expected": expected_technical,
                        "correct": is_technical == expected_technical
                    }
                else:
                    self.print_status(f"❌ Technical analysis failed: {response.status_code}", "error")
                    results[f"test_{i+1}"] = {"success": False, "status": response.status_code}
                    
            except Exception as e:
                self.print_status(f"❌ Technical analysis error: {e}", "error")
                results[f"test_{i+1}"] = {"success": False, "error": str(e)}
        
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
            
            if test_name == "base64_transcription":
                success = result.get("success", False)
            elif test_name == "url_transcription":
                success = any(test.get("success", False) for test in result.values()) if isinstance(result, dict) else False
            elif test_name == "error_handling":
                success = all(test.get("success", False) for test in result.values()) if isinstance(result, dict) else False
            elif test_name == "technical_analysis":
                success = all(test.get("success", False) for test in result.values()) if isinstance(result, dict) else False
            else:
                success = result.get("success", False) if isinstance(result, dict) else False
            
            if success:
                passed_tests += 1
                self.print_status(f"✅ {test_name.upper()}: PASSED", "success")
            else:
                self.print_status(f"❌ {test_name.upper()}: FAILED", "error")
        
        print(f"\n{Colors.BLUE}Overall: {passed_tests}/{total_tests} test suites passed{Colors.NC}")
        
        if passed_tests == total_tests:
            self.print_status("🎉 ALL TESTS PASSED! ASR enhancements working correctly.", "success")
        elif passed_tests > 0:
            self.print_status(f"⚠️  PARTIAL SUCCESS: {passed_tests}/{total_tests} test suites passed.", "warning")
        else:
            self.print_status("❌ ALL TESTS FAILED: ASR enhancements need attention.", "error")
        
        print(f"\n{Colors.YELLOW}Note: URL tests may fail if test URLs are not accessible.{Colors.NC}")
        print(f"{Colors.YELLOW}Base64 and error handling tests are more reliable indicators.{Colors.NC}")
    
    async def run_all_tests(self):
        """Run all ASR enhancement tests"""
        print(f"{Colors.BLUE}{'='*60}{Colors.NC}")
        print(f"{Colors.BLUE}ASR Enhancement Test Suite{Colors.NC}")
        print(f"{Colors.BLUE}{'='*60}{Colors.NC}")
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check if ASR service is available
        if not self.check_asr_health():
            self.print_status("❌ ASR service not available. Tests cannot proceed.", "error")
            return
        
        # Run tests
        self.test_results["base64_transcription"] = self.test_base64_transcription()
        self.test_results["url_transcription"] = self.test_url_transcription()
        self.test_results["error_handling"] = self.test_error_handling()
        self.test_results["technical_analysis"] = self.test_technical_analysis()
        
        # Print summary
        self.print_summary()


async def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ASR enhancements")
    parser.add_argument("--url", default="http://localhost:8084", help="ASR service URL")
    parser.add_argument("--test", choices=["base64", "url", "error", "technical", "all"], 
                       default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    tester = ASRTester(args.url)
    
    if args.test == "all":
        await tester.run_all_tests()
    else:
        if not tester.check_asr_health():
            print("ASR service not available")
            return
        
        if args.test == "base64":
            result = tester.test_base64_transcription()
        elif args.test == "url":
            result = tester.test_url_transcription()
        elif args.test == "error":
            result = tester.test_error_handling()
        elif args.test == "technical":
            result = tester.test_technical_analysis()
        
        print(f"\nTest result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.NC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Test suite error: {e}{Colors.NC}")
        sys.exit(1)
