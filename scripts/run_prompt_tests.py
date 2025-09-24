#!/usr/bin/env python3
"""
Prompt Middleware Test Runner
Runs comprehensive tests for the prompt middleware suite
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

def run_tests(test_type="all", verbose=True, coverage=False):
    """Run tests based on type"""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=services/prompt_middleware", "--cov-report=html", "--cov-report=term"])
    
    # Select tests based on type
    if test_type == "all":
        cmd.extend(["tests/test_prompt_middleware.py", "tests/test_object_inference.py"])
    elif test_type == "unit":
        cmd.extend(["tests/test_prompt_middleware.py", "-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["tests/test_prompt_middleware.py", "-m", "integration"])
    elif test_type == "object_inference":
        cmd.append("tests/test_object_inference.py")
    elif test_type == "voice":
        cmd.extend(["-m", "voice"])
    elif test_type == "geometric":
        cmd.extend(["-m", "geometric"])
    elif test_type == "ai_integration":
        cmd.extend(["-m", "ai_integration"])
    elif test_type == "demo":
        # Run the demo instead of tests
        return run_demo()
    else:
        print(f"Unknown test type: {test_type}")
        return False
    
    # Add test discovery
    cmd.extend(["--tb=short", "--color=yes"])
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run the tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0

def run_demo():
    """Run the object inference demo"""
    print("Running Object Inference Demo...")
    print("=" * 60)
    
    cmd = ["python", "scripts/test_object_inference_demo.py"]
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "pytest",
        "numpy",
        "scipy",
        "torch",
        "transformers",
        "librosa",
        "networkx",
        "matplotlib"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Run prompt middleware tests")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "object_inference", "voice", "geometric", "ai_integration", "demo"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--no-verbose", 
        action="store_true",
        help="Run tests without verbose output"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--check-deps", 
        action="store_true",
        help="Check dependencies and exit"
    )
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        return check_dependencies()
    
    # Check dependencies before running tests
    if not check_dependencies():
        return False
    
    # Run tests
    success = run_tests(
        test_type=args.type,
        verbose=not args.no_verbose,
        coverage=args.coverage
    )
    
    if success:
        print("\n🎉 All tests passed!")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
