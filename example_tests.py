#!/usr/bin/env python3
"""
Example test scenarios for Ollama parallel testing

This script demonstrates different ways to test your local LLM setup.
"""

import asyncio
import subprocess
import sys
from pathlib import Path


def run_test(description: str, command: list):
    """Run a test with description"""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    print()
    
    try:
        result = subprocess.run(command, check=True, capture_output=False)
        print(f"\n✅ Test completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
        return False


def main():
    """Run example test scenarios"""
    script_path = Path(__file__).parent / "test_ollama_parallel.py"
    
    if not script_path.exists():
        print("❌ test_ollama_parallel.py not found!")
        sys.exit(1)
    
    print("🚀 Ollama Parallel Testing Examples")
    print("\nMake sure Ollama is running: ollama serve")
    print("And you have a model available: ollama pull llama2")
    
    input("\nPress Enter to start the tests...")
    
    tests = [
        {
            "description": "Basic Functionality Test (5 requests, 2 concurrent)",
            "command": ["python", str(script_path), "--requests", "5", "--concurrency", "2"]
        },
        {
            "description": "Sequential Processing Test (5 requests, 1 concurrent)",
            "command": ["python", str(script_path), "--requests", "5", "--concurrency", "1"]
        },
        {
            "description": "High Concurrency Test (10 requests, 8 concurrent)",
            "command": ["python", str(script_path), "--requests", "10", "--concurrency", "8"]
        },
        {
            "description": "Coding Task Test",
            "command": [
                "python", str(script_path),
                "--requests", "3",
                "--concurrency", "2",
                "--prompt", "Write a Python function to calculate fibonacci numbers",
                "--max-tokens", "150"
            ]
        },
        {
            "description": "Creative Writing Test",
            "command": [
                "python", str(script_path),
                "--requests", "3",
                "--concurrency", "2",
                "--prompt", "Write a short story about a robot learning to paint",
                "--max-tokens", "200"
            ]
        },
        {
            "description": "Load Test with Results Export",
            "command": [
                "python", str(script_path),
                "--requests", "15",
                "--concurrency", "10",
                "--prompt", "Explain the concept of machine learning",
                "--save-results", "load_test_results.json"
            ]
        }
    ]
    
    successful_tests = 0
    total_tests = len(tests)
    
    for i, test in enumerate(tests, 1):
        print(f"\n📊 Running test {i}/{total_tests}")
        
        if run_test(test["description"], test["command"]):
            successful_tests += 1
        
        if i < total_tests:
            try:
                input("\nPress Enter to continue to next test (Ctrl+C to stop)...")
            except KeyboardInterrupt:
                print("\n⏹️  Testing stopped by user")
                break
    
    print(f"\n{'='*60}")
    print(f"TESTING SUMMARY")
    print(f"{'='*60}")
    print(f"Tests completed: {successful_tests}/{total_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests == total_tests:
        print("\n🎉 All tests completed successfully!")
        print("\n💡 Tips for further testing:")
        print("   - Try different models: --model codellama")
        print("   - Test with longer prompts and responses")
        print("   - Monitor system resources during high concurrency tests")
        print("   - Compare results across different hardware configurations")
    else:
        print("\n⚠️  Some tests failed. Check:")
        print("   - Is Ollama running? (ollama serve)")
        print("   - Is the model available? (ollama list)")
        print("   - Are there sufficient system resources?")


if __name__ == "__main__":
    main()