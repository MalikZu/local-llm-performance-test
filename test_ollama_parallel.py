#!/usr/bin/env python3
"""
Ollama Parallel Testing Script

This script sends multiple parallel requests to an Ollama server to test
local LLM performance and response times.
"""

import asyncio
import aiohttp
import time
import json
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass
from statistics import mean, median


@dataclass
class RequestResult:
    """Store the result of a single request"""
    request_id: int
    success: bool
    response_time: float
    response_text: str = ""
    error_message: str = ""
    tokens_generated: int = 0


class OllamaParallelTester:
    """Test Ollama server with parallel requests"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def send_request(self, request_id: int, model: str, prompt: str, 
                          max_tokens: int = 100) -> RequestResult:
        """Send a single request to Ollama"""
        start_time = time.time()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get("response", "")
                    
                    # Estimate tokens (rough approximation)
                    tokens_generated = len(response_text.split())
                    
                    return RequestResult(
                        request_id=request_id,
                        success=True,
                        response_time=response_time,
                        response_text=response_text,
                        tokens_generated=tokens_generated
                    )
                else:
                    error_text = await response.text()
                    return RequestResult(
                        request_id=request_id,
                        success=False,
                        response_time=response_time,
                        error_message=f"HTTP {response.status}: {error_text}"
                    )
                    
        except asyncio.TimeoutError:
            return RequestResult(
                request_id=request_id,
                success=False,
                response_time=time.time() - start_time,
                error_message="Request timeout"
            )
        except Exception as e:
            return RequestResult(
                request_id=request_id,
                success=False,
                response_time=time.time() - start_time,
                error_message=f"Error: {str(e)}"
            )
    
    async def run_parallel_test(self, model: str, prompt: str, 
                               num_requests: int, concurrency: int,
                               max_tokens: int = 100) -> List[RequestResult]:
        """Run multiple parallel requests"""
        print(f"Starting test with {num_requests} requests, {concurrency} concurrent")
        print(f"Model: {model}")
        print(f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Prompt: {prompt}")
        print("-" * 60)
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_request(request_id: int) -> RequestResult:
            async with semaphore:
                return await self.send_request(request_id, model, prompt, max_tokens)
        
        # Create and run all tasks
        tasks = [limited_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(RequestResult(
                    request_id=i,
                    success=False,
                    response_time=0,
                    error_message=f"Task exception: {str(result)}"
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def print_statistics(self, results: List[RequestResult]):
        """Print test statistics"""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        
        print(f"Total requests: {len(results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        print(f"Success rate: {len(successful_results)/len(results)*100:.1f}%")
        
        if successful_results:
            response_times = [r.response_time for r in successful_results]
            tokens_generated = [r.tokens_generated for r in successful_results]
            
            print("\nResponse Time Statistics:")
            print(f"  Mean: {mean(response_times):.2f}s")
            print(f"  Median: {median(response_times):.2f}s")
            print(f"  Min: {min(response_times):.2f}s")
            print(f"  Max: {max(response_times):.2f}s")
            
            if tokens_generated:
                print("\nToken Generation Statistics:")
                print(f"  Mean tokens: {mean(tokens_generated):.1f}")
                print(f"  Total tokens: {sum(tokens_generated)}")
                
                # Calculate tokens per second
                total_time = sum(response_times)
                total_tokens = sum(tokens_generated)
                if total_time > 0:
                    print(f"  Tokens/second (aggregate): {total_tokens/total_time:.1f}")
        
        if failed_results:
            print("\nFailure Analysis:")
            error_counts = {}
            for result in failed_results:
                error_type = result.error_message.split(':')[0]
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for error_type, count in error_counts.items():
                print(f"  {error_type}: {count} occurrences")
        
        print("\nSample Responses:")
        for i, result in enumerate(successful_results[:3]):
            print(f"  Request {result.request_id}: {result.response_text[:100]}...")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Ollama server with parallel requests")
    parser.add_argument("--model", default="llama2", help="Model name to test")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--requests", type=int, default=10, help="Number of requests to send")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent requests")
    parser.add_argument("--prompt", default="What is the capital of France?", help="Prompt to send")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--save-results", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.concurrency > args.requests:
        args.concurrency = args.requests
        print(f"Warning: Concurrency reduced to {args.concurrency} (max requests)")
    
    start_time = time.time()
    
    async with OllamaParallelTester(args.url) as tester:
        results = await tester.run_parallel_test(
            model=args.model,
            prompt=args.prompt,
            num_requests=args.requests,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens
        )
        
        total_time = time.time() - start_time
        print(f"\nTotal test time: {total_time:.2f}s")
        
        tester.print_statistics(results)
        
        # Save results if requested
        if args.save_results:
            results_data = {
                "test_config": {
                    "model": args.model,
                    "url": args.url,
                    "requests": args.requests,
                    "concurrency": args.concurrency,
                    "prompt": args.prompt,
                    "max_tokens": args.max_tokens,
                    "total_time": total_time
                },
                "results": [
                    {
                        "request_id": r.request_id,
                        "success": r.success,
                        "response_time": r.response_time,
                        "response_text": r.response_text,
                        "error_message": r.error_message,
                        "tokens_generated": r.tokens_generated
                    }
                    for r in results
                ]
            }
            
            with open(args.save_results, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    asyncio.run(main())