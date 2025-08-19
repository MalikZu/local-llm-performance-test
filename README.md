# Ollama Parallel Testing Script

This Python script allows you to test local LLM performance by sending multiple parallel requests to an Ollama server.

## Features

- **Parallel Requests**: Send multiple concurrent requests to test throughput
- **Configurable Parameters**: Customize model, prompt, concurrency, and request count
- **Performance Metrics**: Get detailed statistics on response times and actual token metrics from Ollama
- **Error Handling**: Robust error handling with detailed failure analysis
- **Results Export**: Save test results to JSON for further analysis

## Installation & Quick Start

1. Install dependencies using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install aiohttp
```

2. Make sure Ollama is running:
```bash
ollama serve
```

3. Pull a model (if not already available):
```bash
ollama pull llama2
```

4. Run basic test:
```bash
python test_ollama_parallel.py
```

5. Or run example scenarios:
```bash
python example_tests.py
```

## Usage

### Basic Usage

```bash
python test_ollama_parallel.py
```

This will run a basic test with default parameters:
- Model: `llama2`
- Requests: 10
- Concurrency: 5
- Prompt: "What is the capital of France?"

### Advanced Usage

```bash
python test_ollama_parallel.py \
  --model llama2 \
  --requests 20 \
  --concurrency 10 \
  --prompt "Explain quantum computing in simple terms" \
  --max-tokens 200 \
  --save-results results.json
```

### Command Line Options

- `--model`: Model name to test (default: `llama2`)
- `--url`: Ollama server URL (default: `http://localhost:11434`)
- `--requests`: Number of requests to send (default: 10)
- `--concurrency`: Number of concurrent requests (default: 5)
- `--prompt`: Prompt to send to the model (default: "What is the capital of France?")
- `--max-tokens`: Maximum tokens to generate (default: 100)
- `--timeout`: Request timeout in seconds (default: 60)
- `--save-results`: Save results to JSON file (optional)

## Avoiding Timeouts

If you experience request timeouts, especially when generating many tokens or using slower models:

1. **Increase timeout**: Use `--timeout` to set a longer timeout (in seconds)
   ```bash
   python test_ollama_parallel.py --timeout 600 --max-tokens 500
   ```

2. **Reduce concurrency**: Lower concurrent requests to reduce server load
   ```bash
   python test_ollama_parallel.py --concurrency 2 --max-tokens 500
   ```

3. **Reduce token count**: Use fewer `--max-tokens` for faster responses
   ```bash
   python test_ollama_parallel.py --max-tokens 100
   ```

## Example Output

```
Starting test with 10 requests, 5 concurrent
Model: llama2
Prompt: What is the capital of France?
------------------------------------------------------------

Total test time: 15.32s

============================================================
TEST RESULTS
============================================================
Total requests: 10
Successful: 10
Failed: 0
Success rate: 100.0%

Response Time Statistics:
  Mean: 1.45s
  Median: 1.42s
  Min: 1.12s
  Max: 1.89s

Token Generation Statistics (from Ollama):
  Mean tokens generated: 25.3
  Total tokens generated: 253
  Mean generation rate: 18.2 tokens/sec
  Max generation rate: 22.1 tokens/sec
  Min generation rate: 14.8 tokens/sec
  Aggregate throughput: 17.4 tokens/sec

Prompt Processing Statistics:
  Mean prompt tokens: 8.0
  Mean prompt processing rate: 156.3 tokens/sec

Ollama Timing Breakdown (from server):
  Mean total duration: 1.38s
  Mean generation time: 1.25s
  Mean model load time: 0.02s

Sample Responses:
  Request 0: The capital of France is Paris. Paris is located in the north-central part of France...
  Request 1: The capital of France is Paris. It is the largest city in France and serves as the...
  Request 2: The capital of France is Paris. This beautiful city is known for its iconic landmarks...
```

## Testing Different Scenarios

### Load Testing
Test how your system handles high load:
```bash
python test_ollama_parallel.py --requests 50 --concurrency 20
```

### Latency Testing
Test response times with sequential requests:
```bash
python test_ollama_parallel.py --requests 10 --concurrency 1
```

### Different Models
Test different models:
```bash
python test_ollama_parallel.py --model codellama --prompt "Write a Python function to sort a list"
```

### Long Responses
Test with longer responses:
```bash
python test_ollama_parallel.py --prompt "Write a detailed essay about artificial intelligence" --max-tokens 500
```

## Understanding the Results

- **Response Time**: Time taken for each individual request
- **Tokens/second (aggregate)**: Total tokens generated divided by total time spent generating
- **Success Rate**: Percentage of requests that completed successfully
- **Concurrency Impact**: Compare results with different concurrency levels to find optimal settings

## Troubleshooting

1. **Connection Refused**: Make sure Ollama is running (`ollama serve`)
2. **Model Not Found**: Pull the model first (`ollama pull <model_name>`)
3. **Timeout Errors**: Reduce concurrency or increase timeout in the script
4. **Memory Issues**: Reduce concurrency or number of requests

## Tips for Effective Testing

1. **Start Small**: Begin with low concurrency and few requests
2. **Monitor Resources**: Watch CPU, memory, and GPU usage during tests
3. **Test Different Prompts**: Different prompt types may have varying performance
4. **Save Results**: Use `--save-results` to compare different test runs
5. **Warm-up**: Run a small test first to warm up the model

## JSON Results Format

When using `--save-results`, the output includes:
- Test configuration
- Individual request results
- Response times and token counts
- Error messages for failed requests

This data can be used for further analysis or visualization.