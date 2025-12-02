# Context Utilization Benchmark

Test whether structured (JSON) input allows LLMs to utilize more of their context window than natural language prose.

## Setup

```bash
# Install the Google AI SDK
pip install google-generativeai

# Set your API key (get one from https://makersuite.google.com/app/apikey)
export GOOGLE_API_KEY="your-key-here"

# On Windows PowerShell:
$env:GOOGLE_API_KEY="your-key-here"

# On Windows CMD:
set GOOGLE_API_KEY=your-key-here
```

## Quick Start

```bash
# 1. Verify your setup works (runs smallest scale)
python ledger_benchmark.py quick

# 2. Run a full batch experiment
python ledger_benchmark.py batch --scales "10k,50k,100k" --seeds "42,123,456"

# 3. For Gemini's full 1M context, add larger scales
python ledger_benchmark.py batch --scales "10k,100k,500k,1M" --seeds "42,123,456"
```

## Commands

### generate
Generate benchmark data at a specific scale.
```bash
python ledger_benchmark.py generate --scale 100k --seed 42 --output my_benchmark.json
```

### export
Export benchmark to structured/prose text files.
```bash
python ledger_benchmark.py export --input my_benchmark.json --format both
```

### run
Run a single benchmark against a model.
```bash
python ledger_benchmark.py run --input my_benchmark.json --format structured --model gemini-1.5-pro
python ledger_benchmark.py run --input my_benchmark.json --format prose --model gemini-1.5-pro
```

### verify
Verify the answer computation is correct.
```bash
python ledger_benchmark.py verify --input my_benchmark.json
```

### batch
Run full experiment across multiple scales and seeds.
```bash
python ledger_benchmark.py batch --scales "10k,50k,100k,500k" --seeds "42,123,456,789,101112"
```

### quick
Quick test to verify API setup works.
```bash
python ledger_benchmark.py quick --model gemini-1.5-pro
```

## Available Scales

| Scale | Warehouses | SKUs | Transactions | ~Tokens (Structured) | ~Tokens (Prose) |
|-------|------------|------|--------------|---------------------|-----------------|
| 10k   | 10         | 10   | 500          | 20k                 | 15k             |
| 50k   | 25         | 20   | 2,000        | 80k                 | 60k             |
| 100k  | 50         | 30   | 4,000        | 160k                | 120k            |
| 500k  | 100        | 50   | 20,000       | 800k                | 600k            |
| 1M    | 150        | 75   | 40,000       | 1.6M                | 1.2M            |
| 2M    | 200        | 100  | 80,000       | 3.2M                | 2.4M            |

## How It Works

The benchmark generates an inventory tracking system with:
- Multiple warehouses
- Multiple product SKUs
- Thousands of transactions (sales, restocks, transfers, adjustments)

The question asks: "What is the current inventory of SKU-X at Warehouse-Y?"

The correct answer requires tracing ALL relevant transactions from the beginning.

Two formats are tested:
1. **Structured (JSON)**: Each record is a JSON object with typed fields
2. **Prose**: Each record is a natural language sentence

If structured format yields better accuracy, it supports the hypothesis that schema-aligned input improves context utilization.

## Results

Results are saved to `results/results_TIMESTAMP.csv` with columns:
- scale, seed, format, tokens, expected, model_answer, correct, response, error

## Models

Tested with:
- `gemini-1.5-pro` (1M context)
- `gemini-1.5-flash` (1M context, faster/cheaper)
- `gemini-2.0-flash-exp` (if available)
