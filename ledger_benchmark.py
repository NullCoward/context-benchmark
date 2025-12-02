"""
Ledger Benchmark: Test structured vs prose context utilization in large language models.

This benchmark generates a large inventory tracking dataset where every transaction
matters for computing the final answer. The same data is rendered in three formats:
- Structured (JSON): Standard JSON lines
- TOON: Token-Oriented Object Notation (compact, low-entropy)
- Prose: Natural language descriptions

Usage:
    python ledger_benchmark.py generate --scale 100k --seed 42
    python ledger_benchmark.py run --model gemini-2.5-flash --format structured
    python ledger_benchmark.py run --model gemini-2.5-flash --format toon
    python ledger_benchmark.py run --model gemini-2.5-flash --format prose
"""

import json
import random
import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

# Scale configurations (approximate token targets)
SCALE_CONFIGS = {
    "10k": {"warehouses": 10, "skus": 10, "transactions": 500},
    "50k": {"warehouses": 25, "skus": 20, "transactions": 2000},
    "100k": {"warehouses": 50, "skus": 30, "transactions": 4000},
    "500k": {"warehouses": 100, "skus": 50, "transactions": 20000},
    "1M": {"warehouses": 150, "skus": 75, "transactions": 40000},
    "2M": {"warehouses": 200, "skus": 100, "transactions": 80000},
}


def generate_benchmark(scale: str, seed: int = 42) -> dict:
    """Generate benchmark data at the specified scale."""
    random.seed(seed)
    config = SCALE_CONFIGS[scale]

    num_warehouses = config["warehouses"]
    num_skus = config["skus"]
    num_transactions = config["transactions"]

    warehouses = [f"WH-{i:04d}" for i in range(num_warehouses)]
    skus = [f"SKU-{i:04d}" for i in range(num_skus)]

    # Initialize inventory (random 50-200 per SKU per warehouse)
    initial_inventory = {
        wh: {sku: random.randint(50, 200) for sku in skus}
        for wh in warehouses
    }

    # Track current state (copy of initial)
    current_inventory = {
        wh: dict(inv) for wh, inv in initial_inventory.items()
    }

    transactions = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    current_time = base_time

    for tx_id in range(num_transactions):
        action = random.choices(
            ["sale", "restock", "transfer", "adjustment"],
            weights=[40, 25, 25, 10]  # Sales most common
        )[0]

        wh = random.choice(warehouses)
        sku = random.choice(skus)

        tx = {
            "id": f"TX-{tx_id:06d}",
            "timestamp": current_time.isoformat(),
            "warehouse": wh,
            "sku": sku,
        }

        if action == "sale":
            max_sale = min(25, current_inventory[wh][sku])
            if max_sale > 0:
                qty = random.randint(1, max_sale)
                current_inventory[wh][sku] -= qty
                tx["action"] = "sale"
                tx["quantity"] = qty
                transactions.append(tx)

        elif action == "restock":
            qty = random.randint(10, 100)
            current_inventory[wh][sku] += qty
            tx["action"] = "restock"
            tx["quantity"] = qty
            transactions.append(tx)

        elif action == "transfer":
            other_wh = random.choice([w for w in warehouses if w != wh])
            max_transfer = min(30, current_inventory[wh][sku])
            if max_transfer > 0:
                qty = random.randint(1, max_transfer)
                current_inventory[wh][sku] -= qty
                current_inventory[other_wh][sku] += qty
                tx["action"] = "transfer"
                tx["quantity"] = qty
                tx["destination"] = other_wh
                transactions.append(tx)

        elif action == "adjustment":
            adj = random.randint(-10, 10)
            new_val = max(0, current_inventory[wh][sku] + adj)
            actual_adj = new_val - current_inventory[wh][sku]
            current_inventory[wh][sku] = new_val
            tx["action"] = "adjustment"
            tx["quantity"] = actual_adj
            tx["reason"] = random.choice(["audit", "damage", "found", "correction"])
            transactions.append(tx)

        # Advance time randomly
        current_time += timedelta(seconds=random.randint(30, 300))

    # Pick target for question (ensure it has interesting history)
    # Find a SKU/warehouse combo that had many transactions
    tx_counts = {}
    for tx in transactions:
        key = (tx["warehouse"], tx["sku"])
        tx_counts[key] = tx_counts.get(key, 0) + 1

    # Pick from top 20% most active combinations
    sorted_keys = sorted(tx_counts.keys(), key=lambda k: tx_counts[k], reverse=True)
    target_idx = random.randint(0, len(sorted_keys) // 5)
    target_wh, target_sku = sorted_keys[target_idx]

    answer = current_inventory[target_wh][target_sku]

    return {
        "metadata": {
            "scale": scale,
            "seed": seed,
            "num_warehouses": num_warehouses,
            "num_skus": num_skus,
            "num_transactions": len(transactions),
            "generated_at": datetime.now().isoformat(),
        },
        "initial_inventory": initial_inventory,
        "transactions": transactions,
        "question": {
            "warehouse": target_wh,
            "sku": target_sku,
            "text": f"What is the current inventory count of {target_sku} at {target_wh}?"
        },
        "answer": answer,
        "verification": {
            "target_initial": initial_inventory[target_wh][target_sku],
            "target_tx_count": tx_counts[(target_wh, target_sku)],
        }
    }


def to_structured(data: dict) -> str:
    """Convert benchmark data to structured JSON-lines format."""
    lines = []

    # Header comment
    lines.append("# INVENTORY TRACKING SYSTEM - STRUCTURED DATA")
    lines.append(f"# Warehouses: {data['metadata']['num_warehouses']}")
    lines.append(f"# SKUs: {data['metadata']['num_skus']}")
    lines.append(f"# Transactions: {data['metadata']['num_transactions']}")
    lines.append("")

    # Initial inventory
    lines.append("## INITIAL INVENTORY STATE")
    for wh, inv in data["initial_inventory"].items():
        for sku, qty in inv.items():
            lines.append(json.dumps({"type": "initial", "warehouse": wh, "sku": sku, "quantity": qty}))
    lines.append("")

    # Transactions
    lines.append("## TRANSACTION LOG")
    for tx in data["transactions"]:
        lines.append(json.dumps(tx))
    lines.append("")

    # Question
    lines.append("## QUERY")
    lines.append(json.dumps({
        "type": "query",
        "warehouse": data["question"]["warehouse"],
        "sku": data["question"]["sku"]
    }))
    lines.append("")
    lines.append(f"Question: {data['question']['text']}")
    lines.append("Please respond with only the numeric inventory count.")

    return "\n".join(lines)


def to_toon(data: dict) -> str:
    """
    Convert benchmark data to TOON (Token-Oriented Object Notation) format.

    Uses the python-toon library (https://github.com/xaviviro/python-toon)
    to convert JSON to the official TOON format for 30-60% token savings.
    """
    try:
        import toon
    except ImportError:
        print("Please install python-toon: pip install python-toon")
        return None

    # Flatten initial inventory into a list of objects for better TOON columnar format
    inv_entries = []
    for wh, inv in data["initial_inventory"].items():
        for sku, qty in inv.items():
            inv_entries.append({"warehouse": wh, "sku": sku, "quantity": qty})

    # Build the data structure optimized for TOON
    toon_data = {
        "metadata": {
            "warehouses": data["metadata"]["num_warehouses"],
            "skus": data["metadata"]["num_skus"],
            "transactions": data["metadata"]["num_transactions"],
        },
        "initial_inventory": inv_entries,
        "transactions": data["transactions"],
        "query": {
            "warehouse": data["question"]["warehouse"],
            "sku": data["question"]["sku"],
        }
    }

    # Encode to TOON format
    toon_str = toon.encode(toon_data)

    # Add the question in natural language at the end
    toon_str += f"\n\nQuestion: {data['question']['text']}\nRespond with only the numeric count."

    return toon_str


def to_chat_json(data: dict) -> str:
    """
    Convert benchmark data to chat/message API style JSON.

    Each transaction becomes a "message" like you'd see in a chatbot API,
    with role, timestamp, and content fields. Semi-structured format that
    combines JSON structure with natural language content.
    """
    lines = []

    # System message with context
    lines.append(json.dumps({
        "role": "system",
        "content": f"Inventory tracking system with {data['metadata']['num_warehouses']} warehouses and {data['metadata']['num_skus']} SKUs."
    }))

    # Initial inventory as a series of "assistant" messages
    for wh, inv in data["initial_inventory"].items():
        for sku, qty in inv.items():
            lines.append(json.dumps({
                "role": "assistant",
                "type": "init",
                "warehouse": wh,
                "sku": sku,
                "content": f"Initial inventory: {qty} units of {sku} at {wh}"
            }))

    # Transactions as messages
    for tx in data["transactions"]:
        if tx["action"] == "sale":
            content = f"Sold {tx['quantity']} units of {tx['sku']} from {tx['warehouse']}"
        elif tx["action"] == "restock":
            content = f"Restocked {tx['quantity']} units of {tx['sku']} at {tx['warehouse']}"
        elif tx["action"] == "transfer":
            content = f"Transferred {tx['quantity']} units of {tx['sku']} from {tx['warehouse']} to {tx['destination']}"
        elif tx["action"] == "adjustment":
            content = f"Adjusted {tx['sku']} at {tx['warehouse']} by {tx['quantity']} units ({tx.get('reason', 'unspecified')})"
        else:
            content = str(tx)

        lines.append(json.dumps({
            "role": "assistant",
            "type": tx["action"],
            "timestamp": tx["timestamp"],
            "warehouse": tx["warehouse"],
            "sku": tx["sku"],
            "quantity": tx["quantity"],
            "content": content
        }))

    # Query as user message
    lines.append(json.dumps({
        "role": "user",
        "content": data["question"]["text"] + " Respond with only the numeric count."
    }))

    return "\n".join(lines)


def to_csv(data: dict) -> str:
    """
    Convert benchmark data to CSV tabular format.

    Uses standard CSV with headers - tabular but not JSON organized.
    Initial inventory and transactions are in separate sections.
    """
    lines = []

    # Header comment
    lines.append(f"# Inventory Tracking System")
    lines.append(f"# Warehouses: {data['metadata']['num_warehouses']}")
    lines.append(f"# SKUs: {data['metadata']['num_skus']}")
    lines.append(f"# Transactions: {data['metadata']['num_transactions']}")
    lines.append("")

    # Initial inventory section
    lines.append("## INITIAL_INVENTORY")
    lines.append("warehouse,sku,quantity")
    for wh, inv in data["initial_inventory"].items():
        for sku, qty in inv.items():
            lines.append(f"{wh},{sku},{qty}")
    lines.append("")

    # Transactions section
    lines.append("## TRANSACTIONS")
    lines.append("id,timestamp,warehouse,sku,action,quantity,destination,reason")
    for tx in data["transactions"]:
        tx_id = tx.get("id", "")
        ts = tx.get("timestamp", "")
        wh = tx.get("warehouse", "")
        sku = tx.get("sku", "")
        action = tx.get("action", "")
        qty = tx.get("quantity", "")
        dest = tx.get("destination", "")
        reason = tx.get("reason", "")
        lines.append(f"{tx_id},{ts},{wh},{sku},{action},{qty},{dest},{reason}")

    lines.append("")
    lines.append("## QUERY")
    lines.append(data["question"]["text"])
    lines.append("Respond with only the numeric inventory count.")

    return "\n".join(lines)


def to_prose(data: dict) -> str:
    """Convert benchmark data to natural language prose format."""
    lines = []

    # Header
    lines.append("INVENTORY TRACKING SYSTEM - DAILY OPERATIONS LOG")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"This document contains the complete inventory records for a warehouse network")
    lines.append(f"consisting of {data['metadata']['num_warehouses']} warehouses tracking")
    lines.append(f"{data['metadata']['num_skus']} different product SKUs.")
    lines.append("")

    # Initial inventory
    lines.append("INITIAL INVENTORY COUNTS")
    lines.append("-" * 30)
    lines.append("")

    for wh, inv in data["initial_inventory"].items():
        wh_num = wh.replace("WH-", "Warehouse ")
        lines.append(f"{wh_num} starting inventory:")
        for sku, qty in inv.items():
            sku_name = sku.replace("SKU-", "Product SKU ")
            lines.append(f"  - {sku_name}: {qty} units")
        lines.append("")

    # Transactions
    lines.append("TRANSACTION HISTORY")
    lines.append("-" * 30)
    lines.append("")

    for tx in data["transactions"]:
        ts = datetime.fromisoformat(tx["timestamp"])
        time_str = ts.strftime("%B %d, %Y at %I:%M:%S %p")
        wh_name = tx["warehouse"].replace("WH-", "Warehouse ")
        sku_name = tx["sku"].replace("SKU-", "Product SKU ")

        if tx["action"] == "sale":
            lines.append(f"On {time_str}, {wh_name} recorded a sale of {tx['quantity']} units of {sku_name}.")

        elif tx["action"] == "restock":
            lines.append(f"On {time_str}, {wh_name} received a restock shipment of {tx['quantity']} units of {sku_name}.")

        elif tx["action"] == "transfer":
            dest_name = tx["destination"].replace("WH-", "Warehouse ")
            lines.append(f"On {time_str}, {wh_name} transferred {tx['quantity']} units of {sku_name} to {dest_name}.")

        elif tx["action"] == "adjustment":
            reason = tx.get("reason", "unspecified")
            if tx["quantity"] >= 0:
                lines.append(f"On {time_str}, {wh_name} recorded an inventory adjustment of +{tx['quantity']} units for {sku_name} due to {reason}.")
            else:
                lines.append(f"On {time_str}, {wh_name} recorded an inventory adjustment of {tx['quantity']} units for {sku_name} due to {reason}.")

    lines.append("")
    lines.append("=" * 50)
    lines.append("QUERY")
    lines.append("=" * 50)
    lines.append("")
    lines.append(data["question"]["text"])
    lines.append("")
    lines.append("Please respond with only the numeric inventory count.")

    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4)."""
    return len(text) // 4


def verify_answer(data: dict) -> int:
    """Independently compute the answer to verify correctness."""
    target_wh = data["question"]["warehouse"]
    target_sku = data["question"]["sku"]

    # Start with initial
    count = data["initial_inventory"][target_wh][target_sku]

    # Apply all transactions
    for tx in data["transactions"]:
        if tx["warehouse"] == target_wh and tx["sku"] == target_sku:
            if tx["action"] == "sale":
                count -= tx["quantity"]
            elif tx["action"] == "restock":
                count += tx["quantity"]
            elif tx["action"] == "transfer":
                count -= tx["quantity"]  # Outgoing transfer
            elif tx["action"] == "adjustment":
                count += tx["quantity"]

        # Check incoming transfers
        if tx["action"] == "transfer" and tx.get("destination") == target_wh and tx["sku"] == target_sku:
            count += tx["quantity"]

    return count


def run_openai(prompt: str, model: str = "gpt-4o-mini") -> dict:
    """
    Run prompt against OpenAI API.

    Model format: "model_name" or "model_name:reasoning_effort"
    Examples:
      - "gpt-5" (default reasoning)
      - "gpt-5:minimal" (minimal reasoning)
      - "gpt-5.1:none" (no reasoning - fast mode)
      - "gpt-5:high" (maximum reasoning)

    Returns dict with keys: content, prompt_tokens, completion_tokens, reasoning_tokens, elapsed_time, error
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("Please install openai: pip install openai")
        return {"content": None, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "error": "openai not installed"}

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return {"content": None, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "error": "no API key"}

    client = OpenAI(api_key=api_key)

    # Parse model name and optional reasoning_effort
    # Format: "model_name" or "model_name:reasoning_effort"
    reasoning_effort = None
    if ":" in model:
        model_name, reasoning_effort = model.split(":", 1)
    else:
        model_name = model

    # Model ID mapping - based on Dec 2025 OpenAI API
    # See: https://platform.openai.com/docs/models
    model_map = {
        # GPT-4.1 series (1M context)
        "gpt-4.1": "gpt-4.1",
        "gpt-4.1-mini": "gpt-4.1-mini",
        "gpt-4.1-nano": "gpt-4.1-nano",
        # GPT-4o series (128k context)
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        # Reasoning models
        "o3": "o3",
        "o3-mini": "o3-mini",
        "o4-mini": "o4-mini",
        # Legacy
        "gpt-4-turbo": "gpt-4-turbo",
    }
    model_name = model_map.get(model_name, model_name)

    import time as time_module
    start_time = time_module.time()

    try:
        # GPT-5 and newer models use max_completion_tokens, older use max_tokens
        # GPT-5 models use internal reasoning tokens, so we need a much higher limit
        # Also use store=False to disable conversation history storage
        #
        # reasoning_effort values:
        #   - "none": No reasoning (GPT-5.1 only, default for 5.1)
        #   - "minimal": Very few reasoning tokens
        #   - "low": Reduced exploration
        #   - "medium": Default for GPT-5
        #   - "high": Maximum reasoning
        reasoning_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro", "gpt-5.1", "o3", "o3-mini", "o4-mini", "o1", "gpt-4.1"]
        is_reasoning_model = any(m in model_name for m in reasoning_models)

        if is_reasoning_model:
            # Adjust max tokens based on reasoning effort
            if reasoning_effort in ["none", "minimal"]:
                max_tokens = 1000  # Less needed without reasoning overhead
            elif reasoning_effort == "high":
                max_tokens = 100000  # Very high for heavy reasoning
            else:
                max_tokens = 16000  # Default for reasoning models

            # Build API call kwargs
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_tokens,
                "store": False,
            }

            # Add reasoning_effort if specified
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

            response = client.chat.completions.create(**kwargs)
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                store=False,  # Don't store in conversation history
            )

        elapsed_time = time_module.time() - start_time
        content = response.choices[0].message.content

        # Extract token usage
        prompt_tokens = getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0
        completion_tokens = getattr(response.usage, 'completion_tokens', 0) if response.usage else 0

        # Extract reasoning tokens if available (GPT-5, o-series)
        reasoning_tokens = 0
        if response.usage and hasattr(response.usage, 'completion_tokens_details'):
            details = response.usage.completion_tokens_details
            if details and hasattr(details, 'reasoning_tokens'):
                reasoning_tokens = details.reasoning_tokens or 0

        if content is None:
            print(f"    Warning: Response content is None. finish_reason={response.choices[0].finish_reason}")
            if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
                print(f"    Refusal: {response.choices[0].message.refusal}")

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "elapsed_time": elapsed_time,
            "reasoning_effort": reasoning_effort,
            "error": None
        }
    except Exception as e:
        elapsed_time = time_module.time() - start_time
        print(f"    API Error: {e}")
        return {"content": None, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "elapsed_time": elapsed_time, "error": str(e)}


def run_gemini(prompt: str, model: str = "gemini-2.5-flash") -> dict:
    """
    Run prompt against Gemini API.

    Returns dict with keys: content, prompt_tokens, completion_tokens, error
    """
    try:
        import google.generativeai as genai
    except ImportError:
        print("Please install google-generativeai: pip install google-generativeai")
        return {"content": None, "prompt_tokens": 0, "completion_tokens": 0, "error": "genai not installed"}

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Please set GOOGLE_API_KEY environment variable")
        return {"content": None, "prompt_tokens": 0, "completion_tokens": 0, "error": "no API key"}

    genai.configure(api_key=api_key)

    # Map common model names to current API names
    # Use models/ prefix for the API
    model_map = {
        "gemini-2.5-flash": "models/gemini-2.5-flash",
        "gemini-2.5-pro": "models/gemini-2.5-pro",
        "gemini-2.0-flash": "models/gemini-2.0-flash",
        "gemini-flash": "models/gemini-flash-latest",
        "gemini-pro": "models/gemini-pro-latest",
    }
    model_name = model_map.get(model, f"models/{model}" if not model.startswith("models/") else model)

    import time as time_module
    start_time = time_module.time()

    try:
        model_instance = genai.GenerativeModel(model_name)
        response = model_instance.generate_content(prompt)

        elapsed_time = time_module.time() - start_time

        # Extract token usage if available
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, 'usage_metadata'):
            prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)

        return {
            "content": response.text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_time": elapsed_time,
            "error": None
        }
    except Exception as e:
        elapsed_time = time_module.time() - start_time
        print(f"    API Error: {e}")
        return {"content": None, "prompt_tokens": 0, "completion_tokens": 0, "elapsed_time": elapsed_time, "error": str(e)}


def run_model(prompt: str, model: str, provider: str = "openai") -> dict:
    """
    Run prompt against specified provider.

    Returns dict with keys: content, prompt_tokens, completion_tokens, error
    """
    if provider == "openai":
        return run_openai(prompt, model)
    elif provider == "gemini":
        return run_gemini(prompt, model)
    else:
        print(f"Unknown provider: {provider}")
        return {"content": None, "prompt_tokens": 0, "completion_tokens": 0, "error": "unknown provider"}


def run_batch_experiment(scales: list, seeds: list, model: str, provider: str = "openai", output_dir: str = "results"):
    """Run a full batch experiment across multiple scales and seeds."""
    import csv
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"results_{timestamp}.csv")

    results = []

    print(f"Starting batch experiment")
    print(f"Provider: {provider}")
    print(f"Scales: {scales}")
    print(f"Seeds: {seeds}")
    print(f"Model: {model}")
    print(f"Total runs: {len(scales) * len(seeds) * 5} (5 formats per scale/seed)")
    print("=" * 60)

    for scale in scales:
        for seed in seeds:
            print(f"\n[{scale}] Generating with seed {seed}...")
            data = generate_benchmark(scale, seed)

            for fmt in ["structured", "toon", "chat_json", "csv", "prose"]:
                print(f"  Testing {fmt} format...")

                if fmt == "structured":
                    prompt = to_structured(data)
                elif fmt == "toon":
                    prompt = to_toon(data)
                elif fmt == "chat_json":
                    prompt = to_chat_json(data)
                elif fmt == "csv":
                    prompt = to_csv(data)
                else:
                    prompt = to_prose(data)

                token_est = estimate_tokens(prompt)
                expected = data['answer']

                # Call the model
                try:
                    api_result = run_model(prompt, model, provider)
                    response = api_result.get("content")
                    prompt_tokens = api_result.get("prompt_tokens", 0)
                    completion_tokens = api_result.get("completion_tokens", 0)

                    if response:
                        # Extract number from response
                        import re
                        numbers = re.findall(r'\d+', response)
                        model_answer = int(numbers[0]) if numbers else None
                        correct = model_answer == expected

                        result = {
                            "scale": scale,
                            "seed": seed,
                            "format": fmt,
                            "tokens_est": token_est,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "expected": expected,
                            "model_answer": model_answer,
                            "correct": correct,
                            "response": response[:200],  # Truncate for CSV
                            "error": None
                        }

                        status = "[OK]" if correct else "[FAIL]"
                        print(f"    {status} Expected: {expected}, Got: {model_answer} (tokens: {completion_tokens})")
                    else:
                        result = {
                            "scale": scale,
                            "seed": seed,
                            "format": fmt,
                            "tokens_est": token_est,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "expected": expected,
                            "model_answer": None,
                            "correct": False,
                            "response": None,
                            "error": api_result.get("error", "No response")
                        }
                        print(f"    [ERROR] No response from model")

                except Exception as e:
                    result = {
                        "scale": scale,
                        "seed": seed,
                        "format": fmt,
                        "tokens_est": token_est,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "expected": expected,
                        "model_answer": None,
                        "correct": False,
                        "response": None,
                        "error": str(e)
                    }
                    print(f"    [ERROR] {e}")

                results.append(result)

    # Write results to CSV
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print(f"Results saved to: {results_file}")

    # Print summary
    print("\nSUMMARY:")
    print("-" * 40)

    for scale in scales:
        scale_results = [r for r in results if r["scale"] == scale]
        total = len(seeds)

        print(f"{scale}:")
        for fmt in ["structured", "toon", "chat_json", "csv", "prose"]:
            correct = sum(1 for r in scale_results if r["format"] == fmt and r["correct"])
            avg_comp_tokens = sum(r.get("completion_tokens", 0) for r in scale_results if r["format"] == fmt) / total
            print(f"  {fmt:12}: {correct}/{total} correct ({100*correct/total:.0f}%) [avg {avg_comp_tokens:.0f} completion tokens]")

    return results


def run_experiment(scales: list, models: list, runs_per_config: int = 3,
                   seed: int = 42, provider: str = "openai", output_dir: str = None):
    """
    Run a comprehensive experiment for white paper documentation.

    Creates a structured folder with:
    - inputs/ : Generated benchmark files at each scale
    - models/ : Per-model results and individual response files
    - summary.csv : Aggregated results
    - README.md : Experiment documentation
    """
    import csv
    import time as time_module
    from datetime import datetime

    # Create experiment directory
    if output_dir is None:
        output_dir = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "inputs"), exist_ok=True)

    formats = ["structured", "toon", "chat_json", "csv", "prose"]

    print(f"=" * 70)
    print(f"CONTEXT UTILIZATION EXPERIMENT")
    print(f"=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Scales: {scales}")
    print(f"Models: {models}")
    print(f"Formats: {formats}")
    print(f"Runs per configuration: {runs_per_config}")
    print(f"Seed: {seed}")
    print(f"Provider: {provider}")
    total_runs = len(scales) * len(models) * len(formats) * runs_per_config
    print(f"Total API calls: {total_runs}")
    print(f"=" * 70)
    print()

    # Step 1: Generate and save input files for each scale
    print("STEP 1: Generating input files...")
    print("-" * 40)

    benchmark_data = {}
    for scale in scales:
        print(f"  Generating {scale} scale benchmark...")
        scale_dir = os.path.join(output_dir, "inputs", scale)
        os.makedirs(scale_dir, exist_ok=True)

        # Generate benchmark
        data = generate_benchmark(scale, seed)
        benchmark_data[scale] = data

        # Save raw JSON
        json_path = os.path.join(scale_dir, f"benchmark_seed{seed}.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        # Save all format versions
        format_funcs = {
            "structured": to_structured,
            "toon": to_toon,
            "chat_json": to_chat_json,
            "csv": to_csv,
            "prose": to_prose
        }

        for fmt_name, fmt_func in format_funcs.items():
            content = fmt_func(data)
            if content:
                fmt_path = os.path.join(scale_dir, f"{fmt_name}.txt")
                with open(fmt_path, "w", encoding="utf-8") as f:
                    f.write(content)
                tokens = estimate_tokens(content)
                print(f"    {fmt_name}: ~{tokens:,} tokens")

        print(f"    Expected answer: {data['answer']}")
        print()

    # Step 2: Run experiments for each model
    print("STEP 2: Running model experiments...")
    print("-" * 40)

    all_results = []

    for model in models:
        print(f"\n[MODEL: {model}]")
        print("=" * 50)

        model_dir = os.path.join(output_dir, "models", model.replace("/", "_"))
        responses_dir = os.path.join(model_dir, "responses")
        os.makedirs(responses_dir, exist_ok=True)

        model_results = []

        for scale in scales:
            data = benchmark_data[scale]
            expected = data["answer"]

            for fmt in formats:
                # Get prompt
                if fmt == "structured":
                    prompt = to_structured(data)
                elif fmt == "toon":
                    prompt = to_toon(data)
                elif fmt == "chat_json":
                    prompt = to_chat_json(data)
                elif fmt == "csv":
                    prompt = to_csv(data)
                else:
                    prompt = to_prose(data)

                if prompt is None:
                    continue

                prompt_tokens_est = estimate_tokens(prompt)

                for run_num in range(1, runs_per_config + 1):
                    print(f"  {scale}/{fmt} run {run_num}...", end=" ", flush=True)

                    # Make API call
                    api_result = run_model(prompt, model, provider)

                    response_content = api_result.get("content")
                    prompt_tokens = api_result.get("prompt_tokens", 0)
                    completion_tokens = api_result.get("completion_tokens", 0)
                    reasoning_tokens = api_result.get("reasoning_tokens", 0)
                    reasoning_effort = api_result.get("reasoning_effort")
                    elapsed_time = api_result.get("elapsed_time", 0)
                    error = api_result.get("error")

                    # Parse answer
                    model_answer = None
                    correct = False
                    if response_content:
                        import re
                        numbers = re.findall(r'\d+', response_content)
                        if numbers:
                            model_answer = int(numbers[0])
                            correct = (model_answer == expected)

                    # Build result record
                    result = {
                        "model": model,
                        "scale": scale,
                        "format": fmt,
                        "run": run_num,
                        "seed": seed,
                        "expected_answer": expected,
                        "model_answer": model_answer,
                        "correct": correct,
                        "prompt_tokens_est": prompt_tokens_est,
                        "prompt_tokens_actual": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "reasoning_tokens": reasoning_tokens,
                        "reasoning_effort": reasoning_effort,
                        "elapsed_time_sec": round(elapsed_time, 2),
                        "timestamp": datetime.now().isoformat(),
                        "error": error
                    }

                    # Save individual response
                    response_record = {
                        **result,
                        "full_response": response_content
                    }
                    response_file = os.path.join(responses_dir, f"{scale}_{fmt}_run{run_num}.json")
                    with open(response_file, "w", encoding="utf-8") as f:
                        json.dump(response_record, f, indent=2)

                    model_results.append(result)
                    all_results.append(result)

                    # Print status
                    status = "OK" if correct else ("ERR" if error else "FAIL")
                    reasoning_info = f", {reasoning_tokens} reason" if reasoning_tokens else ""
                    print(f"[{status}] ans={model_answer} ({completion_tokens} tok{reasoning_info}, {elapsed_time:.1f}s)")

        # Save model-specific results CSV
        model_csv = os.path.join(model_dir, "results.csv")
        if model_results:
            with open(model_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=model_results[0].keys())
                writer.writeheader()
                writer.writerows(model_results)

    # Step 3: Generate summary
    print("\n" + "=" * 70)
    print("STEP 3: Generating summary...")
    print("-" * 40)

    # Save master CSV
    summary_csv = os.path.join(output_dir, "summary.csv")
    if all_results:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"  Summary CSV: {summary_csv}")

    # Generate README
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"# Context Utilization Experiment\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- **Scales:** {', '.join(scales)}\n")
        f.write(f"- **Models:** {', '.join(models)}\n")
        f.write(f"- **Formats:** {', '.join(formats)}\n")
        f.write(f"- **Runs per config:** {runs_per_config}\n")
        f.write(f"- **Seed:** {seed}\n")
        f.write(f"- **Provider:** {provider}\n")
        f.write(f"- **Total runs:** {total_runs}\n\n")

        f.write(f"## Results Summary\n\n")
        f.write(f"| Model | Scale | Format | Accuracy | Avg Tokens | Avg Time |\n")
        f.write(f"|-------|-------|--------|----------|------------|----------|\n")

        for model in models:
            for scale in scales:
                for fmt in formats:
                    subset = [r for r in all_results
                              if r["model"] == model and r["scale"] == scale and r["format"] == fmt]
                    if subset:
                        correct_count = sum(1 for r in subset if r["correct"])
                        accuracy = correct_count / len(subset) * 100
                        avg_tokens = sum(r["completion_tokens"] for r in subset) / len(subset)
                        avg_time = sum(r["elapsed_time_sec"] for r in subset) / len(subset)
                        f.write(f"| {model} | {scale} | {fmt} | {accuracy:.0f}% ({correct_count}/{len(subset)}) | {avg_tokens:.0f} | {avg_time:.1f}s |\n")

        f.write(f"\n## Folder Structure\n\n")
        f.write(f"```\n")
        f.write(f"{output_dir}/\n")
        f.write(f"├── inputs/           # Generated benchmark files\n")
        f.write(f"│   ├── 10k/\n")
        f.write(f"│   │   ├── benchmark_seed{seed}.json\n")
        f.write(f"│   │   ├── structured.txt\n")
        f.write(f"│   │   ├── toon.txt\n")
        f.write(f"│   │   ├── chat_json.txt\n")
        f.write(f"│   │   ├── csv.txt\n")
        f.write(f"│   │   └── prose.txt\n")
        f.write(f"│   └── ...\n")
        f.write(f"├── models/           # Per-model results\n")
        f.write(f"│   ├── <model>/\n")
        f.write(f"│   │   ├── results.csv\n")
        f.write(f"│   │   └── responses/\n")
        f.write(f"│   │       └── <scale>_<format>_run<n>.json\n")
        f.write(f"│   └── ...\n")
        f.write(f"├── summary.csv       # All results\n")
        f.write(f"└── README.md         # This file\n")
        f.write(f"```\n")

    print(f"  README: {readme_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for model in models:
        print(f"\n{model}:")
        for scale in scales:
            print(f"  {scale}:")
            for fmt in formats:
                subset = [r for r in all_results
                          if r["model"] == model and r["scale"] == scale and r["format"] == fmt]
                if subset:
                    correct_count = sum(1 for r in subset if r["correct"])
                    accuracy = correct_count / len(subset) * 100
                    avg_tokens = sum(r["completion_tokens"] for r in subset) / len(subset)
                    avg_time = sum(r["elapsed_time_sec"] for r in subset) / len(subset)
                    print(f"    {fmt:12}: {accuracy:3.0f}% ({correct_count}/{len(subset)}) | {avg_tokens:6.0f} tok | {avg_time:5.1f}s")

    print("\n" + "=" * 70)
    print(f"Experiment complete! Results in: {output_dir}")
    print("=" * 70)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Ledger Benchmark for Context Utilization")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate benchmark data")
    gen_parser.add_argument("--scale", choices=SCALE_CONFIGS.keys(), default="100k",
                           help="Target token scale")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    gen_parser.add_argument("--output", type=str, default="benchmark_data.json",
                           help="Output file for benchmark data")

    # Export command
    exp_parser = subparsers.add_parser("export", help="Export to structured/toon/prose format")
    exp_parser.add_argument("--input", type=str, default="benchmark_data.json",
                           help="Input benchmark data file")
    exp_parser.add_argument("--format", choices=["structured", "toon", "chat_json", "csv", "prose", "all"], default="all",
                           help="Output format")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmark against model")
    run_parser.add_argument("--input", type=str, default="benchmark_data.json",
                           help="Input benchmark data file")
    run_parser.add_argument("--format", choices=["structured", "toon", "chat_json", "csv", "prose"], required=True,
                           help="Input format to use")
    run_parser.add_argument("--model", type=str, default="gpt-4o-mini",
                           help="Model to use")
    run_parser.add_argument("--provider", type=str, choices=["openai", "gemini"], default="openai",
                           help="API provider (openai or gemini)")

    # Verify command
    ver_parser = subparsers.add_parser("verify", help="Verify answer computation")
    ver_parser.add_argument("--input", type=str, default="benchmark_data.json",
                           help="Input benchmark data file")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Run batch experiment across scales")
    batch_parser.add_argument("--scales", type=str, default="10k,50k,100k,500k",
                             help="Comma-separated list of scales to test")
    batch_parser.add_argument("--seeds", type=str, default="42,123,456",
                             help="Comma-separated list of seeds")
    batch_parser.add_argument("--model", type=str, default="gpt-4o-mini",
                             help="Model to use")
    batch_parser.add_argument("--provider", type=str, choices=["openai", "gemini"], default="openai",
                             help="API provider (openai or gemini)")
    batch_parser.add_argument("--output-dir", type=str, default="results",
                             help="Directory for results")

    # Quick test command (single run to verify setup)
    quick_parser = subparsers.add_parser("quick", help="Quick test with smallest scale")
    quick_parser.add_argument("--model", type=str, default="gpt-4o-mini",
                             help="Model to use")
    quick_parser.add_argument("--provider", type=str, choices=["openai", "gemini"], default="openai",
                             help="API provider (openai or gemini)")

    # Experiment command (comprehensive white paper experiment)
    exp_parser = subparsers.add_parser("experiment", help="Run comprehensive experiment for white paper")
    exp_parser.add_argument("--scales", type=str, default="10k",
                           help="Comma-separated list of scales (e.g., '10k,50k,100k')")
    exp_parser.add_argument("--models", type=str, required=True,
                           help="Comma-separated list of models (e.g., 'gpt-5-mini,gpt-5,gpt-5.1')")
    exp_parser.add_argument("--runs", type=int, default=3,
                           help="Number of runs per configuration (default: 3)")
    exp_parser.add_argument("--seed", type=int, default=42,
                           help="Random seed for benchmark generation")
    exp_parser.add_argument("--provider", type=str, choices=["openai", "gemini"], default="openai",
                           help="API provider")
    exp_parser.add_argument("--output-dir", type=str, default=None,
                           help="Output directory (default: experiment_TIMESTAMP)")

    args = parser.parse_args()

    if args.command == "generate":
        print(f"Generating benchmark at scale {args.scale} with seed {args.seed}...")
        data = generate_benchmark(args.scale, args.seed)

        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Generated {len(data['transactions'])} transactions")
        print(f"Target: {data['question']['warehouse']} / {data['question']['sku']}")
        print(f"Answer: {data['answer']}")
        print(f"Saved to {args.output}")

        # Also show token estimates
        structured = to_structured(data)
        toon_str = to_toon(data)
        chat_json_str = to_chat_json(data)
        csv_str = to_csv(data)
        prose = to_prose(data)
        print(f"\nToken estimates:")
        print(f"  Structured (JSON): ~{estimate_tokens(structured):,} tokens")
        print(f"  TOON:              ~{estimate_tokens(toon_str):,} tokens")
        print(f"  Chat JSON:         ~{estimate_tokens(chat_json_str):,} tokens")
        print(f"  CSV:               ~{estimate_tokens(csv_str):,} tokens")
        print(f"  Prose:             ~{estimate_tokens(prose):,} tokens")

    elif args.command == "export":
        with open(args.input, "r") as f:
            data = json.load(f)

        if args.format in ["structured", "all"]:
            structured = to_structured(data)
            out_file = args.input.replace(".json", "_structured.txt")
            with open(out_file, "w") as f:
                f.write(structured)
            print(f"Structured format: {out_file} (~{estimate_tokens(structured):,} tokens)")

        if args.format in ["toon", "all"]:
            toon_str = to_toon(data)
            out_file = args.input.replace(".json", "_toon.txt")
            with open(out_file, "w") as f:
                f.write(toon_str)
            print(f"TOON format: {out_file} (~{estimate_tokens(toon_str):,} tokens)")

        if args.format in ["chat_json", "all"]:
            chat_json_str = to_chat_json(data)
            out_file = args.input.replace(".json", "_chat_json.txt")
            with open(out_file, "w") as f:
                f.write(chat_json_str)
            print(f"Chat JSON format: {out_file} (~{estimate_tokens(chat_json_str):,} tokens)")

        if args.format in ["csv", "all"]:
            csv_str = to_csv(data)
            out_file = args.input.replace(".json", "_csv.txt")
            with open(out_file, "w") as f:
                f.write(csv_str)
            print(f"CSV format: {out_file} (~{estimate_tokens(csv_str):,} tokens)")

        if args.format in ["prose", "all"]:
            prose = to_prose(data)
            out_file = args.input.replace(".json", "_prose.txt")
            with open(out_file, "w") as f:
                f.write(prose)
            print(f"Prose format: {out_file} (~{estimate_tokens(prose):,} tokens)")

    elif args.command == "run":
        with open(args.input, "r") as f:
            data = json.load(f)

        if args.format == "structured":
            prompt = to_structured(data)
        elif args.format == "toon":
            prompt = to_toon(data)
        elif args.format == "chat_json":
            prompt = to_chat_json(data)
        elif args.format == "csv":
            prompt = to_csv(data)
        else:
            prompt = to_prose(data)

        print(f"Running against {args.model} with {args.format} format...")
        print(f"Prompt size: ~{estimate_tokens(prompt):,} tokens")
        print(f"Expected answer: {data['answer']}")
        print()

        api_result = run_model(prompt, args.model, args.provider)
        response = api_result.get("content")
        if response:
            print(f"Model response: {response}")
            print(f"Completion tokens: {api_result.get('completion_tokens', 0)}")

            # Try to extract number
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                model_answer = int(numbers[0])
                if model_answer == data['answer']:
                    print("\n[OK] CORRECT!")
                else:
                    print(f"\n[FAIL] INCORRECT (expected {data['answer']}, got {model_answer})")

    elif args.command == "verify":
        with open(args.input, "r") as f:
            data = json.load(f)

        computed = verify_answer(data)
        stored = data['answer']

        print(f"Stored answer: {stored}")
        print(f"Computed answer: {computed}")

        if computed == stored:
            print("[OK] Verification passed!")
        else:
            print("[FAIL] Verification FAILED!")

    elif args.command == "batch":
        scales = [s.strip() for s in args.scales.split(",")]
        seeds = [int(s.strip()) for s in args.seeds.split(",")]

        # Validate scales
        for s in scales:
            if s not in SCALE_CONFIGS:
                print(f"Invalid scale: {s}")
                print(f"Valid scales: {list(SCALE_CONFIGS.keys())}")
                return

        run_batch_experiment(scales, seeds, args.model, args.provider, args.output_dir)

    elif args.command == "quick":
        print("Running quick test (10k scale, single seed)...")
        print("This verifies your API key and setup work correctly.\n")

        data = generate_benchmark("10k", 42)
        print(f"Generated benchmark: {len(data['transactions'])} transactions")
        print(f"Question: {data['question']['text']}")
        print(f"Expected answer: {data['answer']}\n")

        for fmt in ["structured", "toon", "chat_json", "csv", "prose"]:
            print(f"Testing {fmt} format...")
            if fmt == "structured":
                prompt = to_structured(data)
            elif fmt == "toon":
                prompt = to_toon(data)
            elif fmt == "chat_json":
                prompt = to_chat_json(data)
            elif fmt == "csv":
                prompt = to_csv(data)
            else:
                prompt = to_prose(data)

            print(f"  Token estimate: ~{estimate_tokens(prompt):,}")

            api_result = run_model(prompt, args.model, args.provider)
            response = api_result.get("content")
            completion_tokens = api_result.get("completion_tokens", 0)
            reasoning_tokens = api_result.get("reasoning_tokens", 0)
            elapsed_time = api_result.get("elapsed_time", 0)

            # Always show token usage
            prompt_tokens = api_result.get("prompt_tokens", 0)
            reasoning_info = f", Reasoning: {reasoning_tokens:,}" if reasoning_tokens else ""
            print(f"  Prompt: {prompt_tokens:,}, Completion: {completion_tokens:,}{reasoning_info}, Time: {elapsed_time:.2f}s")

            if response:
                print(f"  Response: {response[:100]}...")
                import re
                numbers = re.findall(r'\d+', response)
                if numbers:
                    model_answer = int(numbers[0])
                    if model_answer == data['answer']:
                        print(f"  [OK] Correct!")
                    else:
                        print(f"  [FAIL] Expected {data['answer']}, got {model_answer}")
            else:
                error = api_result.get("error")
                if error:
                    print(f"  [ERROR] {error}")
                else:
                    print(f"  [ERROR] No response (content was None - may have hit token limit)")
            print()

        print("Quick test complete! If all five tests passed, you're ready for batch runs.")

    elif args.command == "experiment":
        scales = [s.strip() for s in args.scales.split(",")]
        models = [m.strip() for m in args.models.split(",")]

        # Validate scales
        for s in scales:
            if s not in SCALE_CONFIGS:
                print(f"Invalid scale: {s}")
                print(f"Valid scales: {list(SCALE_CONFIGS.keys())}")
                return

        run_experiment(
            scales=scales,
            models=models,
            runs_per_config=args.runs,
            seed=args.seed,
            provider=args.provider,
            output_dir=args.output_dir
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
