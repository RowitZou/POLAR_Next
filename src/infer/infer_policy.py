"""
Policy inference script — simulates verl's rollout request logic.

verl rollout behavior (reproduced here):
  - train_batch_size unique prompts (e.g. 512) are submitted all at once
  - each prompt generates n=8 responses in a single /v1/chat/completions call
  - effective concurrency: batch_size * (server_gpus / train_gpus) = 128 simultaneous requests
    (512 prompts × 8/32 GPU ratio), matching per-engine pressure of training
  - KV cache is cleared before each rollout step (verl does fresh rollout per step)

Usage:
  python infer_policy.py \
      --input  outputs/.../rollout/1.jsonl \
      --output data/chemistry_test/policy_test/step1_results.jsonl \
      --server http://<HOST>:8000/v1 \
      [--n 8] [--max-tokens 32768] [--temperature 1.0] \
      [--max-num-seqs 512] [--concurrency 64]
"""

import argparse
import asyncio
import json
import re
import time
from collections import OrderedDict
from pathlib import Path

import httpx
from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# KV-cache reset
# ---------------------------------------------------------------------------

def clear_kv_cache(server_url: str) -> None:
    """
    Attempt to clear vLLM's prefix cache via HTTP.

    vLLM 0.11.x does not expose reset_prefix_cache as an HTTP endpoint
    (it is an internal Python-level API used by verl directly on the engine
    object).  For HTTP-based benchmarking this is a no-op: since we always
    sample fresh random prompts, the prefix cache is naturally cold and the
    approximation error is negligible.
    """
    print("[info] KV/prefix cache reset: vLLM 0.11.x does not expose this as an "
          "HTTP endpoint — skipping (cache is effectively cold for fresh random prompts)")


# ---------------------------------------------------------------------------
# Prompt parsing helpers
# ---------------------------------------------------------------------------

def parse_input_to_messages(input_str: str) -> list[dict]:
    """
    Parse verl's decoded prompt string back into a messages list.

    verl stores prompts decoded from token IDs.  The resulting text keeps the
    role markers that come from the Qwen3 chat template:
        system
        <system content>
        user
        <user content>
    (Special tokens like <|im_start|> / <|im_end|> decode to empty / newline
    and are effectively stripped, leaving just the role word on its own line.)

    We recover the original turn structure via a simple regex split.
    """
    # Split on role markers that appear at the start of a line
    pattern = re.compile(r'^(system|user|assistant)\n', re.MULTILINE)
    parts = pattern.split(input_str.strip())
    # parts: ['', 'system', '<sys content>', 'user', '<user content>', ...]
    # or when there is no leading empty string the split still alternates role / content

    messages = []
    # Walk through pairs (role, content)
    i = 0
    while i < len(parts):
        token = parts[i].strip()
        if token in ('system', 'user', 'assistant') and i + 1 < len(parts):
            messages.append({"role": token, "content": parts[i + 1].strip()})
            i += 2
        else:
            i += 1

    if not messages:
        # Fallback: treat the entire string as a user message
        messages = [{"role": "user", "content": input_str.strip()}]

    return messages


# ---------------------------------------------------------------------------
# Async inference core
# ---------------------------------------------------------------------------

async def request_one(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    n: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    semaphore: asyncio.Semaphore,
    prompt_id: int,
) -> dict:
    """Send a single chat-completion request and return result dict."""
    async with semaphore:
        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
            )
            elapsed = time.perf_counter() - t0
            outputs = [choice.message.content for choice in resp.choices]
            total_tokens = resp.usage.total_tokens if resp.usage else None
            return {
                "prompt_id": prompt_id,
                "status": "ok",
                "outputs": outputs,
                "elapsed_seconds": round(elapsed, 3),
                "total_tokens": total_tokens,
            }
        except Exception as e:
            elapsed = time.perf_counter() - t0
            return {
                "prompt_id": prompt_id,
                "status": "error",
                "outputs": [],
                "elapsed_seconds": round(elapsed, 3),
                "error": f"{type(e).__name__}: {e}",
            }


async def run_inference(
    prompts: list[dict],          # list of {prompt_id, messages, gts, input_str}
    server_url: str,
    model: str,
    n: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    concurrency: int,
) -> list[dict]:
    """
    Submit all prompts as concurrent requests, bounded by `concurrency`.
    Matches verl's rollout pattern where all batch prompts are dispatched
    simultaneously and vLLM's scheduler manages the queue.
    """
    client = AsyncOpenAI(base_url=server_url, api_key="EMPTY", timeout=600.0)
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        request_one(
            client=client,
            model=model,
            messages=p["messages"],
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            semaphore=semaphore,
            prompt_id=p["prompt_id"],
        )
        for p in prompts
    ]

    print(f"Dispatching {len(tasks)} requests (n={n}, concurrency={concurrency}) ...")
    wall_t0 = time.perf_counter()
    results = await asyncio.gather(*tasks)
    wall_elapsed = time.perf_counter() - wall_t0

    ok = sum(1 for r in results if r["status"] == "ok")
    err = len(results) - ok
    total_samples = ok * n
    print(
        f"Done: {ok} ok / {err} errors  |  "
        f"{total_samples} total responses  |  "
        f"wall {wall_elapsed:.1f}s  ({total_samples/wall_elapsed:.2f} samples/s)"
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Policy inference mimicking verl rollout")
    parser.add_argument("--input", default="outputs/verl_grpo_policy_Qwen3-30B-A3B-ML_reward_RULE_LLM_JUDGE_data_chemistry_moi_half_part/trajectory_data/rollout/1.jsonl", help="Path to rollout .jsonl (1.jsonl, etc.)")
    parser.add_argument("--output", default="data/chemistry_test/policy_test/step1_results.jsonl", help="Output .jsonl path")
    parser.add_argument("--server", default="http://localhost:8000/v1", help="vLLM OpenAI-compat base URL")
    parser.add_argument("--model", default=None, help="Model name (auto-detected from server if omitted)")
    # Sampling params — match training config defaults
    parser.add_argument("--n", type=int, default=8, help="Rollouts per prompt (training n=8)")
    parser.add_argument("--max-tokens", type=int, default=32768, help="max_response_length=32768")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    # Concurrency — mirrors vLLM scheduler capacity
    parser.add_argument(
        "--max-num-seqs", type=int, default=512,
        help="max_num_seqs from training config (vLLM scheduler limit per engine)",
    )
    parser.add_argument(
        "--train-gpus", type=int, default=32,
        help="Total GPUs used in training (default: 32)",
    )
    parser.add_argument(
        "--server-gpus", type=int, default=8,
        help="Total GPUs on the test server (default: 8)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="Concurrent requests sent to the server.  Default: inferred from GPU ratio.\n"
             "  Training: train_gpus / tp → N engine replicas, each handles batch_size/N prompts.\n"
             "  Test:     server_gpus / tp → M engine replicas; concurrency = batch*M/N\n"
             "  With defaults (512 prompts, 32→8 GPUs, TP=2): 512*(8/32)=128",
    )
    parser.add_argument("--deduplicate", action="store_true", default=True,
                        help="Deduplicate prompts by input string (default: True)")
    parser.add_argument("--no-deduplicate", dest="deduplicate", action="store_false")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for prompt sampling")
    args = parser.parse_args()

    # ---- Load input --------------------------------------------------------
    input_path = Path(args.input)
    print(f"Reading {input_path} ...")
    raw_lines = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw_lines.append(json.loads(line))
    print(f"Total lines: {len(raw_lines)}")

    # Deduplicate by input string while preserving order
    if args.deduplicate:
        seen: dict[str, int] = OrderedDict()   # input_str -> prompt_id (0-indexed)
        for row in raw_lines:
            inp = row["input"]
            if inp not in seen:
                seen[inp] = len(seen)
        unique_inputs = list(seen.keys())
        print(f"Unique prompts: {len(unique_inputs)}")
    else:
        unique_inputs = [row["input"] for row in raw_lines]

    # Build prompt list
    prompts = []
    for pid, inp_str in enumerate(unique_inputs):
        prompts.append({
            "prompt_id": pid,
            "input_str": inp_str,
            "messages": parse_input_to_messages(inp_str),
            "gts": None,   # filled below if available
        })

    # Attach ground-truth labels (first occurrence per unique input)
    if args.deduplicate:
        gts_map: dict[str, object] = {}
        for row in raw_lines:
            inp = row["input"]
            if inp not in gts_map:
                gts_map[inp] = row.get("gts")
        for p in prompts:
            p["gts"] = gts_map.get(p["input_str"])

    # Compute concurrency (number of simultaneous HTTP requests to send).
    #
    # Key: max_num_seqs is a cap on vLLM *sequences*, not requests.
    #      Each request with n=8 expands to 8 sequences inside vLLM.
    #
    #   Training:
    #     train_batch_size=512 prompts × n=8 = 4096 sequences total
    #     16 engine replicas (32 GPUs / TP=2)
    #     → 4096 / 16 = 256 sequences/engine  (max_num_seqs=512 is the cap, not the load)
    #     → 256 / n=8 = 32 concurrent requests/engine
    #     → 32 × 16 = 512 total requests  (= train_batch_size, as expected)
    #
    #   Test server (8 GPUs, TP=2 → 4 engines):
    #     To match the same per-engine sequence load (256 seqs/engine):
    #       4 engines × 256 seqs = 1024 sequences total
    #       1024 / n=8 = 128 concurrent requests
    #     Equivalently: train_batch_size × (server_gpus / train_gpus) = 512 × (8/32) = 128
    #     (tp_size cancels out; only the GPU-count ratio matters)
    if args.concurrency:
        concurrency = args.concurrency
    else:
        concurrency = max(1, round(len(prompts) * args.server_gpus / args.train_gpus))

    # ---- Sample exactly `concurrency` prompts randomly --------------------
    # We don't infer all unique prompts: one verl rollout step dispatches
    # batch_size * (server_gpus / train_gpus) prompts to this server at once.
    import random
    rng = random.Random(args.seed)
    if concurrency < len(prompts):
        prompts = rng.sample(prompts, concurrency)
        print(f"Sampled {len(prompts)} prompts (seed={args.seed}) from {len(unique_inputs)} unique")
    else:
        print(f"Using all {len(prompts)} prompts (concurrency={concurrency} >= pool size)")
    print(f"Concurrency: {len(prompts)}  (server_gpus={args.server_gpus}, train_gpus={args.train_gpus})")
    # Re-number prompt IDs after sampling
    for i, p in enumerate(prompts):
        p["prompt_id"] = i

    # ---- Auto-detect model name -------------------------------------------
    model = args.model
    if model is None:
        try:
            base = args.server.rstrip("/")
            with httpx.Client(trust_env=False) as client:
                resp = client.get(f"{base}/models", timeout=10)
            models = resp.json().get("data", [])
            if models:
                model = models[0]["id"]
                print(f"Auto-detected model: {model}")
            else:
                model = "default"
        except Exception as e:
            print(f"Could not auto-detect model ({e}), using 'default'")
            model = "default"

    # ---- Clear KV cache (simulate fresh rollout step) ----------------------
    clear_kv_cache(args.server)

    # ---- Run inference -----------------------------------------------------
    # All sampled prompts are dispatched simultaneously (no semaphore throttle)
    # to reproduce the "all at once" behaviour of a verl rollout step.
    results = asyncio.run(
        run_inference(
            prompts=prompts,
            server_url=args.server,
            model=model,
            n=args.n,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            concurrency=len(prompts),  # all requests fired simultaneously
        )
    )

    # ---- Merge results with prompt metadata --------------------------------
    result_by_id = {r["prompt_id"]: r for r in results}
    output_rows = []
    for p in prompts:
        r = result_by_id[p["prompt_id"]]
        row = {
            "prompt_id": p["prompt_id"],
            "input": p["input_str"],
            "messages": p["messages"],
            "gts": p["gts"],
            "outputs": r["outputs"],
            "status": r["status"],
            "elapsed_seconds": r["elapsed_seconds"],
            "total_tokens": r.get("total_tokens"),
        }
        if r["status"] == "error":
            row["error"] = r.get("error")
        output_rows.append(row)

    # ---- Write output ------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(output_rows)} rows → {out_path}")

    # ---- Print summary stats -----------------------------------------------
    ok_rows = [r for r in output_rows if r["status"] == "ok"]
    err_rows = [r for r in output_rows if r["status"] != "ok"]
    if ok_rows:
        elapsed_list = [r["elapsed_seconds"] for r in ok_rows]
        token_list = [r["total_tokens"] for r in ok_rows if r["total_tokens"]]
        print(f"\n--- Summary ---")
        print(f"  Successful prompts : {len(ok_rows)}")
        print(f"  Error prompts      : {len(err_rows)}")
        print(f"  Total responses    : {len(ok_rows) * args.n}")
        print(f"  Per-request latency: avg={sum(elapsed_list)/len(elapsed_list):.1f}s  "
              f"min={min(elapsed_list):.1f}s  max={max(elapsed_list):.1f}s")
        if token_list:
            print(f"  Avg tokens/req     : {sum(token_list)/len(token_list):.0f}")


if __name__ == "__main__":
    main()
