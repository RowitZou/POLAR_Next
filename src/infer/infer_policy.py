"""
Policy inference script.

Usage:
  python src/infer/infer_policy.py \
      --input  data/chemistry_test/policy_test/eval_prompts_MH_RULE_LLM_JUDGE_step240.jsonl \
      --output data/chemistry_test/policy_test/eval_prompts_MH_RULE_LLM_JUDGE_step240_results.jsonl \
      --server http://<HOST>:8000/v1 \
      [--n 1] [--max-tokens 32768] [--temperature 1.0] [--concurrency 128]

Input JSONL fields: input (str), gts (optional), any extra fields are passed through.
Output JSONL fields: prompt_id, input, messages, gts, outputs, status, elapsed_seconds, total_tokens.

Supports resume: already-written output rows are skipped on restart.
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx
from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# Prompt parsing
# ---------------------------------------------------------------------------

def parse_input_to_messages(input_str: str) -> list[dict]:
    """
    Parse the "system\\n...\\nuser\\n..." format back into a messages list.
    This format is used by verl's decoded prompts and our judger_test files.
    """
    pattern = re.compile(r'^(system|user|assistant)\n', re.MULTILINE)
    parts = pattern.split(input_str.strip())

    messages = []
    i = 0
    while i < len(parts):
        token = parts[i].strip()
        if token in ('system', 'user', 'assistant') and i + 1 < len(parts):
            messages.append({"role": token, "content": parts[i + 1].strip()})
            i += 2
        else:
            i += 1

    if not messages:
        messages = [{"role": "user", "content": input_str.strip()}]
    return messages


# ---------------------------------------------------------------------------
# Async inference
# ---------------------------------------------------------------------------

async def request_one(
    client: AsyncOpenAI,
    model: str,
    prompt_id: int,
    messages: list[dict],
    n: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    semaphore: asyncio.Semaphore,
) -> dict:
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
            return {
                "prompt_id": prompt_id,
                "status": "ok",
                "outputs": [c.message.content for c in resp.choices],
                "elapsed_seconds": round(elapsed, 3),
                "total_tokens": resp.usage.total_tokens if resp.usage else None,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Simple policy inference script")
    parser.add_argument("--input", required=True, help="Input JSONL file (must have 'input' field)")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--server", default="http://localhost:8000/v1", help="OpenAI-compat API base URL")
    parser.add_argument("--model", default=None, help="Model name (auto-detected if omitted)")
    parser.add_argument("--n", type=int, default=1, help="Responses per prompt (default: 1)")
    parser.add_argument("--max-tokens", type=int, default=32768, help="Max response tokens (default: 32768)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=128, help="Max concurrent requests (default: 128)")
    args = parser.parse_args()

    # ---- Load input --------------------------------------------------------
    input_path = Path(args.input)
    print(f"Loading {input_path} ...")
    rows = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Total prompts: {len(rows)}")

    prompts = []
    for pid, row in enumerate(rows):
        prompts.append({
            "prompt_id": pid,
            "input_str": row["input"],
            "messages": parse_input_to_messages(row["input"]),
            "gts": row.get("gts"),
            "extra": {k: v for k, v in row.items() if k not in ("input", "gts")},
        })

    # ---- Resume: skip already-written rows ---------------------------------
    out_path = Path(args.output)
    done_ids: set[int] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(json.loads(line)["prompt_id"])
                    except Exception:
                        pass
        print(f"Resuming: {len(done_ids)} already done, {len(prompts) - len(done_ids)} remaining")

    pending = [p for p in prompts if p["prompt_id"] not in done_ids]
    if not pending:
        print("All done.")
        return

    # ---- Auto-detect model -------------------------------------------------
    model = args.model
    if model is None:
        try:
            base = args.server.rstrip("/")
            with httpx.Client(trust_env=False) as client:
                resp = client.get(f"{base}/models", timeout=10)
            models = resp.json().get("data", [])
            model = models[0]["id"] if models else "default"
            print(f"Auto-detected model: {model}")
        except Exception as e:
            print(f"Could not auto-detect model ({e}), using 'default'")
            model = "default"

    # ---- Run inference -----------------------------------------------------
    async def run():
        client = AsyncOpenAI(base_url=args.server, api_key="EMPTY", timeout=3600.0)
        semaphore = asyncio.Semaphore(args.concurrency)

        tasks = [
            request_one(
                client=client,
                model=model,
                prompt_id=p["prompt_id"],
                messages=p["messages"],
                n=args.n,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                semaphore=semaphore,
            )
            for p in pending
        ]
        print(f"Dispatching {len(tasks)} requests (n={args.n}, concurrency={args.concurrency}) ...")

        results = {}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a") as fout:
            for coro in asyncio.as_completed(tasks):
                r = await coro
                results[r["prompt_id"]] = r
                p = prompts[r["prompt_id"]]
                row = {
                    "prompt_id": r["prompt_id"],
                    "input": p["input_str"],
                    "messages": p["messages"],
                    "gts": p["gts"],
                    **p["extra"],
                    "outputs": r["outputs"],
                    "status": r["status"],
                    "elapsed_seconds": r["elapsed_seconds"],
                    "total_tokens": r.get("total_tokens"),
                }
                if r["status"] == "error":
                    row["error"] = r.get("error")
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()

                done = len(results)
                if done % 100 == 0 or done == len(tasks):
                    ok = sum(1 for v in results.values() if v["status"] == "ok")
                    print(f"  [{done}/{len(tasks)}] ok={ok} errors={done-ok}")

        return list(results.values())

    t0 = time.perf_counter()
    results = asyncio.run(run())
    elapsed = time.perf_counter() - t0

    # ---- Summary -----------------------------------------------------------
    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] != "ok"]
    print("\n--- Summary ---")
    print(f"  Prompts processed : {len(results)}")
    print(f"  OK / Errors       : {len(ok)} / {len(err)}")
    print(f"  Total responses   : {len(ok) * args.n}")
    print(f"  Wall time         : {elapsed:.1f}s  ({len(ok) * args.n / elapsed:.2f} responses/s)")
    if ok:
        lat = [r["elapsed_seconds"] for r in ok]
        print(f"  Latency (per req) : avg={sum(lat)/len(lat):.1f}s  min={min(lat):.1f}s  max={max(lat):.1f}s")
        toks = [r["total_tokens"] for r in ok if r.get("total_tokens")]
        if toks:
            print(f"  Avg tokens/req    : {sum(toks)/len(toks):.0f}")
    if err:
        print("\n  Error samples:")
        for r in err[:5]:
            print(f"    [{r['prompt_id']}] {r.get('error', '')[:120]}")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
