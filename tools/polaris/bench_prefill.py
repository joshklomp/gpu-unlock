#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Prefill and decode speed of a running vLLM server, the way Aqua sees it.

What the unlock should move is prefill (compute-bound: Marlin int4 GEMMs,
attention, the linear-attention kernels, all BF16 with FP32 accumulation).
Decode is memory-bandwidth-bound and should not move; if it does, something
else changed between runs. Standard library only, so any Python works:

    python3 bench_prefill.py run --label baseline-220W -o prefill-baseline.json
    python3 bench_prefill.py run --label unlocked-220W -o prefill-unlocked.json
    python3 bench_prefill.py compare prefill-baseline.json prefill-unlocked.json

Every timed request starts with a fresh nonce, so prefix caching cannot
shorten it. After each size the same prompt is sent once more: the TTFT
drop and the server's prefix-cache counters show whether caching actually
hits on this model in production (an open question in the Aqua handoff).
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.parse
import uuid

ENV_FILE = "/home/phiral/llm/vllm.env"
WORDS = ("tensor prefill decode cache block layer head token batch kernel stream "
         "router sample memory bandwidth latency vector matrix shard rank queue "
         "window context budget reserve history file system order prefix hit").split()


def api_key(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if os.environ.get("VLLM_API_KEY"):
        return os.environ["VLLM_API_KEY"]
    try:
        with open(ENV_FILE) as stream:
            for line in stream:
                m = re.match(r'\s*VLLM_API_KEY\s*=\s*"?([^"\n]+)"?', line)
                if m:
                    return m.group(1).strip()
    except OSError:
        pass
    return None


class Server:
    def __init__(self, base: str, key: str | None, timeout: float):
        url = urllib.parse.urlsplit(base)
        self.https = url.scheme == "https"
        self.host = url.hostname or "127.0.0.1"
        self.port = url.port or (443 if self.https else 80)
        self.prefix = url.path.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if key:
            self.headers["Authorization"] = f"Bearer {key}"
        self.timeout = timeout

    def connect(self) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=self.timeout)

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, bytes]:
        conn = self.connect()
        try:
            conn.request(method, self.prefix + path,
                         json.dumps(body) if body is not None else None, self.headers)
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    def json(self, method: str, path: str, body: dict | None = None) -> dict:
        status, data = self.request(method, path, body)
        if status != 200:
            raise RuntimeError(f"{method} {path}: HTTP {status}: {data[:300]!r}")
        return json.loads(data)

    def metrics(self) -> dict[str, float]:
        """Prefix-cache counters from /metrics, summed over label sets."""
        try:
            status, data = self.request("GET", "/metrics")
        except OSError:
            return {}
        if status != 200:
            return {}
        out: dict[str, float] = {}
        for line in data.decode(errors="replace").splitlines():
            if line.startswith("#") or "prefix_cache" not in line:
                continue
            name = re.split(r"[{\s]", line, maxsplit=1)[0]
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(None, 1)[1])
            except (IndexError, ValueError):
                pass
        return out

    def stream(self, model: str, prompt: str, max_tokens: int) -> dict:
        body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
                "temperature": 0, "ignore_eos": True, "stream": True,
                "stream_options": {"include_usage": True}}
        conn = self.connect()
        t0 = time.perf_counter()
        first = None
        usage = {}
        chunks = 0
        try:
            conn.request("POST", self.prefix + "/v1/completions", json.dumps(body), self.headers)
            resp = conn.getresponse()
            if resp.status != 200:
                raise RuntimeError(f"completions: HTTP {resp.status}: {resp.read()[:300]!r}")
            for raw in resp:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    if choice.get("text"):
                        chunks += 1
                        if first is None:
                            first = time.perf_counter()
        finally:
            conn.close()
        end = time.perf_counter()
        if first is None:
            raise RuntimeError("stream ended without any text")
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", chunks)
        ttft = first - t0
        decode = end - first
        return {"prompt_tokens": prompt_tokens, "completion_tokens": completion,
                "ttft_s": ttft, "prefill_tok_s": prompt_tokens / ttft if ttft else 0.0,
                "decode_tok_s": (completion - 1) / decode if decode > 0 and completion > 1 else 0.0,
                "total_s": end - t0}


def filler(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def build_prompt(server: Server, model: str, tokens: int, tokenize: bool) -> str:
    """Filler text of about `tokens` tokens (calibrated with /tokenize when available)."""
    per_word = 1.3
    if tokenize:
        sample = filler(2000, 1)
        try:
            count = server.json("POST", "/tokenize", {"model": model, "prompt": sample})["count"]
            per_word = count / 2000
        except (RuntimeError, KeyError, OSError):
            pass
    return filler(max(1, int(tokens / per_word)), tokens)


def run(args: argparse.Namespace) -> None:
    server = Server(args.base_url, api_key(args.key), args.timeout)
    model = args.model or server.json("GET", "/v1/models")["data"][0]["id"]
    sizes = [int(s) for s in args.sizes.split(",")]
    report = {"label": args.label, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "base_url": args.base_url, "model": model, "gen_tokens": args.gen, "sizes": []}
    # Warm-up: first requests can include CUDA graph capture or JIT.
    server.stream(model, f"[{uuid.uuid4()}] warm up", 8)
    for size in sizes:
        body = build_prompt(server, model, size, not args.no_tokenize)
        runs = []
        for _ in range(args.repeats):
            runs.append(server.stream(model, f"[{uuid.uuid4()}] {body}", args.gen))
        entry = {"target_tokens": size,
                 "prompt_tokens": runs[0]["prompt_tokens"],
                 "ttft_s_median": statistics.median(r["ttft_s"] for r in runs),
                 "prefill_tok_s_median": statistics.median(r["prefill_tok_s"] for r in runs),
                 "decode_tok_s_median": statistics.median(r["decode_tok_s"] for r in runs),
                 "runs": runs}
        # Cache probe: an exact repeat of a prompt the server has just seen.
        seen = f"[{uuid.uuid4()}] {body}"
        server.stream(model, seen, 1)
        before = server.metrics()
        repeat = server.stream(model, seen, 1)
        after = server.metrics()
        entry["cached_ttft_s"] = repeat["ttft_s"]
        entry["prefix_cache_delta"] = {k: after[k] - before.get(k, 0.0) for k in after
                                       if after[k] != before.get(k, 0.0)}
        report["sizes"].append(entry)
        print(f"{entry['prompt_tokens']:>7} tok  TTFT {entry['ttft_s_median']:7.2f}s  "
              f"prefill {entry['prefill_tok_s_median']:8.0f} tok/s  "
              f"decode {entry['decode_tok_s_median']:6.1f} tok/s  "
              f"cached TTFT {repeat['ttft_s']:6.2f}s", flush=True)
    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(text + "\n")
    else:
        print(text)


def compare(args: argparse.Namespace) -> None:
    a, b = (json.load(open(path)) for path in (args.before, args.after))
    print(f"{a['label']}  ->  {b['label']}")
    print(f"{'tokens':>8} {'TTFT before':>12} {'TTFT after':>11} {'prefill':>9} {'decode':>8}")
    by_size = {s["target_tokens"]: s for s in b["sizes"]}
    for s in a["sizes"]:
        t = by_size.get(s["target_tokens"])
        if not t:
            continue
        pre = t["prefill_tok_s_median"] / s["prefill_tok_s_median"] - 1
        dec = (t["decode_tok_s_median"] / s["decode_tok_s_median"] - 1
               if s["decode_tok_s_median"] else 0.0)
        print(f"{s['prompt_tokens']:>8} {s['ttft_s_median']:>11.2f}s {t['ttft_s_median']:>10.2f}s "
              f"{pre:>+8.1%} {dec:>+8.1%}")
    if a.get("model") != b.get("model"):
        print(f"warning: different models ({a.get('model')} vs {b.get('model')})", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="benchmark a running server")
    r.add_argument("--label", required=True)
    r.add_argument("-o", "--output")
    r.add_argument("--base-url", default="http://127.0.0.1:9000")
    r.add_argument("--model", help="default: the first model /v1/models lists")
    r.add_argument("--key", help=f"default: $VLLM_API_KEY, then {ENV_FILE}")
    r.add_argument("--sizes", default="1024,8192,32768")
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--gen", type=int, default=128, help="tokens generated per request")
    r.add_argument("--timeout", type=float, default=600)
    r.add_argument("--no-tokenize", action="store_true", help="skip /tokenize calibration")
    c = sub.add_parser("compare", help="compare two run outputs")
    c.add_argument("before")
    c.add_argument("after")
    args = parser.parse_args()
    try:
        (run if args.cmd == "run" else compare)(args)
    except (RuntimeError, OSError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
