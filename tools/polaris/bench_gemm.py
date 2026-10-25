#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Dense GEMM throughput per GPU, before and after the unlock.

The number gpu-unlock changes directly: FP16/BF16 (and TF32) tensor-core
issue rate with FP32 accumulation. Run it with the vLLM venv's Python, with
vLLM stopped - it refuses to share a GPU, because a contended run measures
the other process, not the card:

    sudo systemctl stop 'vllm@*' vllm
    ~/vllm-env/bin/python bench_gemm.py --label baseline-220W -o gemm-baseline.json

Reports the median of several timed runs per dtype per card, plus the power
and SM clock sampled during them: under a 220 W cap the clock falls as the
tensor cores do more per cycle, which is where the unlock's gain goes.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import subprocess
import threading
import time


def busy_gpus() -> set[str]:
    """Bus IDs with a compute process on them (nvidia-smi view)."""
    if not shutil.which("nvidia-smi"):
        return set()
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_bus_id", "--format=csv,noheader"],
        capture_output=True, text=True, check=False).stdout
    return {line.strip().lower() for line in out.splitlines() if line.strip()}


class Sampler(threading.Thread):
    """Polls power draw and SM clock for one GPU while a benchmark runs."""

    def __init__(self, index: int, period: float = 0.1):
        super().__init__(daemon=True)
        self.index, self.period = index, period
        self.samples: list[tuple[float, float]] = []
        self.stop = threading.Event()

    def run(self) -> None:
        if not shutil.which("nvidia-smi"):
            return
        while not self.stop.is_set():
            out = subprocess.run(
                ["nvidia-smi", "-i", str(self.index),
                 "--query-gpu=power.draw,clocks.sm", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False).stdout.strip()
            try:
                watts, mhz = (float(x) for x in out.split(","))
                self.samples.append((watts, mhz))
            except ValueError:
                pass
            self.stop.wait(self.period)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        watts = [w for w, _ in self.samples]
        mhz = [m for _, m in self.samples]
        return {"power_w_median": statistics.median(watts), "power_w_max": max(watts),
                "sm_mhz_median": statistics.median(mhz)}


def gemm(torch, device, dtype_name: str, n: int, iters: int, repeats: int) -> dict:
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "tf32": torch.float32}[dtype_name]
    torch.backends.cuda.matmul.allow_tf32 = dtype_name == "tf32"
    # FP32 accumulation for half types: the rate GeForce cards halve and the
    # unlock restores. Reduced-precision reductions would measure something else.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    c = torch.empty(n, n, device=device, dtype=dtype)
    for _ in range(3):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize(device)
    results = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            torch.matmul(a, b, out=c)
        end.record()
        torch.cuda.synchronize(device)
        seconds = start.elapsed_time(end) / 1000
        results.append(2 * n ** 3 * iters / seconds / 1e12)
    del a, b, c
    torch.cuda.empty_cache()
    return {"tflops_median": statistics.median(results), "tflops_runs": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="e.g. baseline-220W, unlocked-220W")
    parser.add_argument("-o", "--output", help="write JSON here as well as printing it")
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtypes", default="bf16,fp16,tf32")
    parser.add_argument("--force", action="store_true", help="run even if a GPU is busy")
    args = parser.parse_args()

    import torch  # deferred so --help works without it

    if not torch.cuda.is_available():
        parser.exit(1, "CUDA not available to this Python\n")
    busy = busy_gpus()
    if busy and not args.force:
        parser.exit(1, f"GPU(s) in use by another process: {sorted(busy)}; stop vLLM first\n")

    report = {"label": args.label, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "host": platform.node(), "torch": torch.__version__, "cuda": torch.version.cuda,
              "size": args.size, "gpus": []}
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gpu = {"index": index, "name": props.name, "results": {}}
        for dtype_name in args.dtypes.split(","):
            sampler = Sampler(index)
            sampler.start()
            result = gemm(torch, torch.device("cuda", index), dtype_name,
                          args.size, args.iters, args.repeats)
            sampler.stop.set()
            sampler.join()
            result.update(sampler.summary())
            gpu["results"][dtype_name] = result
            print(f"gpu{index} {dtype_name}: {result['tflops_median']:.1f} TFLOPS "
                  f"@ {result.get('power_w_median', float('nan')):.0f} W, "
                  f"{result.get('sm_mhz_median', float('nan')):.0f} MHz", flush=True)
        report["gpus"].append(gpu)
    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
