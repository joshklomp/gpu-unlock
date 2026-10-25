# gpu-unlock on Polaris

Polaris: Ubuntu 26.04, kernel `7.0.0-31-generic`, 2× RTX 3090 (`10de:2204`, no
NVLink, TP=2), 220 W cap set by the vLLM unit, serving
`Frozenlock/Qwen3.8-27B-int4-AutoRound` (BF16 activations, Marlin int4 GEMMs,
MTP speculative decoding, FP8 KV cache).

## What to expect, before measuring it

- **Prefill should get faster. Decode should not.** Prefill is compute-bound
  BF16 MMA with FP32 accumulation, the rate the unlock restores. Decode, even
  with MTP verifying 4 tokens a step, is memory-bandwidth-bound. Upstream
  reports +14–36% LLM prefill.
- **The 220 W cap eats into it.** The lowest upstream GEMM gain (+25%) came from
  cards capped at 225 W. More work per clock means more watts per clock, so a
  capped card lowers its clock. Polaris will land near the bottom of the
  range, not the top. Measure at 220 W, and once at a higher cap, before
  deciding whether the cap should move.
- **The checkpoint is already BF16** (`"dtype": "bfloat16"` in its
  `config.json`). Keep it that way: after the unlock BF16 runs at the full
  rate and FP16 does not (3090 Ti: BF16 +80%, FP16 +35%). Do not add
  `--dtype float16`.
- Where it shows in Aqua: time to first token on long prompts, the first
  turn of a chat with a large file block, and any turn where the prefix
  cache misses.

## Procedure

Each step is checkable before the next one. Keep the JSON outputs.

1. **Preflight** (read-only): `sudo sh tools/polaris/preflight.sh`. Fix any
   FAIL. The two that matter on Ubuntu: Secure Boot needs the DKMS MOK key
   enrolled, and neither `nvidia` nor `nouveau` may be inside the initramfs
   (udev loads them before `gpu_unlock`; the card is then refused with -16).
2. **Baseline, locked, 220 W.**
   ```sh
   python3 tools/polaris/bench_prefill.py run --label locked-220W -o prefill-locked-220W.json
   sudo systemctl stop 'vllm@*' vllm      # bench_gemm refuses a busy GPU
   ~/vllm-env/bin/python tools/polaris/bench_gemm.py --label locked-220W -o gemm-locked-220W.json
   ```
   `bench_gemm.py` runs at whatever power limit the cards are at. The vLLM
   unit sets 220 W at start; if it hasn't run since boot, set it:
   `sudo nvidia-smi -pl 220`.
3. **Install**: `make install` (Ubuntu path: `apt-get build-dep`, then
   `dpkg-buildpackage`, then `apt-get install`). It downloads the pinned
   NVIDIA inputs and checks their SHA-256.
   Add `gpu_unlock.targets=all` to `GRUB_CMDLINE_LINUX_DEFAULT` in
   `/etc/default/grub`, then `sudo update-grub`.
4. **Power off, then power on.** Do a full shutdown, not `reboot`. A card a
   driver has already used this power cycle is refused (WPR2 enabled), and
   a warm reboot may not cut its power. A full power cycle removes the
   question.
5. **Verify**: `sh tools/polaris/status.sh` should list both cards as
   `unlocked`, with exit code 0. `--json` gives the same thing for vllmctl.
6. **Measure again, same cap.** Repeat step 2 with `--label unlocked-220W`, then
   ```sh
   python3 tools/polaris/bench_prefill.py compare prefill-locked-220W.json prefill-unlocked-220W.json
   ```
   Expect prefill up, decode flat (±3%). If decode moved, something else
   changed between the runs. Rule that out before crediting the unlock.
7. **Optional, the cap decision.** One GEMM and one prefill run at, say,
   `nvidia-smi -pl 280`, unlocked. Whether the extra watts are worth it
   depends on PSU headroom and thermals, and that's your call, not
   something to infer from these numbers alone.

Rollback: remove `gpu_unlock.targets=` from the command line and
`update-grub`. The module stays loaded but does nothing. `apt remove
gpu-unlock-dkms gpu-unlock-firmware` removes it and rebuilds the
initramfs.

## Side effect worth knowing

`bench_prefill.py` sends each prompt a second time and reads the server's
prefix-cache counters. That answers the open handoff question, "is prefix
caching on and hitting, in production?", for this hybrid
(linear-attention + attention) model. A cached TTFT far below the cold
one, with a non-zero `prefix_cache_hits` delta, is the evidence the
system → files → history prompt order has been waiting for.

## Wiring it into vllmctl (not done here)

`status.sh --json` is shaped for `/status`. An `ExecStartPre=-…/status.sh
--quiet` line in `vllm@.service` would log a locked boot without blocking the
start (note the leading `-`). Neither change is made by this repo.
