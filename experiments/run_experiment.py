#!/usr/bin/env python3
"""Run the four comparative conditions, or a tau sweep, over one input.

    python experiments/run_experiment.py --input footage.mp4 --runs A B C D
    python experiments/run_experiment.py --input footage.mp4 --sweep
    python experiments/run_experiment.py --input footage.mp4 --runs C \
        --provider gemini --max-seconds 21600

Run A is the ungated, append-only baseline. It is expensive by construction
-- that is its job. Do not point it at six hours on a paid provider until you
have checked the cost on a short clip first.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vigil.config import Config          # noqa: E402
from vigil.pipeline import Pipeline      # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("experiment")

# run_id -> (gate_enabled, memory_bounded, escalation_enabled, label)
CONDITIONS = {
    "A": (False, False, False, "baseline: no gate, append-only memory"),
    "B": (True,  False, False, "gated: SSIM+VAD, append-only memory"),
    "C": (True,  True,  False, "full: SSIM+VAD, bounded memory"),
    "D": (True,  True,  True,  "escalated: full + audio escalation"),
}


def build_config(args, run_id: str, gate: bool, bounded: bool,
                 esc: bool, tau: float | None = None) -> Config:
    cfg = Config.from_env()
    cfg.provider = args.provider
    cfg.gate_enabled = gate
    cfg.memory_bounded = bounded
    cfg.escalation_enabled = esc
    cfg.log_dir = args.log_dir
    cfg.run_id = run_id
    if tau is not None:
        cfg.ssim_tau = tau
    if args.fps:
        cfg.capture_fps = args.fps
    if args.concurrency:
        cfg.max_concurrency = args.concurrency
    if args.speed:
        cfg.replay_speed = args.speed
    return cfg


async def one_run(args, run_id: str, gate: bool, bounded: bool, esc: bool,
                  tau: float | None, label: str) -> dict:
    cfg = build_config(args, run_id, gate, bounded, esc, tau)
    log.info("=== run %s: %s (tau=%.3f, provider=%s) ===",
             run_id, label, cfg.ssim_tau, cfg.provider)
    pipe = Pipeline(args.input, cfg, with_audio=not args.no_audio)
    t0 = time.monotonic()
    summary = await pipe.run_until_complete(max_seconds=args.max_seconds)
    summary.update({
        "run_id": run_id,
        "label": label,
        "tau": cfg.ssim_tau,
        "gate_enabled": gate,
        "memory_bounded": bounded,
        "escalation_enabled": esc,
        "wall_seconds": round(time.monotonic() - t0, 1),
        "gate": pipe.gate_stats.snapshot(),
    })
    log.info("run %s done: %s", run_id, json.dumps(summary["gate"]))
    return summary


async def main_async(args) -> None:
    os.makedirs(args.log_dir, exist_ok=True)
    results = []

    if args.sweep:
        for tau in args.sweep_values:
            rid = f"sweep-tau{tau:.2f}".replace(".", "")
            results.append(await one_run(
                args, rid, True, True, False, tau,
                f"tau sweep at {tau:.2f}"))
    else:
        for r in args.runs:
            key = r.upper()
            if key not in CONDITIONS:
                log.error("unknown run %s (expected A, B, C or D)", r)
                continue
            gate, bounded, esc, label = CONDITIONS[key]
            results.append(await one_run(
                args, key, gate, bounded, esc, args.tau, label))

    out = os.path.join(args.log_dir, "summary.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    log.info("wrote %s", out)

    print("\n%-14s %10s %10s %10s %10s %8s" %
          ("run", "retention", "tokens", "tok/hour", "p95 lat", "429s"))
    for r in results:
        print("%-14s %9.1f%% %10d %10.0f %9.2fs %8d" % (
            r["run_id"], r["gate"]["frame_retention"] * 100,
            r["tokens_total"], r["tokens_per_hour"],
            r["p95_latency_s"], r["rate_limit_faults"]))


def parse_args():
    p = argparse.ArgumentParser(description="VIGIL experiment runner")
    p.add_argument("--input", required=True,
                   help="video file, stream URL, or webcam index")
    p.add_argument("--runs", nargs="+", default=["A", "B", "C", "D"])
    p.add_argument("--sweep", action="store_true",
                   help="run a tau sweep instead of conditions A-D")
    p.add_argument("--sweep-values", nargs="+", type=float,
                   default=[0.80, 0.86, 0.92, 0.96, 0.99])
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--provider", default=os.environ.get("VIGIL_PROVIDER",
                                                        "mock"))
    p.add_argument("--max-seconds", type=float, default=None,
                   help="stop each run after this many seconds")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--speed", type=float, default=None,
                   help="replay a file faster than real time (dev only)")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--log-dir", default="logs")
    return p.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main_async(parse_args()))
    except KeyboardInterrupt:
        print("\ninterrupted")
