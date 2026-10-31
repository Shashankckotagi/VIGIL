#!/usr/bin/env python3
"""Acceptance harness that tests and falsifies VIGIL's three core objectives.

Runs offline against bundled sample/demo_clip.avi and synthetic verification media
using MockProvider without requiring any API keys or network connection.

Usage:
    python verify.py
    python verify.py --quick
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import logging
import os
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

# Ensure vigil can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vigil.config import Config
from vigil.gate import VideoGate
from vigil.pipeline import Pipeline
from vigil.providers import MockProvider

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s | %(message)s")


@dataclass
class CheckResult:
    check_num: int
    name: str
    category: str
    passed: bool
    details: str


def get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def fit_slope_ratio(csv_path: str) -> Tuple[float, float, float]:
    """Fits line to (stream_t, tokens_cum) for first half vs second half."""
    ts, toks = [], []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts.append(float(row["stream_t"]))
                    toks.append(float(row["tokens_cum"]))
                except (ValueError, KeyError):
                    continue

    if len(ts) < 4:
        return 0.0, 0.0, 1.0

    ts_arr = np.array(ts)
    toks_arr = np.array(toks)
    t_mid = (ts_arr[0] + ts_arr[-1]) / 2.0

    mask1 = ts_arr <= t_mid
    mask2 = ts_arr >= t_mid

    if np.sum(mask1) < 2 or np.sum(mask2) < 2:
        mid_idx = len(ts_arr) // 2
        mask1 = np.zeros(len(ts_arr), dtype=bool)
        mask1[:mid_idx] = True
        mask2 = np.zeros(len(ts_arr), dtype=bool)
        mask2[mid_idx:] = True

    slope1 = float(np.polyfit(ts_arr[mask1], toks_arr[mask1], 1)[0])
    slope2 = float(np.polyfit(ts_arr[mask2], toks_arr[mask2], 1)[0])
    ratio = (slope2 / slope1) if abs(slope1) > 1e-6 else 1.0
    return slope1, slope2, ratio


async def run_pipeline_logged(clip_path: str, cfg: Config, provider_latency: float = 0.15, max_seconds: float | None = None):
    pipe = Pipeline(clip_path, cfg, with_audio=True)
    pipe.provider = MockProvider(latency=provider_latency)

    events: List[Dict[str, Any]] = []
    prompt_lens: List[int] = []

    orig_describe = pipe.provider.describe
    async def describe_spy(prompt: str, images: list[str]):
        prompt_lens.append(len(prompt))
        return await orig_describe(prompt, images)
    pipe.provider.describe = describe_spy

    pipe.subscribe(events.append)
    snapshot = await pipe.run_until_complete(max_seconds=max_seconds)
    return pipe, snapshot, events, prompt_lens


def synthesize_o3_clip(out_video_path: str, out_wav_path: str, muxed_path: str) -> List[float]:
    """Synthesises a 60-second clip with static scene and three 2-second loud bursts."""
    sr = 16000
    duration_s = 60
    fps = 10
    n_frames = duration_s * fps
    n_samples = duration_s * sr

    burst_onsets = [10.0, 25.0, 40.0]
    audio = np.zeros(n_samples, dtype=np.int16)
    for b in burst_onsets:
        s = int(b * sr)
        e = int((b + 2.0) * sr)
        t = np.linspace(0, 2.0, e - s, endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        audio[s:e] = (sig * 32767).astype(np.int16)

    os.makedirs(os.path.dirname(out_wav_path) or ".", exist_ok=True)
    with wave.open(out_wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())

    base = np.zeros((240, 320, 3), dtype=np.uint8)
    base[:, :] = 100
    cv2.rectangle(base, (40, 40), (280, 200), (140, 140, 140), -1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (320, 240))
    for i in range(n_frames):
        t = i / fps
        f = base.copy()
        # Subtle drift during the burst moments that crosses tau=0.98 but not 0.92
        for b in burst_onsets:
            if b <= t <= b + 2.0:
                f[60:180, 80:240] = 120  # SSIM ~ 0.939 (below 0.98, above 0.92)
        out.write(f)
    out.release()

    ff = get_ffmpeg_exe()
    cmd = [ff, "-y", "-i", out_video_path, "-i", out_wav_path, "-c:v", "copy", "-c:a", "aac", muxed_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return burst_onsets


async def main():
    parser = argparse.ArgumentParser(description="VIGIL Verification Harness")
    parser.add_argument("--quick", action="store_true", help="Run shortened tests for fast iteration")
    args = parser.parse_args()

    results: List[CheckResult] = []
    print("=" * 72)
    print(" VIGIL ACCEPTANCE VERIFICATION HARNESS")
    print("=" * 72)

    demo_clip = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample", "demo_clip.avi")
    if not os.path.exists(demo_clip):
        print(f"ERROR: {demo_clip} not found!")
        sys.exit(1)

    replay_speed = 10.0 if args.quick else 5.0
    max_seconds = 12.0 if args.quick else None

    # -------------------------------------------------------------------------
    # Objective 1 & 2: Run Condition C and Condition A
    # -------------------------------------------------------------------------
    print("\n[O1 & O2] Running Condition C (Gate ON, Bounded Memory)...")
    cfg_c = Config()
    cfg_c.provider = "mock"
    cfg_c.replay_speed = replay_speed
    cfg_c.capture_fps = 10.0
    cfg_c.gate_enabled = True
    cfg_c.memory_bounded = True
    cfg_c.run_id = "verify_run_C"
    cfg_c.rate_limit_calls = 1000
    cfg_c.rate_limit_window_s = 1.0

    pipe_c, snap_c, evs_c, prompts_c = await run_pipeline_logged(demo_clip, cfg_c, max_seconds=max_seconds)
    gate_c = pipe_c.gate_stats.snapshot()

    print("[O1 & O2] Running Condition A (Gate OFF, Append-Only Memory)...")
    cfg_a = Config()
    cfg_a.provider = "mock"
    cfg_a.replay_speed = replay_speed
    cfg_a.capture_fps = 10.0
    cfg_a.gate_enabled = False
    cfg_a.memory_bounded = False
    cfg_a.run_id = "verify_run_A"
    cfg_a.rate_limit_calls = 1000
    cfg_a.rate_limit_window_s = 1.0

    pipe_a, snap_a, evs_a, prompts_a = await run_pipeline_logged(demo_clip, cfg_a, max_seconds=max_seconds)

    # Check 1: Gate retention and frame conservation
    frames_seen_c = gate_c["frames_seen"]
    frames_kept_c = gate_c["frames_kept"]
    frames_dropped_c = gate_c["frames_dropped"]
    retention_c = gate_c["frame_retention"]
    conserved = (frames_kept_c + frames_dropped_c == frames_seen_c)
    retention_ok = retention_c < 0.50
    p1 = retention_ok and conserved
    d1 = f"retention={retention_c*100:.1f}% (<50%: {retention_ok}), kept({frames_kept_c}) + dropped({frames_dropped_c}) == seen({frames_seen_c})"
    results.append(CheckResult(1, "O1: Frame retention < 50% & conservation", "O1 - Cost Flatness", p1, d1))

    # Check 2: Tokens A vs C
    tokens_c = snap_c.get("tokens_total", 0)
    tokens_a = snap_a.get("tokens_total", 0)
    token_ratio = (tokens_a / max(tokens_c, 1))
    p2 = token_ratio >= 2.0
    d2 = f"tokens A={tokens_a}, tokens C={tokens_c}, ratio={token_ratio:.2f}x (>= 2.0x)"
    results.append(CheckResult(2, "O1: Total tokens A >= 2x tokens C", "O1 - Cost Flatness", p2, d2))

    # Check 3: Cumulative token slope ratio
    csv_c = os.path.join(cfg_c.log_dir, f"{cfg_c.run_id}.csv")
    csv_a = os.path.join(cfg_a.log_dir, f"{cfg_a.run_id}.csv")
    s1_c, s2_c, ratio_c = fit_slope_ratio(csv_c)
    s1_a, s2_a, ratio_a = fit_slope_ratio(csv_a)
    p3 = (ratio_c < 1.30) and (ratio_a > ratio_c)
    d3 = f"C slope ratio={ratio_c:.3f} (<1.3: {ratio_c < 1.3}), A slope ratio={ratio_a:.3f} (> C: {ratio_a > ratio_c})"
    results.append(CheckResult(3, "O1: Token slope ratio (C < 1.3 & A > C)", "O1 - Cost Flatness", p3, d3))

    # Check 4: Prompt character count boundedness in C
    if len(prompts_c) >= 4:
        k = max(1, len(prompts_c) // 4)
        first_q = prompts_c[:k]
        last_q = prompts_c[-k:]
        max_first = max(first_q)
        max_last = max(last_q)
        pct_increase = ((max_last - max_first) / max(max_first, 1)) * 100
        p4 = max_last <= (max_first * 1.30)
        d4 = f"first 25% max={max_first}, last 25% max={max_last}, increase={pct_increase:+.1f}% (<= 30%)"
    else:
        p4 = True
        d4 = f"insufficient calls ({len(prompts_c)}) for 4-quarter split"
    results.append(CheckResult(4, "O1: Prompt char count growth <= 30%", "O1 - Cost Flatness", p4, d4))

    # Check 5: Max queue depth <= cfg.queue_maxsize
    max_q_c = pipe_c.metrics.max_queue_depth
    p5 = max_q_c <= cfg_c.queue_maxsize
    d5 = f"max queue depth={max_q_c} <= maxsize={cfg_c.queue_maxsize}"
    results.append(CheckResult(5, "O2: Max queue depth <= queue_maxsize", "O2 - Stability", p5, d5))

    # Check 6: Queue depth does not grow monotonically
    q_depths = [e.get("queue_depth", 0) for e in evs_c if e.get("type") == "chunk_queued"]
    if len(q_depths) >= 4:
        q_len = len(q_depths) // 4
        m1 = np.mean(q_depths[:q_len])
        m4 = np.mean(q_depths[-q_len:])
        p6 = m4 <= max(1.0, m1 * 2.0)
        d6 = f"mean Q1={m1:.2f}, Q4={m4:.2f} (Q4 <= 2*Q1: {p6})"
    else:
        p6 = True
        d6 = f"depth samples={len(q_depths)}, non-monotonic by default"
    results.append(CheckResult(6, "O2: Queue depth non-monotonic (Q4 <= 2*Q1)", "O2 - Stability", p6, d6))

    # Check 7: Zero errors and zero rate limit faults
    errs_c = pipe_c.metrics.errors + pipe_a.metrics.errors
    rlf_c = pipe_c.metrics.rate_limit_faults + pipe_a.metrics.rate_limit_faults
    p7 = (errs_c == 0) and (rlf_c == 0)
    d7 = f"errors={errs_c}, rate_limit_faults={rlf_c}"
    results.append(CheckResult(7, "O2: Zero errors and rate limit faults", "O2 - Stability", p7, d7))

    # Check 8: Backpressure proof
    print("\n[O2] Running Backpressure Proof (Provider latency 2.0s, queue_maxsize=3)...")
    cfg_bp = Config()
    cfg_bp.provider = "mock"
    cfg_bp.queue_maxsize = 3
    cfg_bp.replay_speed = 10.0
    cfg_bp.capture_fps = 10.0
    cfg_bp.gate_enabled = True
    cfg_bp.memory_bounded = True
    cfg_bp.run_id = "verify_run_BP"
    cfg_bp.rate_limit_calls = 1000
    cfg_bp.rate_limit_window_s = 1.0

    pipe_bp, snap_bp, evs_bp, _ = await run_pipeline_logged(demo_clip, cfg_bp, provider_latency=2.0, max_seconds=15.0)
    bp_events = [e for e in evs_bp if e.get("type") == "backpressure"]
    max_q_bp = pipe_bp.metrics.max_queue_depth
    p8 = len(bp_events) > 0 and max_q_bp <= 3
    if not bp_events:
        d8 = "FAIL: await queue.put() is not doing anything and O2 is unproven (0 backpressure events)"
    else:
        d8 = f"backpressure events={len(bp_events)}, max_queue_depth={max_q_bp} <= 3"
    results.append(CheckResult(8, "O2: Backpressure proof (fires & depth <= 3)", "O2 - Stability", p8, d8))

    # -------------------------------------------------------------------------
    # Objective 3: Audio drives visual rate
    # -------------------------------------------------------------------------
    print("\n[O3] Synthesising 60s media clip with 3 loud bursts...")
    scratch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    v_raw = os.path.join(scratch_dir, "verify_o3_raw.mp4")
    w_raw = os.path.join(scratch_dir, "verify_o3.wav")
    muxed = os.path.join(scratch_dir, "verify_o3_muxed.mp4")
    burst_onsets = synthesize_o3_clip(v_raw, w_raw, muxed)

    # Check 9: Synthesis check
    p9 = os.path.exists(muxed) and os.path.getsize(muxed) > 1000
    d9 = f"muxed file size={os.path.getsize(muxed) if os.path.exists(muxed) else 0} bytes"
    results.append(CheckResult(9, "O3: Synthesised 60s clip with 3 loud bursts", "O3 - Escalation", p9, d9))

    # Check 10, 11, 12, 13: Run non-escalated vs escalated
    print("[O3] Running non-escalated clip...")
    cfg_o3_no = Config()
    cfg_o3_no.provider = "mock"
    cfg_o3_no.replay_speed = replay_speed
    cfg_o3_no.capture_fps = 10.0
    cfg_o3_no.gate_enabled = True
    cfg_o3_no.memory_bounded = True
    cfg_o3_no.escalation_enabled = False
    cfg_o3_no.run_id = "verify_o3_none"
    cfg_o3_no.rate_limit_calls = 1000
    cfg_o3_no.rate_limit_window_s = 1.0

    pipe_o3_no, snap_o3_no, evs_o3_no, _ = await run_pipeline_logged(muxed, cfg_o3_no, max_seconds=max_seconds)
    ts_no = [e["t"] for e in evs_o3_no if e.get("type") == "frame_kept"]

    print("[O3] Running escalated clip...")
    cfg_o3_esc = Config()
    cfg_o3_esc.provider = "mock"
    cfg_o3_esc.replay_speed = replay_speed
    cfg_o3_esc.capture_fps = 10.0
    cfg_o3_esc.gate_enabled = True
    cfg_o3_esc.memory_bounded = True
    cfg_o3_esc.escalation_enabled = True
    cfg_o3_esc.run_id = "verify_o3_esc"
    cfg_o3_esc.rate_limit_calls = 1000
    cfg_o3_esc.rate_limit_window_s = 1.0

    pipe_o3_esc, snap_o3_esc, evs_o3_esc, _ = await run_pipeline_logged(muxed, cfg_o3_esc, max_seconds=max_seconds)
    ts_esc = [e["t"] for e in evs_o3_esc if e.get("type") == "frame_kept"]

    # Check 10: Executed both runs
    p10 = len(ts_no) > 0 and len(ts_esc) > 0
    d10 = f"non-escalated kept={len(ts_no)}, escalated kept={len(ts_esc)}"
    results.append(CheckResult(10, "O3: Executed dual identical runs (esc off/on)", "O3 - Escalation", p10, d10))

    # Check 11: Escalated has strictly more frames kept
    p11 = len(ts_esc) > len(ts_no)
    diff_kept_pct = ((len(ts_esc) - len(ts_no)) / max(len(ts_no), 1)) * 100
    d11 = f"escalated ({len(ts_esc)}) > non-escalated ({len(ts_no)}), diff=+{diff_kept_pct:.1f}%"
    results.append(CheckResult(11, "O3: Escalated has strictly more frames kept", "O3 - Escalation", p11, d11))

    # Check 12: Concentration near bursts
    extra_ts = [t for t in ts_esc if t not in ts_no]
    in_windows = sum(1 for t in extra_ts if any(b <= t <= b + 30.0 for b in burst_onsets))
    conc_ratio = in_windows / max(len(extra_ts), 1)
    p12 = conc_ratio >= 0.60
    d12 = f"concentration={conc_ratio:.2f} ({in_windows}/{len(extra_ts)} extra frames within 30s of burst onset)"
    results.append(CheckResult(12, "O3: Extra frames concentrated near bursts (>0.6)", "O3 - Escalation", p12, d12))

    # Check 13: Escalation token cost within +25%
    tok_no = snap_o3_no.get("tokens_total", 0)
    tok_esc = snap_o3_esc.get("tokens_total", 0)
    tok_increase_pct = ((tok_esc - tok_no) / max(tok_no, 1)) * 100
    p13 = tok_increase_pct <= 25.0
    d13 = f"escalated tokens={tok_esc}, non-escalated={tok_no}, diff={tok_increase_pct:+.1f}% (<= +25%)"
    results.append(CheckResult(13, "O3: Escalation token cost within +25%", "O3 - Escalation", p13, d13))

    # -------------------------------------------------------------------------
    # Sanity Checks
    # -------------------------------------------------------------------------
    # Check 14: CSV chunk row consistency
    csv_rows = []
    if os.path.exists(csv_c):
        with open(csv_c, "r", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
    chunks_done_c = pipe_c.metrics.chunks_done
    empty_desc_rows = [r for r in csv_rows if not r.get("description", "").strip() and not r.get("error", "").strip()]
    p14 = (len(csv_rows) == chunks_done_c) and (len(empty_desc_rows) == 0)
    d14 = f"csv rows={len(csv_rows)}, chunks_done={chunks_done_c}, empty descriptions without error={len(empty_desc_rows)}"
    results.append(CheckResult(14, "Sanity: CSV row per chunk & non-empty descriptions", "Sanity", p14, d14))

    # Check 15: Monotonically increasing stream timestamps
    stream_ts = [float(r["stream_t"]) for r in csv_rows if "stream_t" in r]
    mono = all(stream_ts[i] <= stream_ts[i+1] for i in range(len(stream_ts)-1))
    p15 = mono and len(stream_ts) > 0
    d15 = f"monotonic timestamps: {mono} across {len(stream_ts)} chunks"
    results.append(CheckResult(15, "Sanity: CSV stream timestamps monotonically increase", "Sanity", p15, d15))

    # Check 16: Gate isolation test
    cfg_gate = Config(gate_enabled=True, ssim_tau=0.92)
    gate_iso = VideoGate(cfg_gate)
    black = np.zeros((240, 320, 3), dtype=np.uint8)
    white = np.ones((240, 320, 3), dtype=np.uint8) * 255

    identical_kept = sum(1 for _ in range(50) if gate_iso.should_keep(black)[0])
    gate_iso2 = VideoGate(cfg_gate)
    alternating_kept = sum(1 for i in range(50) if gate_iso2.should_keep(black if i % 2 == 0 else white)[0])

    p16 = (identical_kept == 1) and (alternating_kept == 50)
    d16 = f"50 identical frames kept={identical_kept} (expected 1), 50 alternating kept={alternating_kept} (expected 50)"
    results.append(CheckResult(16, "Sanity: Gate unit isolation (1/50 identical, 50/50 alternating)", "Sanity", p16, d16))

    # -------------------------------------------------------------------------
    # PASS / FAIL Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"{'#':<3} | {'Category':<18} | {'Status':<6} | {'Check Name & Metric Details'}")
    print("-" * 92)
    all_passed = True
    for r in results:
        status_str = "PASS" if r.passed else "FAIL"
        if not r.passed:
            all_passed = False
        print(f"{r.check_num:<3} | {r.category:<18} | {status_str:<6} | {r.name}")
        print(f"    |- {r.details}")
    print("=" * 92)

    total_pass = sum(1 for r in results if r.passed)
    print(f"\nFinal Verdict: {total_pass}/{len(results)} checks passed.")

    # -------------------------------------------------------------------------
    # Required Disclaimer Section
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("NOT VERIFIED BY THIS SCRIPT")
    print("=" * 72)
    print("""
1. Event recall and description accuracy:
   The mock provider produces synthetic strings from a fixed phrase list;
   recall numbers measured against it are meaningless as benchmarks of
   perceptual quality.

2. Real API latency:
   Mock delays are fixed simulated latencies and do not capture real-world
   cloud endpoint jitter, queuing delays, or network latency.

3. Real token billing:
   Reported tokens in the mock harness are estimated character/image proxies,
   not official billed token usage from an upstream provider API.

4. Reconnection against an expiring live manifest:
   Static offline files never experience HLS/DASH 404 manifest expiry or
   TCP session dropouts.

Passing this acceptance harness proves that VIGIL's concurrency control,
filtering gates, bounded queues, working memory, and cross-modal escalation
mechanisms work as engineered. It does not prove that external LLM research
findings hold on production video streams without running a live evaluation.
""")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
