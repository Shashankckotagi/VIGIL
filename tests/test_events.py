"""Acceptance tests for observable pipeline event stream.

Runs the pipeline on sample/demo_clip.avi with MockProvider, subscribes to
all emitted events, and asserts that every pipeline action is observable with
accurate fields, costs, and frame count conservation.

Run with:
    python tests/test_events.py
    python -m pytest tests/test_events.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vigil.config import Config
from vigil.pipeline import Pipeline


def test_all_pipeline_events_observable():
    """Runs on sample/demo_clip.avi and asserts all required events appear with correct cost."""
    clip_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sample", "demo_clip.avi"
    )
    assert os.path.exists(clip_path), f"Sample clip not found: {clip_path}"

    cfg = Config()
    cfg.provider = "mock"
    cfg.capture_fps = 10.0
    cfg.replay_speed = 12.0
    cfg.chunk_window_s = 1.0
    cfg.chunk_max_frames = 2
    cfg.memory_bounded = True
    cfg.memory_recent_k = 1
    cfg.memory_compact_every = 1
    cfg.rate_limit_calls = 2
    cfg.rate_limit_window_s = 0.5
    cfg.emit_verbose_events = True

    events: list[dict] = []
    pipe = Pipeline(clip_path, cfg, with_audio=True)
    pipe.subscribe(events.append)

    asyncio.run(pipe.run_until_complete(max_seconds=8))

    # 1. Assert at least one of each required event type appears
    required_events = [
        "gate_drop_run",
        "frame_kept",
        "audio_speech",
        "audio_silence",
        "chunk_queued",
        "memory_compacted",
        "rate_limit_wait",
    ]
    for req in required_events:
        matches = [e for e in events if e.get("type") == req]
        assert len(matches) > 0, f"Expected at least one '{req}' event, got none."

    # 2. Assert no gate_drop_run has count == 0 and fields are present
    for e in events:
        if e["type"] == "gate_drop_run":
            assert e.get("count", 0) > 0, f"gate_drop_run must have count > 0, got {e}"
            assert "t_start" in e and "t_end" in e, f"gate_drop_run missing timestamps: {e}"
            assert "ssim_min" in e and "ssim_max" in e, f"gate_drop_run missing ssim range: {e}"

        elif e["type"] == "frame_kept":
            assert "t" in e and "ssim" in e and "tau" in e and "reason" in e
            assert e["reason"] in ("change", "escalated")

        elif e["type"] == "audio_speech":
            assert "t_start" in e and "t_end" in e and "duration_ms" in e and "peak_rms" in e

        elif e["type"] == "audio_silence":
            assert "t_start" in e and "t_end" in e and "duration_ms" in e and "frames_dropped" in e

        elif e["type"] == "chunk_queued":
            assert "chunk" in e and "t" in e and "images" in e and "speech_ms" in e and "queue_depth" in e

        elif e["type"] == "memory_compacted":
            assert "compaction_number" in e and "events_folded" in e and "summary_chars" in e

        elif e["type"] == "rate_limit_wait":
            assert "waited_s" in e and e["waited_s"] > 0

    # 3. Assert frame count conservation: sum(drop counts) + sum(frame_kept) == frames_seen
    total_drops = sum(e["count"] for e in events if e["type"] == "gate_drop_run")
    total_kept = sum(1 for e in events if e["type"] == "frame_kept")
    frames_seen = pipe.gate_stats.frames_seen
    assert total_drops + total_kept == frames_seen, (
        f"Frame count mismatch: {total_drops} (drops) + {total_kept} (kept) != {frames_seen} (seen)"
    )

    # 4. Assert every event carries a cost field: "free" or "billed"
    for e in events:
        assert "cost" in e, f"Event {e} missing required 'cost' field"
        if e["type"] in ("description", "memory_compacted"):
            assert e["cost"] == "billed", f"Event {e['type']} should have cost 'billed', got {e['cost']}"
        else:
            assert e["cost"] == "free", f"Event {e['type']} should have cost 'free', got {e['cost']}"


def test_verbose_events_flag():
    """Assert cfg.emit_verbose_events=False suppresses high-rate events."""
    clip_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sample", "demo_clip.avi"
    )
    cfg = Config()
    cfg.provider = "mock"
    cfg.capture_fps = 10.0
    cfg.replay_speed = 15.0
    cfg.chunk_window_s = 1.0
    cfg.chunk_max_frames = 2
    cfg.emit_verbose_events = False

    events: list[dict] = []
    pipe = Pipeline(clip_path, cfg, with_audio=False)
    pipe.subscribe(events.append)

    asyncio.run(pipe.run_until_complete(max_seconds=3))

    verbose_types = {"frame", "gate_drop_run", "frame_kept", "chunk_queued", "rate_limit_wait"}
    emitted_types = {e.get("type") for e in events}
    assert not (verbose_types & emitted_types), f"Verbose events emitted when flag is False: {verbose_types & emitted_types}"


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)

