"""Tests that run offline with no network and no model.

    python -m pytest tests/ -v        (or: python tests/test_vigil.py)
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vigil.chunker import Chunker, Escalator, encode_frame   # noqa: E402
from vigil.config import Config                              # noqa: E402
from vigil.gate import AudioGate, GateStats, VideoGate, ssim  # noqa: E402
from vigil.memory import AppendOnlyMemory, RollingMemory     # noqa: E402
from vigil.metrics import RateLimiter, RunMetrics            # noqa: E402
from vigil.providers import MockProvider                     # noqa: E402


def _frame(value: int, size: int = 240) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _noisy(value: int, size: int = 240, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((size, size, 3), value, dtype=np.int16)
    base += rng.integers(-2, 3, base.shape, dtype=np.int16)
    return np.clip(base, 0, 255).astype(np.uint8)


# --- gate ---------------------------------------------------------------
def test_identical_frames_are_dropped():
    cfg = Config()
    g = VideoGate(cfg, GateStats())
    assert g.should_keep(_frame(100))[0] is True      # first always kept
    for _ in range(20):
        keep, score = g.should_keep(_frame(100))
        assert keep is False
        assert score > 0.99
    assert g.stats.frames_kept == 1
    assert g.stats.frames_seen == 21
    assert g.stats.frame_retention < 0.05


def test_changed_frame_is_kept():
    cfg = Config()
    g = VideoGate(cfg, GateStats())
    g.should_keep(_frame(0))
    keep, score = g.should_keep(_frame(255))
    assert keep is True
    assert score < cfg.ssim_tau


def test_sensor_noise_does_not_trigger():
    cfg = Config()
    g = VideoGate(cfg, GateStats())
    g.should_keep(_noisy(120, seed=1))
    kept = sum(g.should_keep(_noisy(120, seed=s))[0] for s in range(2, 20))
    assert kept <= 2, "noise alone should rarely cross the threshold"


def test_gate_disabled_keeps_everything():
    cfg = Config()
    cfg.gate_enabled = False
    g = VideoGate(cfg, GateStats())
    for _ in range(10):
        assert g.should_keep(_frame(50))[0] is True
    assert g.stats.frame_retention == 1.0


def test_tau_controls_retention():
    strict, loose = Config(), Config()
    strict.ssim_tau, loose.ssim_tau = 0.50, 0.999
    gs, gl = VideoGate(strict, GateStats()), VideoGate(loose, GateStats())
    frames = [_noisy(100 + (i % 7) * 3, seed=i) for i in range(40)]
    for f in frames:
        gs.should_keep(f)
        gl.should_keep(f)
    assert gl.stats.frames_kept >= gs.stats.frames_kept


def test_ssim_bounds():
    a = _frame(10)[:, :, 0]
    assert ssim(a, a) > 0.99
    assert ssim(a, _frame(250)[:, :, 0]) < 0.5


# --- audio --------------------------------------------------------------
def test_silence_rms_is_zero():
    cfg = Config()
    ag = AudioGate(cfg, GateStats())
    silence = b"\x00\x00" * 480
    assert ag.rms(silence) == 0.0


def test_loud_audio_has_high_rms():
    cfg = Config()
    ag = AudioGate(cfg, GateStats())
    sr, n = cfg.audio_sample_rate, 480
    t = np.arange(n) / sr
    tone = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
    assert ag.rms(tone.tobytes()) > 0.3


def test_audio_gate_handles_wrong_frame_size():
    cfg = Config()
    ag = AudioGate(cfg, GateStats())
    speech, rms = ag.is_speech(b"\x01\x00" * 97)   # deliberately wrong length
    assert isinstance(speech, bool)
    assert rms >= 0.0


# --- escalation ---------------------------------------------------------
def test_escalation_relaxes_tau_then_restores():
    cfg = Config()
    cfg.escalation_enabled = True
    cfg.escalation_hold_s = 0.25
    cfg.escalation_rms_threshold = 0.05
    stats = GateStats()
    gate = VideoGate(cfg, stats)
    esc = Escalator(cfg, gate, stats)

    assert gate.tau == cfg.ssim_tau
    assert esc.consider(speech=True, rms=0.9) is True
    assert gate.tau == cfg.escalation_tau
    assert stats.escalations == 1

    import time as _t
    _t.sleep(0.3)
    esc.tick()
    assert gate.tau == cfg.ssim_tau


def test_escalation_ignores_quiet_speech():
    cfg = Config()
    cfg.escalation_enabled = True
    cfg.escalation_rms_threshold = 0.5
    gate = VideoGate(cfg, GateStats())
    esc = Escalator(cfg, gate, GateStats())
    assert esc.consider(speech=True, rms=0.01) is False
    assert gate.tau == cfg.ssim_tau


def test_escalation_off_by_default():
    cfg = Config()
    gate = VideoGate(cfg, GateStats())
    esc = Escalator(cfg, gate, GateStats())
    assert esc.consider(speech=True, rms=1.0) is False


# --- chunker ------------------------------------------------------------
def test_chunk_emits_at_max_frames():
    cfg = Config()
    cfg.chunk_max_frames = 3
    ch = Chunker(cfg)
    assert ch.add_frame(_frame(10), 0.0) is None
    assert ch.add_frame(_frame(20), 0.5) is None
    chunk = ch.add_frame(_frame(30), 1.0)
    assert chunk is not None
    assert len(chunk.frames_b64) == 3
    assert chunk.payload_bytes > 0


def test_chunk_due_after_window():
    cfg = Config()
    cfg.chunk_window_s = 2.0
    ch = Chunker(cfg)
    ch.add_frame(_frame(10), 0.0)
    assert ch.due(1.0) is False
    assert ch.due(2.5) is True
    chunk = ch.flush(2.5)
    assert chunk is not None and chunk.index == 0


def test_empty_flush_returns_none():
    assert Chunker(Config()).flush(1.0) is None


def test_encode_frame_is_base64_jpeg():
    import base64
    s = encode_frame(_frame(128))
    raw = base64.b64decode(s)
    assert raw[:2] == b"\xff\xd8"      # JPEG SOI marker


# --- memory -------------------------------------------------------------
class _Chunk:
    t_start = 12.0
    speech_ms = 0
    escalated = False


def test_bounded_memory_prompt_stops_growing():
    cfg = Config()
    cfg.memory_recent_k = 5
    m = RollingMemory(cfg)
    sizes = []
    for i in range(80):
        m.append(f"Observation number {i} with some descriptive padding text.")
        sizes.append(len(m.build_prompt(_Chunk())))
    # Prompt size must plateau, not climb with history length.
    assert max(sizes[40:]) <= max(sizes[:40]) * 1.25
    assert len(m.recent) == 5


def test_append_only_memory_grows():
    cfg = Config()
    m = AppendOnlyMemory(cfg)
    first = len(m.build_prompt(_Chunk()))
    for i in range(60):
        m.append(f"Observation number {i} with some descriptive padding text.")
    assert len(m.build_prompt(_Chunk())) > first * 5


def test_compaction_runs_and_sets_summary():
    cfg = Config()
    cfg.memory_recent_k = 2
    cfg.memory_compact_every = 3
    m = RollingMemory(cfg)

    async def go():
        p = MockProvider(cfg, latency=0.0)
        for i in range(12):
            m.append(f"event {i}")
            await m.maybe_compact(p)
        return p

    p = asyncio.run(go())
    assert m.compactions >= 1
    assert m.summary != ""
    assert p.calls >= 1


def test_anchor_always_present():
    cfg = Config()
    for m in (RollingMemory(cfg), AppendOnlyMemory(cfg)):
        assert cfg.anchor_text in m.build_prompt(_Chunk())


# --- metrics ------------------------------------------------------------
def test_rate_limiter_blocks_past_quota():
    async def go():
        rl = RateLimiter(max_calls=3, window_s=0.4)
        waited_total = 0.0
        for _ in range(5):
            waited_total += await rl.acquire()
        return waited_total

    assert asyncio.run(go()) > 0.0


def test_metrics_percentiles_and_hours():
    m = RunMetrics()
    for i in range(100):
        m.record(10, 5, latency=i / 100.0, queue_depth=i % 5, stream_t=i * 60)
    assert m.tokens_total == 1500
    assert 0.9 <= m.percentile(95) <= 1.0
    assert m.max_queue_depth == 4
    assert len(m.tokens_by_hour) >= 2


# --- provider -----------------------------------------------------------
def test_mock_provider_reports_tokens():
    async def go():
        p = MockProvider(Config(), latency=0.0)
        return await p.describe("a prompt", ["aGVsbG8=", "aGVsbG8="])

    c = asyncio.run(go())
    assert c.text
    assert c.tokens_in > 500      # two images dominate the count
    assert c.tokens_out > 0


# --- end to end ---------------------------------------------------------
def test_pipeline_end_to_end_on_synthetic_video():
    """Generates a short clip, runs the full pipeline, checks the CSV."""
    import cv2
    from vigil.pipeline import Pipeline

    tmp = tempfile.mkdtemp(prefix="vigil-test-")
    video = os.path.join(tmp, "clip.avi")
    w, h = 320, 240
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"),
                             10.0, (w, h))
    assert writer.isOpened(), "no usable OpenCV video encoder in this env"
    rng = np.random.default_rng(0)
    for i in range(100):
        # Static for most of the clip, then an abrupt change. Frame size
        # must match the writer exactly or every write silently fails.
        val = 40 if i < 60 else 210
        frame = np.full((h, w, 3), val, dtype=np.int16)
        frame += rng.integers(-2, 3, frame.shape, dtype=np.int16)
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()
    assert os.path.exists(video) and os.path.getsize(video) > 1000

    cfg = Config()
    cfg.provider = "mock"
    cfg.capture_fps = 5.0
    cfg.chunk_window_s = 1.0
    cfg.chunk_max_frames = 2
    cfg.log_dir = tmp
    cfg.run_id = "e2e"
    cfg.rate_limit_calls = 1000

    pipe = Pipeline(video, cfg, with_audio=False)
    summary = asyncio.run(pipe.run_until_complete(max_seconds=40))

    assert summary["chunks_done"] >= 1, "pipeline produced no descriptions"
    assert summary["errors"] == 0
    assert pipe.gate_stats.frames_seen > 0
    assert pipe.gate_stats.frames_kept >= 1
    assert pipe.gate_stats.frame_retention < 1.0, "gate dropped nothing"

    csv_path = os.path.join(tmp, "e2e.csv")
    manifest = os.path.join(tmp, "e2e.json")
    assert os.path.exists(csv_path)
    assert os.path.exists(manifest)

    import csv as _csv
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    assert len(rows) >= 1
    assert rows[0]["description"]


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
