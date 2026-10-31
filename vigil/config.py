"""Central configuration. Every tunable number in VIGIL lives here.

Values can be overridden by environment variables or by passing a Config
instance into Pipeline(). Keeping them in one place matters for the
experiments: a run is fully described by this object plus the input file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field


def _env(key: str, default, cast=str):
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        if cast is bool:
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return cast(raw)
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # ---- capture -------------------------------------------------------
    target_height: int = 480          # requested stream height from yt-dlp
    capture_fps: float = 2.0          # frames per second sampled from source
    audio_sample_rate: int = 16000    # webrtcvad accepts 8k/16k/32k/48k only
    reconnect_max_attempts: int = 10
    reconnect_backoff_s: float = 2.0
    replay_speed: float = 1.0         # >1 replays a file faster than real time

    # ---- gate (Objective 1) --------------------------------------------
    gate_enabled: bool = True
    ssim_tau: float = 0.92            # forward frame when ssim < tau
    ssim_size: int = 128              # thumbnail edge length for comparison
    vad_enabled: bool = True
    vad_aggressiveness: int = 2       # 0 (permissive) .. 3 (strict)
    vad_frame_ms: int = 30            # MUST be 10, 20 or 30

    # ---- escalation (Objective 3) --------------------------------------
    escalation_enabled: bool = False
    escalation_rms_threshold: float = 0.08   # normalised RMS, 0..1
    escalation_tau: float = 0.98             # relaxed tau while escalated
    escalation_hold_s: float = 30.0

    # ---- chunking ------------------------------------------------------
    chunk_window_s: float = 4.0
    chunk_max_frames: int = 3
    jpeg_quality: int = 70

    # ---- dispatch (Objective 2) ----------------------------------------
    queue_maxsize: int = 10
    max_concurrency: int = 3
    rate_limit_calls: int = 12        # calls allowed per rate_limit_window_s
    rate_limit_window_s: float = 60.0
    request_timeout_s: float = 60.0
    max_retries: int = 4

    # ---- memory (Objective 1) ------------------------------------------
    memory_bounded: bool = True       # False => append-only baseline
    memory_recent_k: int = 5          # descriptions kept verbatim
    memory_compact_every: int = 10    # evictions between summary rewrites
    memory_summary_words: int = 200
    anchor_text: str = (
        "You are watching a single fixed camera view. Describe only what "
        "visibly changes. Be factual and brief."
    )

    # ---- provider ------------------------------------------------------
    provider: str = "mock"            # mock | ollama | gemini
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_key: str = ""
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5vl:7b"

    # ---- run bookkeeping ------------------------------------------------
    run_id: str = "dev"
    log_dir: str = "logs"
    emit_verbose_events: bool = True
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Config":
        c = cls()
        c.provider = _env("VIGIL_PROVIDER", c.provider)
        c.gemini_api_key = _env("GEMINI_API_KEY", c.gemini_api_key)
        c.gemini_model = _env("GEMINI_MODEL", c.gemini_model)
        c.ollama_host = _env("OLLAMA_HOST", c.ollama_host)
        c.ollama_model = _env("OLLAMA_MODEL", c.ollama_model)
        c.ssim_tau = _env("VIGIL_TAU", c.ssim_tau, float)
        c.capture_fps = _env("VIGIL_FPS", c.capture_fps, float)
        c.log_dir = _env("VIGIL_LOG_DIR", c.log_dir)
        c.replay_speed = _env("VIGIL_REPLAY_SPEED", c.replay_speed, float)
        c.emit_verbose_events = _env("VIGIL_EMIT_VERBOSE_EVENTS", c.emit_verbose_events, bool)
        return c

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("gemini_api_key"):
            d["gemini_api_key"] = "***redacted***"
        return d
