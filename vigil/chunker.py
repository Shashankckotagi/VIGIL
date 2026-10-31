"""Chunking and cross-modal escalation.

Chunk       - bundles surviving frames plus the speech heard alongside them
              into one unit of work for the model.
Escalator   - Objective 3. A loud, voiced moment relaxes the video gate's
              threshold for a fixed hold period, so the visual sampling rate
              rises in response to something the pixels could not reveal.
              Published audio-visual work fuses modalities after both have
              been encoded; here the cheap stream governs the expensive one
              before either is encoded.
"""
from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger("vigil.chunker")


@dataclass
class Chunk:
    index: int
    t_start: float
    t_end: float
    frames_b64: list[str] = field(default_factory=list)
    speech_ms: int = 0
    peak_rms: float = 0.0
    escalated: bool = False
    frames_kept: int = 0
    frames_seen: int = 0

    @property
    def payload_bytes(self) -> int:
        return sum(len(f) for f in self.frames_b64)


def encode_frame(frame: np.ndarray, quality: int = 70) -> str:
    ok, buf = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class Escalator:
    """Holds a relaxed tau for escalation_hold_s after an acoustic trigger."""

    def __init__(self, cfg, video_gate, stats=None):
        self.cfg = cfg
        self.gate = video_gate
        self.stats = stats
        self._until: float = 0.0

    @property
    def active(self) -> bool:
        return time.monotonic() < self._until

    def consider(self, speech: bool, rms: float) -> bool:
        """Trigger on a voiced frame that is also loud. Returns True if
        this call started or extended an escalation window."""
        if not self.cfg.escalation_enabled:
            return False
        if not (speech and rms >= self.cfg.escalation_rms_threshold):
            return False

        was_active = self.active
        self._until = time.monotonic() + self.cfg.escalation_hold_s
        self.gate.set_tau(self.cfg.escalation_tau)
        if not was_active:
            if self.stats is not None:
                self.stats.escalations += 1
            log.info("escalation triggered (rms=%.3f) -> tau=%.3f",
                     rms, self.cfg.escalation_tau)
            return True
        return False

    def tick(self) -> None:
        """Called each loop; restores the baseline tau when the hold ends."""
        if self.cfg.escalation_enabled and not self.active:
            self.gate.reset_tau()


class Chunker:
    """Accumulates kept frames and speech into fixed-window chunks."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._frames: list[str] = []
        self._t_start: float | None = None
        self._t_last: float = 0.0
        self._speech_ms = 0
        self._peak_rms = 0.0
        self._escalated = False
        self._seen = 0
        self._kept = 0
        self._index = 0

    def note_frame_seen(self) -> None:
        self._seen += 1

    def add_frame(self, frame: np.ndarray, t: float) -> Chunk | None:
        if self._t_start is None:
            self._t_start = t
        self._t_last = t
        self._kept += 1
        try:
            self._frames.append(encode_frame(frame, self.cfg.jpeg_quality))
        except Exception:
            log.exception("frame encode failed; skipping frame")
            return None
        if len(self._frames) >= self.cfg.chunk_max_frames:
            return self.flush(t)
        return None

    def add_audio(self, speech: bool, rms: float, t: float,
                  escalated: bool = False) -> None:
        if self._t_start is None:
            self._t_start = t
        self._t_last = max(self._t_last, t)
        if speech:
            self._speech_ms += self.cfg.vad_frame_ms
        self._peak_rms = max(self._peak_rms, rms)
        self._escalated = self._escalated or escalated

    def due(self, now: float) -> bool:
        if self._t_start is None:
            return False
        if not self._frames and self._speech_ms == 0:
            return False
        return (now - self._t_start) >= self.cfg.chunk_window_s

    def flush(self, now: float) -> Chunk | None:
        if not self._frames and self._speech_ms == 0:
            self._reset()
            return None
        chunk = Chunk(
            index=self._index,
            t_start=self._t_start if self._t_start is not None else now,
            t_end=max(self._t_last, now),
            frames_b64=list(self._frames),
            speech_ms=self._speech_ms,
            peak_rms=self._peak_rms,
            escalated=self._escalated,
            frames_kept=self._kept,
            frames_seen=self._seen,
        )
        self._index += 1
        self._reset()
        return chunk

    def _reset(self) -> None:
        self._frames = []
        self._t_start = None
        self._speech_ms = 0
        self._peak_rms = 0.0
        self._escalated = False
        self._seen = 0
        self._kept = 0
