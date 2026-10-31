"""Level-0 gate. The project's central claim lives in this file.

No neural network runs here. Video gating is structural similarity on a
128x128 grayscale thumbnail; audio gating is WebRTC VAD. Both are cheap
enough to run on a Raspberry Pi beside the camera, which is the whole point:
every published streaming gate (StreamMem, StreamMind, STC) runs a vision
encoder on every frame to decide what to drop, and therefore needs a GPU per
stream. The open question this project measures is what recall that costs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAVE_SKIMAGE = True
except ImportError:  # pragma: no cover
    _HAVE_SKIMAGE = False

log = logging.getLogger("vigil.gate")


def _fallback_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM on uint8 images, used when scikit-image is absent."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    return float(num / den) if den else 1.0


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    if _HAVE_SKIMAGE:
        return float(_ssim(a, b, data_range=255))
    return _fallback_ssim(a, b)


@dataclass
class GateStats:
    frames_seen: int = 0
    frames_kept: int = 0
    audio_frames_seen: int = 0
    audio_frames_kept: int = 0
    escalations: int = 0
    ssim_scores: list = field(default_factory=list)

    @property
    def frame_retention(self) -> float:
        return self.frames_kept / self.frames_seen if self.frames_seen else 0.0

    @property
    def audio_retention(self) -> float:
        if not self.audio_frames_seen:
            return 0.0
        return self.audio_frames_kept / self.audio_frames_seen

    def snapshot(self) -> dict:
        return {
            "frames_seen": self.frames_seen,
            "frames_kept": self.frames_kept,
            "frames_dropped": self.frames_seen - self.frames_kept,
            "frame_retention": round(self.frame_retention, 4),
            "audio_frames_seen": self.audio_frames_seen,
            "audio_frames_kept": self.audio_frames_kept,
            "audio_retention": round(self.audio_retention, 4),
            "escalations": self.escalations,
        }


class VideoGate:
    """Keep a frame only when it differs enough from the last kept frame.

    Comparing against the last *kept* frame rather than the immediately
    previous frame matters: it makes slow drift accumulate until it crosses
    the threshold, instead of never triggering because each step is tiny.
    """

    def __init__(self, cfg, stats: GateStats | None = None):
        self.cfg = cfg
        self.stats = stats or GateStats()
        self.tau = cfg.ssim_tau
        self._base_tau = cfg.ssim_tau
        self._last_kept: np.ndarray | None = None

    def set_tau(self, tau: float) -> None:
        self.tau = float(tau)

    def reset_tau(self) -> None:
        self.tau = self._base_tau

    def _thumb(self, frame: np.ndarray) -> np.ndarray:
        n = self.cfg.ssim_size
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return cv2.resize(gray, (n, n), interpolation=cv2.INTER_AREA)

    def should_keep(self, frame: np.ndarray) -> tuple[bool, float]:
        """Return (keep, ssim_score). Score is 1.0 for the first frame."""
        self.stats.frames_seen += 1

        if not self.cfg.gate_enabled:
            self.stats.frames_kept += 1
            return True, 0.0

        thumb = self._thumb(frame)
        if self._last_kept is None:
            self._last_kept = thumb
            self.stats.frames_kept += 1
            return True, 1.0

        score = ssim(self._last_kept, thumb)
        self.stats.ssim_scores.append(score)
        keep = score < self.tau
        if keep:
            self._last_kept = thumb
            self.stats.frames_kept += 1
        return keep, score


class AudioGate:
    """WebRTC VAD wrapper. Also reports RMS, which escalation keys on."""

    def __init__(self, cfg, stats: GateStats | None = None):
        self.cfg = cfg
        self.stats = stats or GateStats()
        self._vad = None
        if cfg.vad_enabled:
            if cfg.vad_frame_ms not in (10, 20, 30):
                raise ValueError("vad_frame_ms must be 10, 20 or 30")
            try:
                import webrtcvad
                self._vad = webrtcvad.Vad(cfg.vad_aggressiveness)
            except ImportError:  # pragma: no cover
                log.warning("webrtcvad missing; audio gate passes everything")

        self._expected = int(cfg.audio_sample_rate * 2
                             * (cfg.vad_frame_ms / 1000.0))

    @staticmethod
    def rms(pcm: bytes) -> float:
        """Normalised RMS in 0..1 for 16-bit mono PCM."""
        if not pcm:
            return 0.0
        a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if a.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(a * a)))

    def is_speech(self, pcm: bytes) -> tuple[bool, float]:
        """Return (speech, rms). VAD needs an exact frame size."""
        self.stats.audio_frames_seen += 1
        level = self.rms(pcm)

        if self._vad is None or not self.cfg.vad_enabled:
            speech = level > 0.02
            if speech:
                self.stats.audio_frames_kept += 1
            return speech, level

        if len(pcm) != self._expected:
            pcm = (pcm[:self._expected] if len(pcm) > self._expected
                   else pcm + b"\x00" * (self._expected - len(pcm)))
        try:
            speech = self._vad.is_speech(pcm, self.cfg.audio_sample_rate)
        except Exception:
            speech = level > 0.01
        if speech:
            self.stats.audio_frames_kept += 1
        return bool(speech), level
