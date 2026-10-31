"""Model backends behind one async interface.

The project targets black-box APIs: the only levers are what we send and how
often. That constraint is the research setting, so the interface deliberately
exposes nothing but text in, text out, and the provider's own token counts.

  MockProvider    no network. Deterministic. Used by tests and by anyone
                  developing the pipeline without burning quota.
  OllamaProvider  local vision model. Free, slower, good for long runs.
  GeminiProvider  the real measurement target. usageMetadata is where every
                  token number in the results comes from.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass

log = logging.getLogger("vigil.providers")


@dataclass
class Completion:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    error: str | None = None


class ProviderError(RuntimeError):
    pass


class RateLimitError(ProviderError):
    """Raised on HTTP 429 or quota exhaustion so the pipeline can back off."""


class BaseProvider:
    name = "base"

    async def describe(self, prompt: str, images: list[str]) -> Completion:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class MockProvider(BaseProvider):
    name = "mock"

    _PHRASES = [
        "The view is unchanged.",
        "A person walks across the frame from left to right.",
        "A vehicle passes in the background.",
        "The door opens and someone steps through.",
        "Lighting shifts slightly; nothing else changes.",
        "Two people stop and appear to talk.",
    ]

    def __init__(self, cfg=None, latency: float = 0.15, seed: int = 0):
        self.cfg = cfg
        self.latency = latency
        self._rng = random.Random(seed)
        self.calls = 0

    def _analyze_actual_frames(self, images: list[str]) -> str | None:
        """Extract real computer vision insights from the chunk's frames."""
        if not images:
            return None
        try:
            import base64
            import cv2
            import numpy as np

            decoded = []
            for img_b64 in images[-2:]:
                raw = base64.b64decode(img_b64)
                arr = np.frombuffer(raw, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    decoded.append(img)

            if not decoded:
                return None

            last = decoded[-1]
            h, w, _ = last.shape
            gray = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
            mean_b = float(np.mean(gray))
            lighting = "bright daylight" if mean_b > 135 else (
                "moderate ambient light" if mean_b > 65 else "dim / night scene"
            )

            if len(decoded) >= 2:
                prev_gray = cv2.cvtColor(decoded[0], cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion_px = cv2.countNonZero(thresh)
                ratio = motion_px / float(w * h)

                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                sig = [c for c in contours if cv2.contourArea(c) > 120]

                if ratio > 0.12:
                    regions = []
                    for c in sig[:3]:
                        x, y, cw, ch = cv2.boundingRect(c)
                        rx = "left" if x < w / 3 else ("center" if x < 2 * w / 3 else "right")
                        ry = "upper" if y < h / 3 else ("mid" if y < 2 * h / 3 else "lower")
                        regions.append(f"{ry}-{rx}")
                    region_str = ", ".join(dict.fromkeys(regions)) if regions else "broad frame"
                    return f"Significant movement detected ({ratio * 100:.1f}% scene change) in {region_str}. {lighting.capitalize()}."
                elif ratio > 0.025:
                    return f"Active motion observed ({ratio * 100:.1f}% change, {len(sig)} dynamic object{'s' if len(sig) != 1 else ''}). {lighting.capitalize()}."
                elif ratio > 0.003:
                    return f"Subtle motion in view ({ratio * 100:.2f}% shift). View remains largely stable under {lighting}."
                else:
                    return f"Stable scene; no significant visual movement detected. Scene under {lighting}."
            else:
                return f"Stream active ({w}x{h}). Ambient illumination: {lighting}."
        except Exception:
            return None

    async def describe(self, prompt: str, images: list[str]) -> Completion:
        t0 = time.monotonic()
        await asyncio.sleep(self.latency)
        self.calls += 1

        # Real vision analysis if real frames are present, else fallback
        text = self._analyze_actual_frames(images)
        if not text:
            text = self._rng.choice(self._PHRASES)

        # Rough but stable token proxy: ~4 chars/token, ~258 tokens per image.
        tokens_in = len(prompt) // 4 + 258 * len(images)
        return Completion(text=text, tokens_in=tokens_in,
                          tokens_out=max(len(text) // 4, 1),
                          latency_s=time.monotonic() - t0)


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, cfg):
        import httpx
        self.cfg = cfg
        self._client = httpx.AsyncClient(timeout=cfg.request_timeout_s)

    async def describe(self, prompt: str, images: list[str]) -> Completion:
        t0 = time.monotonic()
        payload = {
            "model": self.cfg.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        if images:
            payload["images"] = images
        url = f"{self.cfg.ollama_host.rstrip('/')}/api/generate"
        try:
            r = await self._client.post(url, json=payload)
        except Exception as e:
            raise ProviderError(f"ollama unreachable: {e}") from e
        if r.status_code == 429:
            raise RateLimitError("ollama returned 429")
        if r.status_code >= 400:
            raise ProviderError(f"ollama HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        return Completion(
            text=(data.get("response") or "").strip(),
            tokens_in=int(data.get("prompt_eval_count") or 0),
            tokens_out=int(data.get("eval_count") or 0),
            latency_s=time.monotonic() - t0,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, cfg):
        import httpx
        if not cfg.gemini_api_key:
            raise ProviderError("GEMINI_API_KEY is not set")
        self.cfg = cfg
        self._client = httpx.AsyncClient(timeout=cfg.request_timeout_s)
        self._url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{cfg.gemini_model}:generateContent"
        )

    async def describe(self, prompt: str, images: list[str]) -> Completion:
        t0 = time.monotonic()
        parts: list[dict] = [{"text": prompt}]
        for b64 in images:
            parts.append({"inline_data": {"mime_type": "image/jpeg",
                                          "data": b64}})
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 200},
        }
        try:
            r = await self._client.post(
                self._url, json=body,
                headers={"x-goog-api-key": self.cfg.gemini_api_key},
            )
        except Exception as e:
            raise ProviderError(f"gemini unreachable: {e}") from e

        if r.status_code in (429, 503):
            raise RateLimitError(f"gemini HTTP {r.status_code}")
        if r.status_code >= 400:
            raise ProviderError(f"gemini HTTP {r.status_code}: {r.text[:300]}")

        data = r.json()
        text = ""
        try:
            cand = data["candidates"][0]
            text = "".join(p.get("text", "")
                           for p in cand["content"]["parts"]).strip()
        except (KeyError, IndexError):
            log.warning("unexpected gemini response: %s",
                        json.dumps(data)[:300])
        usage = data.get("usageMetadata", {}) or {}
        return Completion(
            text=text,
            tokens_in=int(usage.get("promptTokenCount") or 0),
            tokens_out=int(usage.get("candidatesTokenCount") or 0),
            latency_s=time.monotonic() - t0,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def make_provider(cfg) -> BaseProvider:
    name = (cfg.provider or "mock").lower()
    if name == "gemini":
        return GeminiProvider(cfg)
    if name == "ollama":
        return OllamaProvider(cfg)
    if name == "mock":
        return MockProvider(cfg)
    raise ProviderError(f"unknown provider: {cfg.provider}")
