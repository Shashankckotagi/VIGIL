"""Working memory. Objective 1's second half.

Two implementations share one interface so a run can switch between them:

  RollingMemory   bounded. Anchor + recursive summary + last k descriptions.
                  Prompt size converges to a constant.
  AppendOnlyMemory  the run-A baseline. Every description is carried forward,
                  so prompt size grows linearly and token cost grows with it.
                  It exists to be beaten, and to produce the climbing line in
                  the paper's headline figure.

The anchor block is pinned at the head of every prompt, following
StreamingLLM's attention-sink finding: a sliding window without a stable
anchor degrades.
"""
from __future__ import annotations

import logging
from collections import deque

log = logging.getLogger("vigil.memory")

_SUMMARY_INSTRUCTION = (
    "Rewrite the running summary below so it absorbs the new events, keeping "
    "people, objects and ordering. Stay under {words} words. Reply with the "
    "summary text only."
)


class BaseMemory:
    def build_prompt(self, chunk) -> str:
        raise NotImplementedError

    def append(self, description: str) -> None:
        raise NotImplementedError

    async def maybe_compact(self, provider) -> dict | None:
        return None

    def snapshot(self) -> dict:
        raise NotImplementedError


class AppendOnlyMemory(BaseMemory):
    """Unbounded history. Baseline only -- do not use for long runs."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.history: list[str] = []

    def build_prompt(self, chunk) -> str:
        parts = [self.cfg.anchor_text, ""]
        if self.history:
            parts.append("Everything observed so far, in order:")
            parts.extend(f"- {h}" for h in self.history)
            parts.append("")
        parts.append(_now_line(chunk))
        return "\n".join(parts)

    def append(self, description: str) -> None:
        self.history.append(description)

    def snapshot(self) -> dict:
        return {"mode": "append_only", "history_len": len(self.history),
                "summary_chars": 0, "recent": len(self.history)}


class RollingMemory(BaseMemory):
    """Bounded memory: anchor + summary + last k descriptions."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.summary: str = ""
        self.recent: deque[str] = deque(maxlen=max(cfg.memory_recent_k, 1))
        self._pending: list[str] = []     # evicted, not yet summarised
        self.compactions = 0

    def build_prompt(self, chunk) -> str:
        parts = [self.cfg.anchor_text, ""]
        if self.summary:
            parts += ["Summary of earlier events:", self.summary, ""]
        if self.recent:
            parts.append("Most recent observations:")
            parts += [f"- {r}" for r in self.recent]
            parts.append("")
        parts.append(_now_line(chunk))
        return "\n".join(parts)

    def append(self, description: str) -> None:
        if len(self.recent) == self.recent.maxlen and self.recent:
            self._pending.append(self.recent[0])   # about to be evicted
        self.recent.append(description)

    async def maybe_compact(self, provider) -> dict | None:
        """Fold evicted descriptions into the summary. One extra model call
        per memory_compact_every evictions -- the entire cost of bounding."""
        if len(self._pending) < self.cfg.memory_compact_every:
            return None

        batch, self._pending = self._pending, []
        instruction = _SUMMARY_INSTRUCTION.format(
            words=self.cfg.memory_summary_words)
        prompt = (
            f"{instruction}\n\n"
            f"Running summary:\n{self.summary or '(empty)'}\n\n"
            f"New events:\n" + "\n".join(f"- {b}" for b in batch)
        )
        try:
            result = await provider.describe(prompt, images=[])
            text = (result.text or "").strip()
            if text:
                self.summary = text
                self.compactions += 1
                log.info("memory compacted (#%d, %d chars)",
                         self.compactions, len(self.summary))
                return {
                    "compaction_number": self.compactions,
                    "events_folded": len(batch),
                    "summary_chars": len(self.summary),
                }
        except Exception:
            log.exception("compaction failed; keeping previous summary")
            # Put the batch back so nothing is silently lost.
            self._pending = batch + self._pending
        return None

    def snapshot(self) -> dict:
        return {
            "mode": "bounded",
            "summary_chars": len(self.summary),
            "recent": len(self.recent),
            "pending": len(self._pending),
            "compactions": self.compactions,
        }


def _now_line(chunk) -> str:
    bits = [f"This is the view at t={chunk.t_start:.1f}s."]
    if chunk.speech_ms:
        bits.append(f"About {chunk.speech_ms} ms of speech was heard.")
    if chunk.escalated:
        bits.append("A loud acoustic event occurred just before this moment.")
    bits.append("Describe in one or two sentences what is happening now. "
                "If nothing has changed, say so in four words.")
    return " ".join(bits)


def make_memory(cfg) -> BaseMemory:
    return RollingMemory(cfg) if cfg.memory_bounded else AppendOnlyMemory(cfg)
