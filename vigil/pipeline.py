"""The pipeline. All concurrency in VIGIL lives in this file.

Shape:

    capture threads  ->  thread queues  ->  feeder (async)  ->  bounded
    asyncio.Queue  ->  worker pool  ->  provider  ->  memory + CSV + UI

The bounded queue is Objective 2. When workers fall behind, the queue fills;
the feeder then stops draining the capture queues, which fill in turn, and
capture itself slows. Nothing grows without limit and nothing is silently
discarded without being counted.

Everything is observable through subscribe(): the web server attaches a
callback and forwards events to the browser over a WebSocket.
"""
from __future__ import annotations

import asyncio
import logging
import os
import queue as _queue
import time
from typing import Callable

from .chunker import Chunker, Escalator
from .config import Config
from .gate import AudioGate, GateStats, VideoGate
from .memory import make_memory
from .metrics import CsvLogger, RateLimiter, RunMetrics, write_manifest
from .providers import ProviderError, RateLimitError, make_provider
from .source import StreamSource

log = logging.getLogger("vigil.pipeline")

Event = dict
Subscriber = Callable[[Event], None]


class Pipeline:
    def __init__(self, url: str, cfg: Config | None = None,
                 with_audio: bool = True):
        self.cfg = cfg or Config.from_env()
        self.url = url
        self.with_audio = with_audio

        self.gate_stats = GateStats()
        self.video_gate = VideoGate(self.cfg, self.gate_stats)
        self.audio_gate = AudioGate(self.cfg, self.gate_stats)
        self.escalator = Escalator(self.cfg, self.video_gate, self.gate_stats)
        self.chunker = Chunker(self.cfg)
        self.memory = make_memory(self.cfg)
        self.metrics = RunMetrics(run_id=self.cfg.run_id)

        self.provider = make_provider(self.cfg)
        self.limiter = RateLimiter(self.cfg.rate_limit_calls,
                                   self.cfg.rate_limit_window_s)

        self.source = StreamSource(url, self.cfg, with_audio=with_audio)
        self.queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.cfg.queue_maxsize)

        self._subs: list[Subscriber] = []
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._stopping = False
        self._csv: CsvLogger | None = None
        self._tokens_cum = 0

        # Run-tracking state for collapsed observable events
        self._drop_run: dict | None = None
        self._speech_run: dict | None = None
        self._silence_run: dict | None = None

    # -- run flush helpers ------------------------------------------------
    def _flush_drop_run(self) -> None:
        if self._drop_run and self._drop_run["count"] > 0:
            if self.cfg.emit_verbose_events:
                self._emit({
                    "type": "gate_drop_run",
                    "t_start": round(self._drop_run["t_start"], 3),
                    "t_end": round(self._drop_run["t_end"], 3),
                    "count": self._drop_run["count"],
                    "ssim_min": round(self._drop_run["ssim_min"], 4),
                    "ssim_max": round(self._drop_run["ssim_max"], 4),
                    "cost": "free",
                })
            self._drop_run = None

    def _flush_speech_run(self) -> None:
        if self._speech_run and self._speech_run["count"] > 0:
            if self.cfg.emit_verbose_events:
                dur_ms = int(round(self._speech_run["count"] * self.cfg.vad_frame_ms))
                self._emit({
                    "type": "audio_speech",
                    "t_start": round(self._speech_run["t_start"], 3),
                    "t_end": round(self._speech_run["t_end"], 3),
                    "duration_ms": dur_ms,
                    "peak_rms": round(self._speech_run["peak_rms"], 4),
                    "cost": "free",
                })
            self._speech_run = None

    def _flush_silence_run(self) -> None:
        if self._silence_run and self._silence_run["count"] > 0:
            if self.cfg.emit_verbose_events:
                dur_ms = int(round(self._silence_run["count"] * self.cfg.vad_frame_ms))
                self._emit({
                    "type": "audio_silence",
                    "t_start": round(self._silence_run["t_start"], 3),
                    "t_end": round(self._silence_run["t_end"], 3),
                    "duration_ms": dur_ms,
                    "frames_dropped": self._silence_run["count"],
                    "cost": "free",
                })
            self._silence_run = None

    # -- observers --------------------------------------------------------
    def subscribe(self, fn: Subscriber) -> None:
        self._subs.append(fn)

    def unsubscribe(self, fn: Subscriber) -> None:
        if fn in self._subs:
            self._subs.remove(fn)

    def _emit(self, event: Event) -> None:
        if "cost" not in event:
            if event.get("type") in ("description", "memory_compacted"):
                event["cost"] = "billed"
            else:
                event["cost"] = "free"
        for fn in list(self._subs):
            try:
                fn(event)
            except Exception:
                log.exception("subscriber raised; dropping it")
                self.unsubscribe(fn)

    # -- status -----------------------------------------------------------
    def status(self) -> dict:
        return {
            "running": self._running,
            "url": self.url,
            "provider": self.provider.name,
            "tau": round(self.video_gate.tau, 4),
            "escalated": self.escalator.active,
            "queue_depth": self.queue.qsize(),
            "frames_captured": self.source.frames_captured,
            "reconnects": self.source.reconnects,
            "gate": self.gate_stats.snapshot(),
            "memory": self.memory.snapshot(),
            "metrics": self.metrics.snapshot(),
        }

    # -- lifecycle --------------------------------------------------------
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stopping = False

        csv_path = os.path.join(self.cfg.log_dir, f"{self.cfg.run_id}.csv")
        self._csv = CsvLogger(csv_path)
        log.info("run %s logging to %s", self.cfg.run_id, csv_path)

        self.source.start()
        self._tasks.append(asyncio.create_task(self._feeder(), name="feeder"))
        for i in range(self.cfg.max_concurrency):
            self._tasks.append(
                asyncio.create_task(self._worker(i), name=f"worker-{i}"))
        self._emit({"type": "started", "status": self.status()})

    async def stop(self) -> None:
        if not self._running:
            return
        self._stopping = True
        self.source.stop()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._flush_drop_run()
        self._flush_speech_run()
        self._flush_silence_run()
        try:
            await self.provider.aclose()
        except Exception:
            pass
        if self._csv:
            self._csv.close()
        write_manifest(
            os.path.join(self.cfg.log_dir, f"{self.cfg.run_id}.json"),
            self.cfg, self.metrics, self.gate_stats,
            extra={"url": self.url, "reconnects": self.source.reconnects},
        )
        self._running = False
        self._emit({"type": "stopped", "status": self.status()})
        log.info("run %s stopped: %s", self.cfg.run_id,
                 self.metrics.snapshot())

    async def run_until_complete(self, max_seconds: float | None = None):
        """Convenience entry point for experiments and CLI use."""
        await self.start()
        t0 = time.monotonic()
        try:
            while self._running:
                await asyncio.sleep(0.5)
                if max_seconds and (time.monotonic() - t0) >= max_seconds:
                    log.info("time limit reached")
                    break
                if self.source.ended.is_set() and self.queue.empty():
                    # Let in-flight work finish before tearing down.
                    await asyncio.sleep(1.0)
                    if self.queue.empty():
                        log.info("source ended and queue drained")
                        break
        finally:
            await self.stop()
        return self.metrics.snapshot()

    # -- stage 1: gate + chunk -------------------------------------------
    async def _feeder(self) -> None:
        """Drains capture queues, applies the gate, emits chunks.

        Runs in the event loop but touches the thread queues non-blockingly,
        so a stalled source never blocks the workers.
        """
        try:
            while not self._stopping:
                did_work = False

                # --- audio first: escalation must apply to the frames that
                # follow the sound, not the ones before it.
                for _ in range(32):
                    try:
                        a = self.source.audio_q.get_nowait()
                    except _queue.Empty:
                        break
                    did_work = True
                    speech, rms = self.audio_gate.is_speech(a.pcm)
                    fired = self.escalator.consider(speech, rms)
                    if speech or rms > 0:
                        self.chunker.add_audio(speech, rms, a.t,
                                               escalated=fired)
                    if fired:
                        self._emit({"type": "escalation", "t": a.t,
                                    "rms": round(rms, 4),
                                    "tau": self.video_gate.tau,
                                    "cost": "free"})

                    if speech:
                        self._flush_silence_run()
                        if self._speech_run is None:
                            self._speech_run = {
                                "t_start": a.t,
                                "t_end": a.t,
                                "count": 1,
                                "peak_rms": rms,
                            }
                        else:
                            self._speech_run["t_end"] = a.t
                            self._speech_run["count"] += 1
                            self._speech_run["peak_rms"] = max(self._speech_run["peak_rms"], rms)
                            if self._speech_run["t_end"] - self._speech_run["t_start"] >= 30.0:
                                self._flush_speech_run()
                    else:
                        self._flush_speech_run()
                        if self._silence_run is None:
                            self._silence_run = {
                                "t_start": a.t,
                                "t_end": a.t,
                                "count": 1,
                            }
                        else:
                            self._silence_run["t_end"] = a.t
                            self._silence_run["count"] += 1
                            if self._silence_run["t_end"] - self._silence_run["t_start"] >= 30.0:
                                self._flush_silence_run()

                self.escalator.tick()

                # --- video
                for _ in range(16):
                    try:
                        v = self.source.video_q.get_nowait()
                    except _queue.Empty:
                        break
                    did_work = True
                    self.chunker.note_frame_seen()
                    keep, score = self.video_gate.should_keep(v.frame)
                    if self.cfg.emit_verbose_events:
                        self._emit({
                            "type": "frame", "t": round(v.t, 2),
                            "kept": keep, "ssim": round(score, 4),
                            "gate": self.gate_stats.snapshot(),
                            "cost": "free",
                        })
                    if keep:
                        self._flush_drop_run()
                        reason = "escalated" if self.escalator.active else "change"
                        if self.cfg.emit_verbose_events:
                            self._emit({
                                "type": "frame_kept",
                                "t": round(v.t, 3),
                                "ssim": round(score, 4),
                                "tau": round(self.video_gate.tau, 4),
                                "reason": reason,
                                "cost": "free",
                            })
                        chunk = self.chunker.add_frame(v.frame, v.t)
                        if chunk:
                            await self._enqueue(chunk)
                    else:
                        if self._drop_run is None:
                            self._drop_run = {
                                "t_start": v.t,
                                "t_end": v.t,
                                "count": 1,
                                "ssim_min": score,
                                "ssim_max": score,
                            }
                        else:
                            self._drop_run["t_end"] = v.t
                            self._drop_run["count"] += 1
                            self._drop_run["ssim_min"] = min(self._drop_run["ssim_min"], score)
                            self._drop_run["ssim_max"] = max(self._drop_run["ssim_max"], score)
                            if self._drop_run["t_end"] - self._drop_run["t_start"] >= 30.0:
                                self._flush_drop_run()

                # --- time-based flush
                now = self.source.elapsed
                if self.chunker.due(now):
                    chunk = self.chunker.flush(now)
                    if chunk:
                        await self._enqueue(chunk)

                if not did_work:
                    await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("feeder crashed")
        finally:
            self._flush_drop_run()
            self._flush_speech_run()
            self._flush_silence_run()

    async def _enqueue(self, chunk) -> None:
        """Blocking put. This await IS the backpressure: while the queue is
        full the feeder stops draining capture, and capture slows in turn."""
        t0 = time.monotonic()
        await self.queue.put(chunk)
        waited = time.monotonic() - t0
        if self.cfg.emit_verbose_events:
            self._emit({
                "type": "chunk_queued",
                "chunk": chunk.index,
                "t": round(chunk.t_start, 3),
                "images": len(chunk.frames_b64),
                "speech_ms": chunk.speech_ms,
                "queue_depth": self.queue.qsize(),
                "cost": "free",
            })
        if waited > 0.5:
            log.info("backpressure: feeder waited %.1fs for queue space",
                     waited)
            self._emit({"type": "backpressure", "waited_s": round(waited, 2),
                        "queue_depth": self.queue.qsize(), "cost": "free"})

    # -- stage 2: dispatch ------------------------------------------------
    async def _worker(self, wid: int) -> None:
        try:
            while not self._stopping:
                chunk = await self.queue.get()
                try:
                    await self._process(chunk)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("worker %d failed on chunk %d",
                                  wid, chunk.index)
                    self.metrics.errors += 1
                finally:
                    self.queue.task_done()
        except asyncio.CancelledError:
            raise

    async def _process(self, chunk) -> None:
        prompt = self.memory.build_prompt(chunk)
        images = chunk.frames_b64
        depth = self.queue.qsize()

        completion = None
        retries = 0
        last_error = None

        for attempt in range(self.cfg.max_retries):
            waited = await self.limiter.acquire()
            if waited > 0 and self.cfg.emit_verbose_events:
                self._emit({
                    "type": "rate_limit_wait",
                    "waited_s": round(waited, 3),
                    "cost": "free",
                })
            try:
                completion = await self.provider.describe(prompt, images)
                break
            except RateLimitError as e:
                self.metrics.rate_limit_faults += 1
                last_error = str(e)
                retries += 1
                backoff = min(2 ** attempt, 30) + 0.5
                log.warning("rate limited (%s); backing off %.1fs",
                            e, backoff)
                await asyncio.sleep(backoff)
            except ProviderError as e:
                last_error = str(e)
                retries += 1
                await asyncio.sleep(min(2 ** attempt, 10))

        if completion is None:
            self.metrics.errors += 1
            self.metrics.retries += retries
            self._log_row(chunk, None, depth, retries, last_error)
            self._emit({"type": "error", "chunk": chunk.index,
                        "error": last_error, "cost": "free"})
            return

        description = completion.text or "(no description returned)"
        self.memory.append(description)
        compact_res = await self.memory.maybe_compact(self.provider)
        if compact_res:
            self._emit({
                "type": "memory_compacted",
                "compaction_number": compact_res["compaction_number"],
                "events_folded": compact_res["events_folded"],
                "summary_chars": compact_res["summary_chars"],
                "cost": "billed",
            })

        self.metrics.retries += retries
        self.metrics.record(completion.tokens_in, completion.tokens_out,
                            completion.latency_s, depth, chunk.t_start)
        self._tokens_cum += completion.tokens_in + completion.tokens_out

        self._log_row(chunk, completion, depth, retries, None)
        self._emit({
            "type": "description",
            "chunk": chunk.index,
            "t": round(chunk.t_start, 2),
            "text": description,
            "escalated": chunk.escalated,
            "tokens": completion.tokens_in + completion.tokens_out,
            "tokens_cum": self._tokens_cum,
            "latency_s": round(completion.latency_s, 3),
            "status": self.status(),
            "cost": "billed",
        })

    def _log_row(self, chunk, completion, depth, retries, error) -> None:
        if not self._csv:
            return
        self._csv.write({
            "chunk_index": chunk.index,
            "wall_t": round(time.time(), 3),
            "stream_t": round(chunk.t_start, 3),
            "frames_seen": chunk.frames_seen,
            "frames_kept": chunk.frames_kept,
            "frame_retention_cum": round(self.gate_stats.frame_retention, 4),
            "speech_ms": chunk.speech_ms,
            "peak_rms": round(chunk.peak_rms, 4),
            "escalated": int(chunk.escalated),
            "tau": round(self.video_gate.tau, 4),
            "images_sent": len(chunk.frames_b64),
            "payload_bytes": chunk.payload_bytes,
            "tokens_in": completion.tokens_in if completion else 0,
            "tokens_out": completion.tokens_out if completion else 0,
            "tokens_cum": self._tokens_cum,
            "latency_s": round(completion.latency_s, 3) if completion else 0,
            "queue_depth": depth,
            "retries": retries,
            "rate_limit_faults": self.metrics.rate_limit_faults,
            "error": error or "",
            "description": (completion.text if completion else "")[:500],
        })
