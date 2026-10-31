"""Capture stage.

Reads an unbounded video+audio source and pushes timestamped items into two
thread-safe queues. Video runs in one daemon thread, audio in an ffmpeg
subprocess read by a second thread. Both stamp items from a single
time.monotonic() origin so the chunker can align them.

Supported inputs:
  - YouTube live / VOD URL (resolved through yt-dlp to an HLS manifest)
  - direct RTSP / HTTP / .m3u8 URL
  - local file path (replayed at real-time rate for reproducible runs)
  - integer webcam index as a string, e.g. "0"

Live HLS manifests expire after a few hours. resolve() is therefore called
again whenever reads start failing; this is normal operation, not an error
path, and is the reason an unbounded run survives overnight.
"""
from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("vigil.source")


@dataclass
class VideoItem:
    t: float            # seconds since run start
    frame: np.ndarray   # BGR uint8


@dataclass
class AudioItem:
    t: float            # seconds since run start
    pcm: bytes          # 16-bit mono little-endian at config.audio_sample_rate


class SourceError(RuntimeError):
    pass


def resolve_url(url: str, target_height: int = 480) -> str:
    """Turn a page URL into something OpenCV/ffmpeg can open.

    Local paths, webcam indices and direct media URLs pass straight through.
    Anything that looks like a site page goes to yt-dlp.
    """
    if os.path.exists(url):
        return url
    if url.isdigit():
        return url
    low = url.lower()
    direct = (".m3u8", ".mp4", ".mkv", ".ts", ".flv", ".webm")
    if low.startswith("rtsp://") or low.endswith(direct):
        return url

    try:
        import yt_dlp
    except ImportError as e:  # pragma: no cover
        raise SourceError("yt-dlp is required to resolve page URLs") from e

    opts = {
        "quiet": True,
        "no_warnings": True,
        "format": f"best[height<={target_height}]/best",
        "noplaylist": True,
    }

    info = None
    # First try: normal extraction with format filtering.
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception:
            pass

    if info is None:
        # Second try: skip format selection entirely and just extract
        # metadata.  HLS-only live streams (e.g. YouTube) have formats
        # whose protocol is m3u8_native; yt-dlp's format selector
        # sometimes rejects them even with "best".
        log.warning("format filter failed; extracting without processing")
        opts.pop("format", None)
        with yt_dlp.YoutubeDL(opts) as ydl2:
            info = ydl2.extract_info(url, download=False, process=False)

    if info.get("url"):
        return info["url"]

    # Walk formats and pick the best one whose height is at most
    # target_height, falling back to any playable URL.
    best_url = None
    for f in info.get("formats", []):
        furl = f.get("url")
        if not furl:
            continue
        h = f.get("height") or 0
        if h and h <= target_height:
            best_url = furl          # keep scanning for a higher-res match
        elif best_url is None:
            best_url = furl          # fallback: anything playable
    if best_url:
        return best_url
    raise SourceError(f"yt-dlp returned no playable URL for {url}")


def find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class StreamSource:
    """Threaded capture with automatic re-resolution of expired manifests."""

    def __init__(self, url: str, cfg, with_audio: bool = True):
        self.url = url
        self.cfg = cfg
        self._ffmpeg_bin = find_ffmpeg()
        self.with_audio = with_audio and self._ffmpeg_bin is not None
        if with_audio and not self.with_audio:
            log.warning("ffmpeg not found on PATH; running video-only")

        self.video_q: "queue.Queue[VideoItem]" = queue.Queue(maxsize=120)
        self.audio_q: "queue.Queue[AudioItem]" = queue.Queue(maxsize=400)

        self._stop = threading.Event()
        self._t0: Optional[float] = None
        self._threads: list[threading.Thread] = []
        self._ff: Optional[subprocess.Popen] = None
        self._resolved: Optional[str] = None
        self.is_file = os.path.exists(url)

        self.frames_captured = 0
        self.reconnects = 0
        self._next_stream_sample = 0.0
        self.ended = threading.Event()   # set when a finite file runs out
        self.latest_frame: Optional[np.ndarray] = None

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        self._t0 = time.monotonic()
        self._resolved = resolve_url(self.url, self.cfg.target_height)
        log.info("resolved source (%s)", "file" if self.is_file else "stream")

        tv = threading.Thread(target=self._video_loop, name="vigil-video",
                              daemon=True)
        tv.start()
        self._threads.append(tv)

        if self.with_audio:
            ta = threading.Thread(target=self._audio_loop, name="vigil-audio",
                                  daemon=True)
            ta.start()
            self._threads.append(ta)

    def stop(self) -> None:
        self._stop.set()
        if self._ff and self._ff.poll() is None:
            try:
                self._ff.terminate()
                self._ff.wait(timeout=3)
            except Exception:
                try:
                    self._ff.kill()
                except Exception:
                    pass
        for t in self._threads:
            t.join(timeout=3)

    @property
    def elapsed(self) -> float:
        return 0.0 if self._t0 is None else time.monotonic() - self._t0

    # -- video ------------------------------------------------------------
    def _open_capture(self) -> cv2.VideoCapture:
        target = int(self._resolved) if self._resolved.isdigit() else self._resolved
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            raise SourceError("OpenCV could not open the resolved source")
        return cap

    def _video_loop(self) -> None:
        interval = 1.0 / max(self.cfg.capture_fps, 0.01)
        attempts = 0
        cap = None
        try:
            cap = self._open_capture()
            next_sample = time.monotonic()
            consecutive_fail = 0

            # For a finite file we pace replay by the file's own frame rate
            # rather than reading as fast as the disk allows. Without this a
            # 6-hour clip finishes in minutes, every run samples a different
            # set of frames, and timings cannot be compared between runs.
            src_fps = 0.0
            frame_idx = 0
            if self.is_file:
                try:
                    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
                except Exception:
                    src_fps = 0.0
                if not (src_fps > 0):
                    src_fps = 25.0
                    log.warning("file reports no FPS; assuming %.1f", src_fps)
                log.info("replaying file at %.2fx (source %.1f fps)",
                         self.cfg.replay_speed, src_fps)

            while not self._stop.is_set():
                ok, frame = cap.read()
                if ok and frame is not None:
                    self.latest_frame = frame

                if not ok:
                    consecutive_fail += 1
                    # A finite file simply ended.
                    if self.is_file:
                        log.info("input file exhausted")
                        self.ended.set()
                        return
                    if consecutive_fail < 3:
                        time.sleep(0.2)
                        continue
                    # Live manifests expire; re-resolve and reopen.
                    attempts += 1
                    if attempts > self.cfg.reconnect_max_attempts:
                        log.error("giving up after %d reconnect attempts",
                                  attempts)
                        self.ended.set()
                        return
                    backoff = self.cfg.reconnect_backoff_s * attempts
                    log.warning("stream read failed; re-resolving in %.1fs "
                                "(attempt %d)", backoff, attempts)
                    time.sleep(backoff)
                    try:
                        cap.release()
                    except Exception:
                        pass
                    try:
                        self._resolved = resolve_url(self.url,
                                                     self.cfg.target_height)
                        cap = self._open_capture()
                        self.reconnects += 1
                        consecutive_fail = 0
                        log.info("reconnected (total %d)", self.reconnects)
                    except Exception as e:
                        log.warning("reconnect failed: %s", e)
                    continue

                consecutive_fail = 0
                attempts = 0

                if self.is_file:
                    # Stream time comes from the file, not the clock.
                    frame_idx += 1
                    stream_t = frame_idx / src_fps
                    target_wall = (self._t0
                                   + stream_t / max(self.cfg.replay_speed, 0.01))
                    delay = target_wall - time.monotonic()
                    if delay > 0:
                        # Sleep in slices so stop() stays responsive.
                        end = time.monotonic() + delay
                        while not self._stop.is_set():
                            remaining = end - time.monotonic()
                            if remaining <= 0:
                                break
                            time.sleep(min(0.05, remaining))
                    if stream_t < self._next_stream_sample:
                        continue
                    self._next_stream_sample = stream_t + interval
                    t_item = stream_t
                else:
                    now = time.monotonic()
                    if now < next_sample:
                        continue
                    next_sample = now + interval
                    t_item = now - self._t0

                item = VideoItem(t=t_item, frame=frame)
                self.frames_captured += 1
                try:
                    self.video_q.put(item, timeout=1.0)
                except queue.Full:
                    # Consumer is behind. Dropping here is the capture-side
                    # face of backpressure; it is counted, never silent.
                    log.debug("video queue full; frame dropped at capture")
        except Exception:
            log.exception("video loop crashed")
            self.ended.set()
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    # -- audio ------------------------------------------------------------
    def _audio_loop(self) -> None:
        sr = self.cfg.audio_sample_rate
        bytes_per_chunk = int(sr * 2 * (self.cfg.vad_frame_ms / 1000.0))
        cmd = [
            self._ffmpeg_bin or "ffmpeg", "-nostdin", "-loglevel", "error",
            "-i", self._resolved,
            "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", str(sr), "-ac", "1", "-",
        ]
        if self.is_file:
            speed = max(self.cfg.replay_speed, 0.01)
            cmd.insert(1, str(speed))
            cmd.insert(1, "-readrate")

        try:
            self._ff = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=bytes_per_chunk * 4,
            )
        except FileNotFoundError:
            log.warning("ffmpeg unavailable; no audio")
            return

        audio_frame_idx = 0
        frame_dur_s = self.cfg.vad_frame_ms / 1000.0
        try:
            while not self._stop.is_set():
                # The pipe MUST be drained continuously or ffmpeg blocks.
                buf = self._ff.stdout.read(bytes_per_chunk)
                if not buf:
                    if self.is_file:
                        return
                    time.sleep(0.2)
                    if self._ff.poll() is not None:
                        return
                    continue
                if len(buf) < bytes_per_chunk:
                    buf = buf + b"\x00" * (bytes_per_chunk - len(buf))
                if self.is_file:
                    audio_frame_idx += 1
                    t_item = audio_frame_idx * frame_dur_s
                else:
                    t_item = time.monotonic() - self._t0
                item = AudioItem(t=t_item, pcm=buf)
                try:
                    self.audio_q.put(item, timeout=0.5)
                except queue.Full:
                    pass
        except Exception:
            log.exception("audio loop crashed")
