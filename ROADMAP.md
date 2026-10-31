# Roadmap — what is done, what is not, and how to continue

## State of the code

**Working and tested (24 offline tests pass):**

- Capture from YouTube/HLS/RTSP/file/webcam, with manifest re-resolution
- Real-time-paced file replay, so runs are reproducible and comparable
- SSIM video gate and WebRTC VAD audio gate, with full counters
- Chunking, JPEG encoding, cross-modal escalation
- Bounded memory with recursive summarisation, and the append-only baseline
- Three providers behind one interface; retries and 429 backoff
- Bounded queue, worker pool, sliding-window rate limiter, real backpressure
- Per-chunk CSV logging and per-run JSON manifests
- FastAPI server, WebSocket, working dashboard
- Experiment runner (A–D and τ sweep) and analysis producing three figures

**Verified end to end** on a synthetic 60-second clip: retention falls from
100% to 30%, tokens fall roughly 4×, and the τ sweep shows recall collapsing
at τ=0.80 while holding at τ≥0.92. The mechanism works. The numbers are from
a toy clip and mean nothing scientifically.

---

## Not done — in priority order

### 1. Record and annotate real footage ← start here, it gates everything

No code required and nothing else can be reported without it. Six hours of a
real fixed scene, plus `annotations.csv` with `start_s,end_s,label`. Have
someone who did not record it do the annotating. Put `audio` in the label of
acoustically-cued events.

### 2. Semantic recall scoring

`compute_recall()` currently asks only whether *some* description landed in
the event's time window. It does not check whether the description is
*correct*. Needs either an LLM judge or manual scoring on a sample.

> **Antigravity prompt:** In `experiments/analyze.py`, replace
> `compute_recall` with a two-tier scorer. Tier 1 keeps the existing temporal
> match. Tier 2 sends the description and the annotation label to the
> provider and asks for a yes/no match, caching results to
> `logs/judge_cache.json` so reruns are free. Report both numbers. Add a
> `--no-judge` flag to skip tier 2. Write a test using MockProvider.

### 3. Second annotator agreement

The methodology promises inter-annotator agreement on a 10% sample. Nothing
computes it.

> **Antigravity prompt:** Add `experiments/agreement.py` that takes two
> annotation CSVs, matches events by temporal overlap above 0.5 IoU, and
> reports Cohen's kappa plus a confusion count. Print a table of
> disagreements so they can be resolved by hand.

### 4. Escalation needs a non-trivial trigger

Right now escalation fires on RMS above a threshold, with VAD confirming
voice. That misses glass breaking and alarms, which are not speech.

> **Antigravity prompt:** In `vigil/chunker.py`, extend `Escalator` with a
> second trigger: spectral flux computed on the PCM frame using numpy's FFT,
> firing when flux exceeds a configurable threshold, independent of VAD. Add
> `escalation_flux_threshold` to Config. Keep the RMS trigger. Log which
> trigger fired in the CSV. Add tests for a tone burst, white-noise burst and
> silence.

### 5. Cost in rupees, not just tokens

Token counts do not persuade anyone outside the room. Price per million
tokens turns them into a bill.

> **Antigravity prompt:** Add `vigil/pricing.py` with a table of per-million
> input and output token prices per model, loaded from `pricing.json` so it
> can be updated without code changes. Add a `cost_inr` column to the CSV and
> a cost row to the results table. Default to USD with a configurable
> conversion rate.

### 6. Reconnect test against a real expiring manifest

The reconnect path is written but only exercised against a file ending. It
must be proven on a live stream over several hours.

> **Antigravity prompt:** Add `tests/test_reconnect.py` using a fake source
> that fails reads after N frames and succeeds after re-resolution. Assert
> `reconnects` increments, that capture resumes, and that stream timestamps
> stay monotonic across the reconnect.

### 7. Dashboard: live SSIM plot and kept-frame thumbnails

Seeing the gate decide is what makes the demo land in a viva.

> **Antigravity prompt:** In `static/index.html`, add a rolling sparkline of
> the last 120 SSIM scores with the τ threshold drawn as a horizontal line,
> and a strip of the last six kept frames as thumbnails. The pipeline already
> emits `frame` events with `ssim`; add the JPEG to that event behind a
> `cfg.emit_thumbnails` flag so it stays off during measurement runs.

### 8. Graceful degradation when the provider dies

Currently a failed chunk is logged and dropped. A long run should survive a
provider outage and say so.

> **Antigravity prompt:** Add a circuit breaker to `vigil/pipeline.py`: after
> K consecutive provider failures, stop dispatching for a cool-off period,
> emit a `degraded` event, and keep gating and logging so the run continues
> with a gap rather than ending. Add config for K and the cool-off. Test with
> a provider stub that always raises.

### 9. Multi-camera

Out of scope for the paper, mentioned in future work. One `Pipeline` per URL
with a shared rate limiter is the natural shape.

---

## How to drive Antigravity on this repo

Give it `README.md` and this file as context first. Then one task per prompt,
each with an acceptance test, and **run the test yourself before moving on**.
An agent will tell you something works when it does not.

A good prompt names the file, the behaviour, and how you will check it:

> In `vigil/gate.py`, add a `MotionGate` that uses frame differencing instead
> of SSIM, selected by `cfg.gate_method`. Keep `VideoGate`'s interface so
> `pipeline.py` is unchanged. Add tests comparing retention of both methods
> on the same synthetic frames.

A bad prompt is "improve the gate."

### Things that will break if you are careless

- `cv2.VideoCapture` blocks. It must stay in a thread. Never call it from the
  event loop.
- ffmpeg's stdout must be drained continuously or ffmpeg freezes on a full
  pipe.
- `webrtcvad` accepts only 10, 20 or 30 ms frames at 8/16/32/48 kHz. Any
  other length raises.
- Frame size handed to `cv2.VideoWriter` must match its constructor exactly,
  or every write silently fails and you get an empty file.
- `await queue.put()` in `_enqueue` is load-bearing. Replacing it with
  `put_nowait` removes backpressure and the project loses Objective 2.

---

## Writing the paper

Target a workshop or IEEE conference, not a main-track vision venue. The
contribution is measurement and systems integration; claiming more invites a
desk reject.

Three contributions, stated plainly:

1. A characterisation of the recall-versus-cost tradeoff for zero-compute
   edge gating under black-box multimodal APIs.
2. A cross-modal escalation rule recovering recall on acoustically-cued
   events at bounded additional token cost.
3. An open pipeline reporting cost in provider-billable units rather than GPU
   memory.

Three figures carry it: the τ sweep curve, the cumulative-tokens plot (A
climbing, C flat — make this one first), and the C-vs-D escalation bar.

**Before claiming novelty**, search Google Scholar and arXiv yourself for
"training-free frame selection streaming", "adaptive sampling video LLM",
"audio-triggered video sampling". A novelty claim is the easiest thing for a
reviewer to destroy.

**Write the limitations section honestly**: one scene type, one provider, one
annotator pair, and SSIM reacting to lighting shifts and autofocus as if they
were events. Naming your own weaknesses is what separates an accepted paper
from a dismantled one.
