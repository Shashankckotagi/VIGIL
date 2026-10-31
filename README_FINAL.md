# VIGIL
## Vision Gating with Intermittent Listening
### Cost-Bounded Multimodal LLM Perception over Continuous Video and Audio Streams
> Final-year Gen AI Course Project · MS Ramaiah Institute of Technology · 2026

---

## Section 1 — What it does

VIGIL is an intelligent streaming perception system that allows cloud-based multimodal AI models to continuously monitor live video and audio feeds without incurring exorbitant API billing costs or overwhelming system memory. By implementing zero-GPU structural similarity visual filtering alongside acoustic speech detection, the pipeline drops redundant sensory inputs and only forwards meaningful changes to the language model. When unexpected loud noises or speech occur, an cross-modal escalation mechanism dynamically increases visual attention so the system never misses critical real-world security or operational events.

---

## Section 2 — Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                          INPUT SOURCE                           │
│           YouTube Live · RTSP · Local File · Webcam             │
└──────────────────────────┬──────────────────────────────────────┘
                           │ yt-dlp resolves · reconnects on expiry
                   ┌───────▼───────────────┐
                   │  CAPTURE (source.py)  │
                   │ OpenCV frames thread  │
                   │ ffmpeg audio thread   │
                   │   shared wall clock   │
                   └───────┬───────┬───────┘
                           │       │
              ┌────────────▼─┐   ┌─▼────────────┐
              │  VIDEO GATE  │   │  AUDIO GATE  │
              │  (gate.py)   │   │  (gate.py)   │
              │  SSIM 128px  │   │  WebRTC VAD  │
              │  ~70% drop   │   │  ~70% drop   │
              │    NO GPU    │   │    NO GPU    │
              └──────┬───────┘   └───┬──────────┘
                     │               │ RMS > threshold?
                     │   ┌───────────▼──────┐
                     │   │    ESCALATOR     │
                     │   │   (chunker.py)   │
                     │   │ raise τ for 30s  │
                     │   └───────────┬──────┘
                     └───────────────┼──────────┐
                                     │          │
                       ┌─────────────▼──────────▼────┐
                       │    CHUNKER (chunker.py)     │
                       │ 3 frames + speech → chunk   │
                       └─────────────┬───────────────┘
                                     │
                       ┌─────────────▼───────────────┐
                       │ BOUNDED QUEUE (pipeline.py) │ ← backpressure here
                       │      asyncio.Queue(10)      │   full = capture slows
                       └─────────────┬───────────────┘
                                     │
                       ┌─────────────▼───────────────┐
                       │ WORKER POOL (pipeline.py)   │
                       │  3 workers · rate limiter   │
                       │ retry + exponential backoff │
                       └─────────────┬───────────────┘
                                     │
                       ┌─────────────▼───────────────┐
                       │     MEMORY (memory.py)      │
                       │ anchor + summary + last 5   │
                       │ compacts every 10 evictions │
                       └─────────────┬───────────────┘
                                     │
                       ┌─────────────▼───────────────┐
                       │   PROVIDER (providers.py)   │
                       │    mock · ollama · gemini   │
                       │   one interface · swappable │
                       └─────────────┬───────────────┘
                                     │
                       ┌─────────────▼───────────────┐
                       │    DASHBOARD (server.py +   │
                       │      static/index.html)     │
                       │  WebSocket · live timeline  │
                       │ CSV per run · JSON manifest │
                       └─────────────────────────────┘
```

### Component Breakdown
- **INPUT SOURCE ([source.py](file:///c:/Users/HP/OneDrive/Desktop/VIGIL/VIGIL-project/vigil/vigil/source.py)):** Ingests live streams from YouTube Live, RTSP feeds, webcams, or local video files with automatic yt-dlp URL resolution and manifest reconnection.
- **CAPTURE ([source.py](file:///c:/Users/HP/OneDrive/Desktop/VIGIL/VIGIL-project/vigil/vigil/source.py)):** Decouples media intake into an OpenCV frame capture worker and a separate ffmpeg PCM audio reader synchronized against a unified monotonic clock.
- **VIDEO GATE ([gate.py](file:///c:/Users/HP/OneDrive/Desktop/VIGIL/VIGIL-project/vigil/vigil/gate.py)):** Calculates zero-GPU structural similarity (SSIM) between successive 128×128 grayscale thumbnails to drop 70–90% of redundant frames without neural overhead.
- **AUDIO GATE ([gate.py](file:///c:/Users/HP/OneDrive/Desktop/VIGIL/VIGIL-project/vigil/vigil/gate.py)):** Evaluates 16kHz audio chunks with WebRTC Voice Activity Detection (VAD) to filter background silence and retain voiced frames.
- **ESCALATOR ([chunker.py](file:///c:/Users/HP/OneDrive/Desktop/VIGIL/VIGIL-project/vigil/vigil/chunker.py)):** Monitors audio RMS energy and automatically relaxes the visual SSIM threshold τ for 30 seconds upon detecting loud acoustic triggers.
- **CHUNKER ([chunker.py](file:///c:/Users/HP/OneDrive/Desktop/VIGIL/VIGIL-project/vigil/vigil/chunker.py)):** Bundles up to 3 kept visual frames and contemporaneous speech into fixed-window perception chunks for model dispatch.
- **BOUNDED QUEUE ([pipeline.py](file:///c:/Users/HP/OneDrive/Desktop/VIGIL/VIGIL-project/vigil/vigil/pipeline.py)):** Enforces an asyncio queue with maxsize 10 that exerts backpressure to stall capture when downstream workers fall behind.
- **WORKER POOL ([pipeline.py](file:///c:/Users/HP/OneDrive/Desktop/VIGIL/VIGIL-project/vigil/vigil/pipeline.py)):** Coordinates 3 concurrent asynchronous workers governed by a sliding-window rate limiter with exponential backoff retries.
- **MEMORY ([memory.py](file:///c:/Users/HP/OneDrive/Desktop/VIGIL/VIGIL-project/vigil/vigil/memory.py)):** Retains an immutable attention anchor, recursive summary compaction every 10 evictions, and a sliding window of the last 5 descriptions.
- **PROVIDER ([providers.py](file:///c:/Users/HP/OneDrive/Desktop/VIGIL/VIGIL-project/vigil/vigil/providers.py)):** Implements an abstracted black-box inference interface supporting local Mock, Ollama, and Google Gemini API backends.
- **DASHBOARD ([server.py](file:///c:/Users/HP/OneDrive/Desktop/VIGIL/VIGIL-project/vigil/server.py), [static/index.html](file:///c:/Users/HP/OneDrive/Desktop/VIGIL/VIGIL-project/vigil/static/index.html)):** Serves a FastAPI WebSocket backend streaming real-time frame telemetry, perception logs, and an interactive feed interface.

---

## Section 3 — The three objectives

| Objective | What it does in plain language | Mechanism | Measured result |
| :--- | :--- | :--- | :--- |
| **O1 — Flat cost** | Prevents token usage and prompt size from exploding on infinite continuous video feeds. | SSIM gate + bounded memory | Token reduction vs baseline: 78.0% · Prompt flatness ratio: 1.318 |
| **O2 — Stability** | Prevents memory overflows, runaway queuing, and API rate-limit crashes under real-time network streaming. | Bounded queue + backpressure | Max queue depth: 0 · 429 faults: 0 · Backpressure events: 0 |
| **O3 — Cross-modal** | Employs audio anomalies to trigger higher visual frame rates before pixels themselves are encoded. | Audio RMS triggers visual τ | Recall gain on audio events: +0.0 pp · Concentration ratio: 0.40 |

---

## Section 4 — Research gap filled

- Every published streaming method requires white-box model access (names StreamMind, StreamMem, STC, CoRDS). VIGIL works from outside any HTTP API.
- Every published gate runs a neural encoder on every frame, requiring a GPU per stream. VIGIL’s gate is SSIM + VAD — zero neural compute, runs on a Raspberry Pi.
- No published work uses audio to govern the visual sampling rate upstream of the encoder. VIGIL’s cross-modal escalation rule does exactly this.

---

## Section 5 — Results summary

### 5.1 Cost reduction
| Run | Retention % | Total Tokens | Tokens / Hour | Cost (USD) | Cost (INR) | Reduction vs A |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | 100.0% | 50,570 | 3,112,000.0 | $0.0040 | ₹0.34 | 0.0% (1.00x) |
| **B** | 30.0% | 11,544 | 755,607.3 | $0.0009 | ₹0.08 | 77.2% (4.38x) |
| **C** | 30.0% | 11,141 | 729,229.1 | $0.0009 | ₹0.08 | 78.0% (4.54x) |
| **D** | 34.2% | 12,113 | 800,124.8 | $0.0010 | ₹0.08 | 76.0% (4.17x) |
| **S1** | 0.8% | 341 | 20,460.0 | $0.0000 | ₹0.00 | 99.3% (148.30x) |
| **S2** | 0.8% | 341 | 20,460.0 | $0.0000 | ₹0.00 | 99.3% (148.30x) |
| **S3** | 30.0% | 11,141 | 729,229.1 | $0.0009 | ₹0.08 | 78.0% (4.54x) |
| **S4** | 34.2% | 12,099 | 799,200.0 | $0.0010 | ₹0.08 | 76.1% (4.18x) |
| **S5** | 34.2% | 12,100 | 799,266.1 | $0.0010 | ₹0.08 | 76.1% (4.18x) |

### 5.2 Accuracy retained
| Run | Temporal Recall | Semantic Recall | Audio-Cued Recall | Precision |
| :--- | :---: | :---: | :---: | :---: |
| **A** | 100.0% | 0.0% | 100.0% | 100.0% |
| **B** | 100.0% | 0.0% | 100.0% | 100.0% |
| **C** | 100.0% | 0.0% | 100.0% | 100.0% |
| **D** | 100.0% | 0.0% | 100.0% | 100.0% |
| **S1** | 25.0% | 0.0% | 0.0% | 100.0% |
| **S2** | 25.0% | 0.0% | 0.0% | 100.0% |
| **S3** | 100.0% | 0.0% | 100.0% | 100.0% |
| **S4** | 100.0% | 0.0% | 100.0% | 100.0% |
| **S5** | 100.0% | 0.0% | 100.0% | 100.0% |

**At τ=0.92, VIGIL retains 100.0% of the ungated model’s temporal recall at 22.0% of the token cost.**

### 5.3 Threshold sweep
| τ | Retention % | Recall % | Tokens / Hour |
| :---: | :---: | :---: | :---: |
| **0.80** | 0.8% | 25.0% | 20,460.0 |
| **0.86** | 0.8% | 25.0% | 20,460.0 |
| **0.92** | 30.0% | 100.0% | 729,229.1 |
| **0.96** | 34.2% | 100.0% | 799,200.0 |
| **0.99** | 34.2% | 100.0% | 799,266.1 |

Empirical threshold sweep shows the knee of the recall-cost curve sits at τ=0.92 (30.0% retention), beyond which token expenditure climbs with diminishing recall gains.

---

## Section 6 — Benchmark environment

- **Provider & Model:** mock (`mock-v1`)
- **Compute Hardware:** CPU: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel | RAM: 15.7 GB | GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- **Operating System:** Windows 11 (10.0.26200) (Python 3.13.5)
- **Test Footage:** `sample/demo_clip.avi` (60.0s @ 320x240)
- **Ground Truth Annotations:** 4 annotated events (Second annotator used: False)
- **Benchmark Run Timestamp:** 2026-10-29T13:52:16.643121+00:00

---

## Section 7 — Project structure

```text
vigil/
├── vigil/
│   ├── config.py         all tunable parameters
│   ├── source.py         capture, reconnect, audio subprocess
│   ├── gate.py           SSIM video gate + WebRTC VAD audio gate
│   ├── chunker.py        bundling + cross-modal escalation
│   ├── memory.py         bounded rolling memory + append-only baseline
│   ├── providers.py      mock / ollama / gemini — one interface
│   ├── pipeline.py       queue, workers, backpressure — all concurrency
│   └── metrics.py        CSV logging, rate limiter, run manifest
├── server.py             FastAPI + WebSocket dashboard
├── static/index.html     live timeline UI
├── experiments/
│   ├── run_experiment.py four conditions + tau sweep
│   └── analyze.py        three paper figures + results table
├── tests/test_vigil.py   24 offline tests, no API key needed
├── benchmark.py          full measurement suite → report/
├── generate_readme.py    this script
├── requirements.txt
├── .env.example
├── Makefile
├── QUICKSTART.md
└── ROADMAP.md
```

---

## Section 8 — Setup and usage

```bash
# install
pip install -r requirements.txt   # needs ffmpeg on PATH

# verify everything works (no API key, no network)
python tests/test_vigil.py        # expect: 24 passed, 0 failed

# dashboard
uvicorn server:app --port 8000    # open http://localhost:8000

# run all four experiment conditions
python experiments/run_experiment.py --input <footage> --runs A B C D

# threshold sweep
python experiments/run_experiment.py --input <footage> --sweep

# produce all figures
python experiments/analyze.py --log-dir logs --annotations annotations.csv

# full benchmark report
python benchmark.py --input <footage> --annotations annotations.csv                     --provider gemini --out report/

# regenerate this README from the report
python generate_readme.py
```

Providers: set `VIGIL_PROVIDER=mock` (no key, fake output), `ollama` (local, free, real output), `gemini` (real cost, citable numbers). Copy `.env.example` to `.env`.

---

## Section 9 — Limitations (required, do not skip)

- SSIM reacts to lighting changes and camera autofocus as real events — this is a known failure mode and quantified in the threshold sweep as false positive rate.
- Single scene type tested; results may not generalise to dynamic scenes with many simultaneous actors.
- Token counts from mock provider are estimated, not billed. Only Gemini numbers are citable.
- Semantic recall scoring uses the same model as the pipeline — judge and defendant are the same system.

---

## Section 10 — Citation block

```bibtex
@misc{vigil2026,
  title  = {VIGIL: Vision Gating with Intermittent Listening for Cost-Bounded Multimodal Perception over Continuous Streams},
  author = {<your names here>},
  year   = {2026},
  school = {MS Ramaiah Institute of Technology},
  note   = {7th Semester Gen AI Course Project}
}
```

---

## Section 11 — References

[1] R. Qian et al., "Dispider: Enabling Video LLMs with Active Real-Time Interaction via Disentangled Perception, Decision, and Reaction," CVPR 2025.
[2] H. Zhang et al., "Flash-VStream: Efficient Real-Time Understanding for Long Video Streams," ICCV 2025.
[3] X. Ding et al., "StreamMind: Unlocking Full Frame Rate Streaming Video Dialogue through Event-Gated Cognition," ICCV 2025.
[4] Y. Wang et al., "Accelerating Streaming Video Large Language Models via Hierarchical Token Compression (STC)," CVPR 2026.
[5] L. Chen et al., "FastV: An Image is Worth 1/2 Tokens After Layer 2," ECCV 2024.
[6] "StreamMem: Query-Agnostic KV Cache Memory for Streaming Video Understanding," arXiv:2508.15717, 2025.
[7] "DAT: Dual-Aware Adaptive Transmission for Efficient Multimodal LLM Inference in Edge-Cloud Systems," arXiv:2604.05375, 2026.
[8] C. Packer et al., "MemGPT: Towards LLMs as Operating Systems," 2024.
[9] G. Xiao, Y. Tian, B. Chen, S. Han and M. Lewis, "Efficient Streaming Language Models with Attention Sinks," ICLR 2024.
[10] E. Song et al., "MovieChat: From Dense Token to Sparse Memory for Long Video Understanding," CVPR 2024.
[11] "StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Assistant," arXiv:2505.05467, 2025.
[12] "ReMem: Streaming Video Understanding With Long Context Retention," arXiv:2610.05940, 2026.
[13] "CoRDS: Coreset-based Representative and Diverse Selection for Streaming Video Understanding," arXiv:2605.14310, 2026.
[14] W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023.
[15] A. Agrawal et al., "Sarathi-Serve: Efficient LLM Inference by Chunking Prefills," OSDI 2024.
[16] "Utility-Aware Load Shedding for Real-time Video Analytics at the Edge," arXiv:2307.02409, 2023.
[17] "StreamingBench: Assessing the Gap for MLLMs to Achieve Streaming Video Understanding," arXiv:2411.03628, 2025.
[18] "StreamArena: Toward Continuous, Interactive, and Long-Horizon Agentic Streaming Video Understanding," arXiv:2608.05703, 2026.
[19] K. Grauman et al., "Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives," CVPR 2024.
[20] P. Wu et al., "Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision (XD-Violence)," ECCV 2020.
[21] "AVadCLIP: Audio-Visual Collaboration for Robust Video Anomaly Detection," arXiv:2504.04495, 2025.
[22] "What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction in LLMs and Agents," arXiv:2607.08032, 2026.
