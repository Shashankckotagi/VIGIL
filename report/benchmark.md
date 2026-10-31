# VIGIL Comprehensive System Measurement Report
**Generated:** 2026-10-29T13:52:16.643121+00:00  
**Provider:** `mock` (`mock-v1`) | **Footage:** `sample/demo_clip.avi` (60.0s, 320x240)

## Summary Results Table
| Run | Gate | Memory | Escalation | τ | Retention % | Total Tokens | Cost (USD) | p95 Latency | Temp. Recall | Audio Recall | Precision |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | OFF | Append | OFF | 0.92 | 100.0% | 50,570 | $0.0040 | 0.16s | 100.0% | 100.0% | 100.0% |
| **B** | ON | Append | OFF | 0.92 | 30.0% | 11,544 | $0.0009 | 0.16s | 100.0% | 100.0% | 100.0% |
| **C** | ON | Bounded | OFF | 0.92 | 30.0% | 11,141 | $0.0009 | 0.15s | 100.0% | 100.0% | 100.0% |
| **D** | ON | Bounded | ON | 0.92 | 34.2% | 12,113 | $0.0010 | 0.16s | 100.0% | 100.0% | 100.0% |
| **S1** | ON | Bounded | OFF | 0.80 | 0.8% | 341 | $0.0000 | 0.15s | 25.0% | 0.0% | 100.0% |
| **S2** | ON | Bounded | OFF | 0.86 | 0.8% | 341 | $0.0000 | 0.15s | 25.0% | 0.0% | 100.0% |
| **S3** | ON | Bounded | OFF | 0.92 | 30.0% | 11,141 | $0.0009 | 0.15s | 100.0% | 100.0% | 100.0% |
| **S4** | ON | Bounded | OFF | 0.96 | 34.2% | 12,099 | $0.0010 | 0.16s | 100.0% | 100.0% | 100.0% |
| **S5** | ON | Bounded | OFF | 0.99 | 34.2% | 12,100 | $0.0010 | 0.16s | 100.0% | 100.0% | 100.0% |

---
## Objective 1 — Cost Boundedness & Memory Flatness
- **Token Cost Reduction (Run C vs Baseline A):** `78.0%` reduction (`4.54x` factor).
- **Cumulative Token Slope Ratio (Run C):** `1.318` (Flatness threshold < 1.30: NO).
- **Cumulative Token Slope Ratio (Run A):** `1.397` (Growing memory baseline).
- **Prompt Length Ratio Q4 / Q1 (Run C):** `1.561` (characters).

## Objective 2 — Responsiveness & Stability
- **Max Queue Depth (Run C):** `0` / 10.
- **Quarterly Mean Queue Depth (Run C):** Q1=`1.00`, Q2=`1.00`, Q3=`1.00`, Q4=`1.00`.
- **Backpressure Events Triggered (Run C):** `0` (0.00s total blocked).
- **HTTP 429 Faults:** `0` | **Errors:** `0` | **Retries:** `0`.

## Objective 3 — Cross-Modal Audio Escalation (D vs C)
- **Escalations Triggered:** `1` (False escalation rate: `0.0%`).
- **Additional Frames Kept vs C:** `+5` frames.
- **Recall Delta on Audio-Cued Events:** `+0.0` pp.
- **Extra Token Cost vs C:** `+8.7%`.
- **Concentration Ratio:** `0.40` (fraction of extra frames within 30s of acoustic burst onset).

## Threshold Sweep (S1–S5) Tradeoff Curve
| Sweep | τ | Retention % | Tokens Total | Tokens / Hour | Temporal Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **S1** | 0.80 | 0.8% | 341 | 20,460.0 | 25.0% |
| **S2** | 0.86 | 0.8% | 341 | 20,460.0 | 25.0% |
| **S3** | 0.92 | 30.0% | 11,141 | 729,229.1 | 100.0% |
| **S4** | 0.96 | 34.2% | 12,099 | 799,200.0 | 100.0% |
| **S5** | 0.99 | 34.2% | 12,100 | 799,266.1 | 100.0% |

---
## CAVEATS
1. **Provider Attribution & Token Billing:** All measurements above were conducted using the `mock` provider (`mock-v1`). If mock was used, token counts are estimated heuristic character/image proxies, not real API billing charges.
2. **Perception & Recall Validity with Mock Provider:** The mock provider outputs strings drawn from a small deterministic phrase list. Consequently, temporal and semantic recall scores obtained with `--provider mock` are synthetic indicators of timing overlap, not measurements of real vision model understanding. High-fidelity recall benchmarks require Gemini or Ollama.
3. **Footage Scope:** Evaluated on a single contiguous clip (`sample/demo_clip.avi`, duration: 60.0s). Results characterize single-camera fixed view performance and may differ across complex scenes featuring dynamic crowds or pan-tilt movements.
4. **Annotations & Agreement:** Ground truth consists of 4 event annotations. Second-annotator cross-validation was not conducted for this run.
5. **Execution Health:** Errors=0, Reconnects=0. All runs completed successfully to the specified limit without premature termination.
