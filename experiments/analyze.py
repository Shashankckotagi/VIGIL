#!/usr/bin/env python3
"""Turn run CSVs into the paper's figures and results table.

    python experiments/analyze.py --log-dir logs --annotations annotations.csv

Figure 1  recall vs tokens/hour across the tau sweep
Figure 2  cumulative tokens vs elapsed time, run A against run C  <- headline
Figure 3  recall on audio-cued events, run C against run D

Recall needs annotations.csv with columns: start_s,end_s,label
Without it the script still produces Figure 2 and the cost table, which is
enough to see whether the project works before annotation is finished.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt    # noqa: E402
import pandas as pd                # noqa: E402

MATCH_WINDOW_S = 10.0   # a description counts if it lands within this window


def load_runs(log_dir: str) -> dict[str, pd.DataFrame]:
    runs = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        if len(df):
            runs[name] = df
    return runs


def load_manifests(log_dir: str) -> dict[str, dict]:
    out = {}
    for path in glob.glob(os.path.join(log_dir, "*.json")):
        name = os.path.splitext(os.path.basename(path))[0]
        if name == "summary":
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                out[name] = json.load(fh)
        except Exception:
            pass
    return out


def compute_recall(df: pd.DataFrame, ann: pd.DataFrame | None,
                   only_audio_cued: bool = False) -> float | None:
    """Fraction of annotated events with a description inside the window.

    Deliberately generous: it asks whether the system said anything at the
    right moment, not whether the wording is correct. Semantic scoring is
    left for manual review on a sample -- see the roadmap.
    """
    if ann is None or df.empty or ann.empty:
        return None
    rows = ann[ann["label"].str.contains("audio", case=False, na=False)] \
        if only_audio_cued else ann
    if rows.empty:
        return None

    times = df["stream_t"].astype(float).to_numpy()
    hits = 0
    for _, ev in rows.iterrows():
        lo = float(ev["start_s"]) - MATCH_WINDOW_S
        hi = float(ev["end_s"]) + MATCH_WINDOW_S
        if ((times >= lo) & (times <= hi)).any():
            hits += 1
    return hits / len(rows)


MIN_SPAN_S = 60.0   # below this a per-hour rate is meaningless


def tokens_per_hour(df: pd.DataFrame) -> float:
    """Tokens per hour of *stream* time.

    Returns NaN when the run is too short to extrapolate: a two-chunk run
    spanning four seconds would otherwise report millions of tokens an hour
    and poison every figure it appears in.
    """
    if df.empty or len(df) < 2:
        return float("nan")
    span = float(df["stream_t"].max() - df["stream_t"].min())
    if span < MIN_SPAN_S:
        return float("nan")
    total = float(df["tokens_in"].sum() + df["tokens_out"].sum())
    return total / (span / 3600.0)


def fig_cost_curve(runs, ann, out_path):
    """Figure 2 -- the headline. Flat line vs climbing line."""
    plt.figure(figsize=(6.2, 4.0))
    plotted = 0
    for name in sorted(runs):
        df = runs[name]
        if not name.upper().startswith(("A", "C")) and len(runs) > 2:
            continue
        cum = (df["tokens_in"] + df["tokens_out"]).cumsum()
        plt.plot(df["stream_t"] / 60.0, cum, label=name, linewidth=1.9)
        plotted += 1
    if not plotted:
        for name, df in runs.items():
            cum = (df["tokens_in"] + df["tokens_out"]).cumsum()
            plt.plot(df["stream_t"] / 60.0, cum, label=name, linewidth=1.9)
    plt.xlabel("Elapsed stream time (minutes)")
    plt.ylabel("Cumulative tokens")
    plt.title("Cost growth: ungated baseline vs bounded pipeline")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25, linewidth=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  wrote {out_path}")


def fig_recall_vs_cost(runs, ann, out_path):
    """Figure 1 -- the tau sweep tradeoff curve."""
    pts = []
    for name, df in runs.items():
        if "sweep" not in name.lower():
            continue
        r = compute_recall(df, ann)
        pts.append((tokens_per_hour(df), r, name,
                    float(df["tau"].iloc[0]) if "tau" in df else None))
    if not pts:
        print("  (no sweep runs; skipping Figure 1)")
        return
    pts = [p for p in pts if p[0] == p[0]]   # drop NaN rates
    if not pts:
        print("  (sweep runs too short to compute a rate; skipping Figure 1)")
        return
    pts.sort()
    xs = [p[0] for p in pts]
    ys = [p[1] if p[1] is not None else float("nan") for p in pts]

    plt.figure(figsize=(6.2, 4.0))
    plt.plot(xs, ys, "o-", linewidth=1.9, markersize=6)
    for x, y, name, tau in pts:
        if tau is not None and y == y:
            plt.annotate(f"τ={tau:.2f}", (x, y), textcoords="offset points",
                         xytext=(6, 6), fontsize=8)
    plt.xlabel("Tokens per hour")
    plt.ylabel("Event recall")
    plt.title("Recall vs cost across gate thresholds")
    plt.grid(alpha=0.25, linewidth=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  wrote {out_path}")


def fig_escalation(runs, ann, out_path):
    """Figure 3 -- run C vs run D on audio-cued events."""
    c = runs.get("C")
    d = runs.get("D")
    if c is None or d is None or ann is None:
        print("  (need runs C and D plus annotations; skipping Figure 3)")
        return
    labels, vals = [], []
    for name, df in (("C (visual only)", c), ("D (escalated)", d)):
        r = compute_recall(df, ann, only_audio_cued=True)
        labels.append(name)
        vals.append(0.0 if r is None else r)
    plt.figure(figsize=(5.0, 4.0))
    plt.bar(labels, vals, width=0.5)
    plt.ylabel("Recall on audio-cued events")
    plt.ylim(0, 1)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f"{v*100:.0f}%", ha="center", fontsize=10)
    plt.title("Cross-modal escalation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  wrote {out_path}")


def results_table(runs, manifests, ann) -> pd.DataFrame:
    rows = []
    for name in sorted(runs):
        df = runs[name]
        man = manifests.get(name, {})
        gate = man.get("gate", {})
        met = man.get("metrics", {})
        rows.append({
            "run": name,
            "frame_retention_%": round(gate.get("frame_retention", 0) * 100, 1),
            "audio_retention_%": round(gate.get("audio_retention", 0) * 100, 1),
            "tokens_total": int(df["tokens_in"].sum() + df["tokens_out"].sum()),
            "tokens_per_hour": (round(tokens_per_hour(df), 0)
                                if tokens_per_hour(df) == tokens_per_hour(df)
                                else ""),
            "p95_latency_s": met.get("p95_latency_s", ""),
            "max_queue_depth": met.get("max_queue_depth", ""),
            "rate_limit_faults": met.get("rate_limit_faults", ""),
            "escalations": gate.get("escalations", 0),
            "recall": (round(compute_recall(df, ann), 3)
                       if compute_recall(df, ann) is not None else ""),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = load_runs(args.log_dir)
    if not runs:
        print(f"No run CSVs found in {args.log_dir}/")
        return
    print(f"Loaded {len(runs)} run(s): {', '.join(sorted(runs))}")

    ann = None
    if args.annotations and os.path.exists(args.annotations):
        ann = pd.read_csv(args.annotations)
        if "label" not in ann:
            ann["label"] = ""
        ann["label"] = ann["label"].fillna("").astype(str)
        print(f"Loaded {len(ann)} annotated events")
    else:
        print("No annotations supplied -- recall columns will be blank")

    manifests = load_manifests(args.log_dir)
    fig_cost_curve(runs, ann, os.path.join(args.out_dir, "fig2_cost.png"))
    fig_recall_vs_cost(runs, ann, os.path.join(args.out_dir, "fig1_recall.png"))
    fig_escalation(runs, ann, os.path.join(args.out_dir, "fig3_escalation.png"))

    table = results_table(runs, manifests, ann)
    csv_out = os.path.join(args.out_dir, "results_table.csv")
    table.to_csv(csv_out, index=False)
    print(f"  wrote {csv_out}\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
