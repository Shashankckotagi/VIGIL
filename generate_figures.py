"""
generate_figures.py — VIGIL Research Paper Figures Generator
Generates publication-quality figures (IEEE standard):
  Fig 1: Threshold Sweep (1400x1000px)
  Fig 2: Cumulative Token Growth (1400x1000px)
  Fig 3: Conditions A-D Comparison (1400x900px)
  Fig 4: System Architecture Diagram (1200x2000px)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def ensure_exact_dimensions(filepath, target_w, target_h):
    """Ensure output PNG matches exact target pixel dimensions."""
    with Image.open(filepath) as img:
        if img.size != (target_w, target_h):
            resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            resized.save(filepath)

# ==============================================================================
# FIGURE 1: Threshold Sweep (τ = 0.80 to 0.99)
# ==============================================================================
def generate_fig1():
    """
    Prompt 1 — Threshold sweep (Figure 1)
    Dual-axis line chart.
    X-axis: SSIM threshold τ with values 0.80, 0.86, 0.92, 0.96, 0.99.
    Left Y-axis: Event recall (%) in blue, values 25, 25, 100, 100, 100.
    Right Y-axis: Tokens per hour (×1000) in red dashed line, values 20, 20, 729, 799, 799.
    Vertical dotted grey line at τ=0.92 labelled “operating point”.
    Title: “Fig. 1 — Recall vs. Token Cost Across τ Sweep”.
    Style: white background, serif font, IEEE paper quality, minimal gridlines, legend in top-right.
    Export: PNG 1400×1000px.
    """
    out_path = os.path.join(OUTPUT_DIR, "fig1_threshold_sweep.png")
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 18
    })

    fig, ax1 = plt.subplots(figsize=(7.0, 5.0), dpi=200)
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    tau_vals = [0.80, 0.86, 0.92, 0.96, 0.99]
    recall_vals = [25, 25, 100, 100, 100]
    token_vals = [20, 20, 729, 799, 799]

    # Left axis: Recall
    color_recall = "#1f77b4"  # IEEE standard blue
    line1 = ax1.plot(tau_vals, recall_vals, color=color_recall, marker="o", linewidth=2.5, markersize=8, label="Event recall (%)")
    ax1.set_xlabel(r"SSIM Threshold ($\tau$)", fontweight="bold", labelpad=10)
    ax1.set_ylabel("Event Recall (%)", color=color_recall, fontweight="bold", labelpad=10)
    ax1.tick_params(axis="y", labelcolor=color_recall)
    ax1.set_ylim(-5, 130)
    ax1.set_xticks(tau_vals)
    ax1.grid(True, linestyle="--", alpha=0.35, color="grey")

    # Right axis: Tokens/hr (x1000)
    ax2 = ax1.twinx()
    color_token = "#d62728"  # IEEE red
    line2 = ax2.plot(tau_vals, token_vals, color=color_token, linestyle="--", marker="s", linewidth=2.5, markersize=8, label=r"Tokens / hr ($\times 10^3$)")
    ax2.set_ylabel(r"Tokens per Hour ($\times 10^3$)", color=color_token, fontweight="bold", labelpad=10)
    ax2.tick_params(axis="y", labelcolor=color_token)
    ax2.set_ylim(-40, 1050)

    # Operating point vertical line
    op_tau = 0.92
    ax1.axvline(x=op_tau, color="#666666", linestyle=":", linewidth=2.2)
    
    # Text annotation for operating point
    ax1.annotate(
        "Operating point\n(τ = 0.92)",
        xy=(op_tau, 55),
        xytext=(0.84, 55),
        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9fa", ec="#666666", lw=1.0),
        fontsize=11,
        fontfamily="serif",
        ha="center"
    )

    # Combined Legend top-right (as specified in Prompt 1)
    lines = line1 + line2 + [matplotlib.lines.Line2D([0], [0], color="#666666", linestyle=":", linewidth=2.2)]
    labels = ["Event recall (%)", r"Tokens / hr ($\times 10^3$)", "Operating point (τ=0.92)"]
    ax1.legend(lines, labels, loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", framealpha=0.95)

    plt.title("Fig. 1 — Recall vs. Token Cost Across τ Sweep", pad=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    ensure_exact_dimensions(out_path, 1400, 1000)
    print(f"Generated Fig 1: {out_path} (1400x1000px)")


# ==============================================================================
# FIGURE 2: Cumulative Token Growth (Headline Figure)
# ==============================================================================
def generate_fig2():
    """
    Prompt 2 — Cumulative token growth (Figure 2 — the headline figure)
    Line chart for a research paper.
    X-axis: Elapsed stream time in seconds, 0 to 60.
    Line 1: “Run A — no gate, append-only memory” in red, rising steeply from 0 to 50570 tokens.
    Line 2: “Run C — VIGIL full pipeline” in blue, rising slowly from 0 to 11141 tokens.
    Gap clearly visible and widens over time.
    Title: “Fig. 2 — Cumulative Token Consumption: Baseline vs. VIGIL”.
    Y-axis label: “Cumulative tokens”.
    Final value labels near the end of each line.
    Style: white background, IEEE paper quality, clean minimal gridlines.
    Export: PNG 1400×1000px.
    """
    out_path = os.path.join(OUTPUT_DIR, "fig2_cumulative_tokens.png")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 18
    })

    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    t = np.linspace(0, 60, 150)
    # Run A: no gate, append-only memory -> quadratic growth as prompt context grows linearly
    # tokens(t) = a*t + b*t^2, tokens(60) = 50570
    # Let b = 10.5, a = 50570/60 - 10.5*60 = 842.8 - 630 = 212.8
    run_a_tokens = 212.8 * t + 10.5 * (t ** 2)

    # Run C: VIGIL full pipeline -> flat linear slope with bounded memory summary
    # tokens(60) = 11141
    run_c_tokens = (11141.0 / 60.0) * t

    # Plot lines
    color_a = "#d62728"  # IEEE Red
    color_c = "#1f77b4"  # IEEE Blue

    ax.plot(t, run_a_tokens, color=color_a, linewidth=3.0, label="Run A — no gate, append-only memory")
    ax.plot(t, run_c_tokens, color=color_c, linewidth=3.0, label="Run C — VIGIL full pipeline")

    # Shaded region to highlight token savings
    ax.fill_between(t, run_c_tokens, run_a_tokens, color=color_c, alpha=0.08, label="Tokens Saved (4.5× Reduction)")

    # Labels at the end of lines
    ax.annotate(
        "50,570 tokens\n(Unbounded O(N²))",
        xy=(60, 50570),
        xytext=(44, 47000),
        arrowprops=dict(arrowstyle="->", color=color_a, lw=1.8),
        fontweight="bold",
        color=color_a,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff1f0", ec=color_a, lw=1.2)
    )

    ax.annotate(
        "11,141 tokens\n(Bounded O(N))",
        xy=(60, 11141),
        xytext=(45, 17500),
        arrowprops=dict(arrowstyle="->", color=color_c, lw=1.8),
        fontweight="bold",
        color=color_c,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f0f7ff", ec=color_c, lw=1.2)
    )

    ax.set_title("Fig. 2 — Cumulative Token Consumption: Baseline vs. VIGIL", pad=16, fontweight="bold")
    ax.set_xlabel("Elapsed stream time in seconds", fontweight="bold", labelpad=10)
    ax.set_ylabel("Cumulative tokens", fontweight="bold", labelpad=10)
    ax.set_xlim(-1, 65)
    ax.set_ylim(-1000, 56000)
    ax.grid(True, linestyle="--", alpha=0.35, color="grey")
    ax.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", framealpha=0.95)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    ensure_exact_dimensions(out_path, 1400, 1000)
    print(f"Generated Fig 2: {out_path} (1400x1000px)")


# ==============================================================================
# FIGURE 3: Conditions Bar Chart (Side-by-Side Panels)
# ==============================================================================
def generate_fig3():
    """
    Prompt 3 — Conditions bar chart (Figure 3)
    Grouped bar chart with two side-by-side panels.
    Left panel: “Frame Retention (%)”: four bars labelled A, B, C, D with values 100, 30, 30, 34.2.
    Right panel: “Total Tokens”: four bars with values 50570, 11544, 11141, 12113.
    Colors: red for A, orange for B, blue for C, green for D.
    Value labels on top of each bar.
    Overall title: “Fig. 3 — Comparison Across Conditions A–D”.
    Export: PNG 1400×900px.
    """
    out_path = os.path.join(OUTPUT_DIR, "fig3_conditions.png")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.titlesize": 18
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 4.5), dpi=200)
    fig.patch.set_facecolor("white")

    conditions = ["A", "B", "C", "D"]
    retention_vals = [100.0, 30.0, 30.0, 34.2]
    token_vals = [50570, 11544, 11141, 12113]
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]  # Red, Orange, Blue, Green

    # Left panel: Frame Retention (%)
    bars1 = ax1.bar(conditions, retention_vals, color=colors, width=0.55, edgecolor="#333333", linewidth=1.2)
    ax1.set_title("Frame Retention (%)", fontweight="bold", pad=12)
    ax1.set_ylabel("Retention Rate (%)", fontweight="bold", labelpad=8)
    ax1.set_ylim(0, 118)
    ax1.grid(axis="y", linestyle="--", alpha=0.35, color="grey")

    for bar, val in zip(bars1, retention_vals):
        height = bar.get_height()
        ax1.annotate(
            f"{val:.1f}%" if val != 100 else "100%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12
        )

    # Right panel: Total Tokens
    bars2 = ax2.bar(conditions, token_vals, color=colors, width=0.55, edgecolor="#333333", linewidth=1.2)
    ax2.set_title("Total Tokens", fontweight="bold", pad=12)
    ax2.set_ylabel("Token Count", fontweight="bold", labelpad=8)
    ax2.set_ylim(0, 59000)
    ax2.grid(axis="y", linestyle="--", alpha=0.35, color="grey")

    for bar, val in zip(bars2, token_vals):
        height = bar.get_height()
        ax2.annotate(
            f"{val:,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12
        )

    # Condition descriptions subtitle/caption
    fig.suptitle("Fig. 3 — Comparison Across Conditions A–D", fontsize=17, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    ensure_exact_dimensions(out_path, 1400, 900)
    print(f"Generated Fig 3: {out_path} (1400x900px)")


# ==============================================================================
# FIGURE 4: System Architecture Diagram (Vertical Flowchart)
# ==============================================================================
def generate_fig4():
    """
    Prompt 4 — System architecture diagram (Figure 4)
    Vertical flowchart / block diagram for a research paper showing the VIGIL system architecture.
    Rectangular blocks with rounded corners connected by downward arrows.
    Order top to bottom:
      1. “INPUT SOURCE” (green border) — subtitle: “YouTube Live · RTSP · Local File · Webcam”
      2. “CAPTURE — source.py” (blue) — subtitle: “OpenCV frames · ffmpeg audio · shared clock”
      3. Two side-by-side blocks:
         “VIDEO GATE — gate.py” (yellow) subtitle “SSIM 128px · no GPU”
         “AUDIO GATE — gate.py” (yellow) subtitle “WebRTC VAD · no GPU”
      4. “ESCALATOR — chunker.py” (light red) — subtitle: “loud audio → raise τ for 30s” — connected from AUDIO GATE with arrow
      5. “CHUNKER — chunker.py” (blue) — subtitle: “3 frames + speech → chunk”
      6. “BOUNDED QUEUE — pipeline.py” (purple border) — subtitle: “asyncio.Queue(10) · backpressure” — sideways label “← full = capture slows”
      7. “WORKER POOL — pipeline.py” (blue) — subtitle: “3 workers · rate limiter · retry”
      8. “MEMORY — memory.py” (blue) — subtitle: “anchor + summary + last 5 descriptions”
      9. “PROVIDER — providers.py” (green border) — subtitle: “mock · ollama · gemini”
    Title at top: “Fig. 4 — VIGIL System Architecture”.
    White background, clean sans-serif font, IEEE paper quality.
    Export: PNG 1200×2000px.
    """
    out_path = os.path.join(OUTPUT_DIR, "fig4_architecture.png")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "figure.titlesize": 18
    })

    # Exact canvas ratio: 1200 / 2000 = 6 / 10
    fig, ax = plt.subplots(figsize=(6.0, 10.0), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title
    ax.text(50, 97.5, "Fig. 4 — VIGIL System Architecture", ha="center", va="center", fontsize=16, fontweight="bold", color="#111111")

    # Helper function to draw rounded boxes
    def draw_box(x_center, y_center, width, height, title, subtitle, bg_color, border_color, border_width=2.0, title_color="#111111"):
        box = patches.FancyBboxPatch(
            (x_center - width / 2.0, y_center - height / 2.0),
            width, height,
            boxstyle="round,pad=0.8,rounding_size=1.8",
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=border_width,
            zorder=3
        )
        ax.add_patch(box)
        ax.text(x_center, y_center + 1.2, title, ha="center", va="center", fontsize=12, fontweight="bold", color=title_color, zorder=4)
        ax.text(x_center, y_center - 1.4, subtitle, ha="center", va="center", fontsize=9.5, color="#333333", zorder=4)

    def draw_arrow(start_x, start_y, end_x, end_y, color="#333333", label=None, label_side="right"):
        ax.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, mutation_scale=15),
            zorder=2
        )
        if label:
            lx = (start_x + end_x) / 2.0 + (1.5 if label_side == "right" else -1.5)
            ly = (start_y + end_y) / 2.0
            ha = "left" if label_side == "right" else "right"
            ax.text(lx, ly, label, ha=ha, va="center", fontsize=9, color="#555555", style="italic")

    # Box dimensions
    standard_w = 76
    standard_h = 5.2

    # 1. INPUT SOURCE (green border)
    y1 = 91.5
    draw_box(50, y1, standard_w, standard_h, "INPUT SOURCE", "YouTube Live  ·  RTSP  ·  Local File  ·  Webcam", "#f4fbf4", "#28a745", border_width=2.5)

    # 2. CAPTURE — source.py (blue)
    y2 = 82.5
    draw_arrow(50, y1 - standard_h/2, 50, y2 + standard_h/2)
    draw_box(50, y2, standard_w, standard_h, "CAPTURE — source.py", "OpenCV frames  ·  ffmpeg audio  ·  shared clock", "#e8f0fe", "#1a73e8")

    # 3. Two side-by-side blocks: VIDEO GATE & AUDIO GATE (yellow)
    y3 = 71.5
    half_w = 36.5
    # Video Gate (Left)
    x3_vid = 29.5
    draw_box(x3_vid, y3, half_w, standard_h + 0.4, "VIDEO GATE — gate.py", "SSIM 128px  ·  no GPU", "#fffde7", "#fbc02d", border_width=2.0)
    # Audio Gate (Right)
    x3_aud = 70.5
    draw_box(x3_aud, y3, half_w, standard_h + 0.4, "AUDIO GATE — gate.py", "WebRTC VAD  ·  no GPU", "#fffde7", "#fbc02d", border_width=2.0)

    # Connect CAPTURE to Gates
    draw_arrow(40, y2 - standard_h/2, x3_vid, y3 + (standard_h+0.4)/2, label="frames", label_side="left")
    draw_arrow(60, y2 - standard_h/2, x3_aud, y3 + (standard_h+0.4)/2, label="audio", label_side="right")

    # 4. ESCALATOR — chunker.py (light red) — connected from AUDIO GATE block
    y4 = 59.5
    x4_esc = 70.5
    draw_arrow(x3_aud, y3 - (standard_h+0.4)/2, x4_esc, y4 + standard_h/2, color="#d32f2f", label="loud audio", label_side="right")
    draw_box(x4_esc, y4, half_w, standard_h, "ESCALATOR — chunker.py", "loud audio → raise τ for 30s", "#ffebee", "#e57373", border_width=2.0)

    # 5. CHUNKER — chunker.py (blue)
    y5 = 48.0
    # Arrow from Video Gate to Chunker
    draw_arrow(x3_vid, y3 - (standard_h+0.4)/2, 38, y5 + standard_h/2, label="kept frames", label_side="left")
    # Arrow from Escalator to Chunker
    draw_arrow(x4_esc, y4 - standard_h/2, 62, y5 + standard_h/2, color="#d32f2f", label="relaxed τ", label_side="right")
    draw_box(50, y5, standard_w, standard_h, "CHUNKER — chunker.py", "3 frames + speech → chunk", "#e8f0fe", "#1a73e8")

    # 6. BOUNDED QUEUE — pipeline.py (purple border)
    y6 = 37.0
    draw_arrow(50, y5 - standard_h/2, 50, y6 + standard_h/2)
    draw_box(50, y6, standard_w, standard_h, "BOUNDED QUEUE — pipeline.py", "asyncio.Queue(10)  ·  backpressure", "#f3e5f5", "#8e24aa", border_width=2.5)

    # Sideways label on queue: ← full = capture slows
    ax.annotate(
        "← full = capture slows",
        xy=(50 + standard_w/2, y6),
        xytext=(50 + standard_w/2 + 2, y6),
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#8e24aa"
    )

    # 7. WORKER POOL — pipeline.py (blue)
    y7 = 26.5
    draw_arrow(50, y6 - standard_h/2, 50, y7 + standard_h/2)
    draw_box(50, y7, standard_w, standard_h, "WORKER POOL — pipeline.py", "3 workers  ·  rate limiter  ·  retry", "#e8f0fe", "#1a73e8")

    # 8. MEMORY — memory.py (blue)
    y8 = 16.5
    draw_arrow(50, y7 - standard_h/2, 50, y8 + standard_h/2)
    draw_box(50, y8, standard_w, standard_h, "MEMORY — memory.py", "anchor + summary + last 5 descriptions", "#e8f0fe", "#1a73e8")

    # 9. PROVIDER — providers.py (green border)
    y9 = 6.5
    draw_arrow(50, y8 - standard_h/2, 50, y9 + standard_h/2)
    draw_box(50, y9, standard_w, standard_h, "PROVIDER — providers.py", "mock  ·  ollama  ·  gemini", "#f4fbf4", "#28a745", border_width=2.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    ensure_exact_dimensions(out_path, 1200, 2000)
    print(f"Generated Fig 4: {out_path} (1200x2000px)")


if __name__ == "__main__":
    print("Generating all 4 IEEE Research Paper Figures...")
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    print("All figures successfully generated in figures/ directory!")
