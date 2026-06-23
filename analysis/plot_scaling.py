#!/usr/bin/env python3
"""
plot_scaling.py — HELIOS T3b parallel scaling visualizer.

Reads  ../logs/T3b_scaling_results.csv   (written by run_parallel_scaling.sh)
Saves  analysis/outputs/T3b_scaling.png

Usage (from helios_revision root):
    python3 analysis/plot_scaling.py

Or from anywhere:
    python3 /path/to/helios_revision/analysis/plot_scaling.py
"""

import csv
import os
import sys

# ── Locate CSV ────────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)

CSV_PATH = os.environ.get(
    "SCALING_CSV",
    os.path.join(_root, "logs", "T3b_scaling_results.csv"),
)
OUT_DIR  = os.path.join(_here, "outputs")
OUT_PATH = os.path.join(OUT_DIR, "T3b_scaling.png")

if not os.path.exists(CSV_PATH):
    print(f"ERROR: results file not found: {CSV_PATH}")
    print("  Run run_parallel_scaling.sh from the build directory first.")
    sys.exit(1)

# ── Parse CSV ─────────────────────────────────────────────────────────────────
workers, total_wall, throughput, efficiency = [], [], [], []

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(
        (row for row in f if not row.strip().startswith("#"))
    )
    for row in reader:
        try:
            workers.append(int(row["workers"]))
            total_wall.append(float(row["total_wall_s"]))
            throughput.append(float(row["throughput_buckets_per_hr"]))
            eff = row["parallel_efficiency"].strip()
            efficiency.append(float(eff) if eff not in ("N/A", "") else None)
        except (ValueError, KeyError) as e:
            print(f"  Skipping malformed row: {row}  ({e})")

if not workers:
    print("ERROR: no valid data rows found in CSV.")
    sys.exit(1)

# Speedup relative to 1-worker baseline
wall_1   = total_wall[0]
speedup  = [wall_1 / w for w in total_wall]
ideal_sp = list(workers)  # ideal linear scaling

# ── Import matplotlib ─────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # headless rendering
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("matplotlib not found.  Install with:  pip install matplotlib")
    sys.exit(1)

# ── Layout ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "HELIOS T3b — Embarrassingly Parallel Scaling\n"
    "(256×512 bucket, N_workers = " + "/".join(str(w) for w in workers) + ")",
    fontsize=12, fontweight="bold", y=1.02,
)

BLUE  = "#1f77b4"
GREEN = "#2ca02c"
RED   = "#d62728"
GRAY  = "#888888"

# ── Subplot 1: Speedup ────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(workers, speedup,  "o-", color=BLUE,  linewidth=2, markersize=8,
        label="Measured")
ax.plot(workers, ideal_sp, "--", color=GRAY,  linewidth=1, label="Ideal linear")
ax.set_xlabel("Concurrent workers")
ax.set_ylabel("Speedup (× 1-worker time)")
ax.set_title("Speedup")
ax.set_xticks(workers)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Annotate speedup values
for x, y in zip(workers, speedup):
    ax.annotate(f"{y:.2f}×", xy=(x, y), xytext=(4, 4),
                textcoords="offset points", fontsize=8, color=BLUE)

# ── Subplot 2: Throughput ─────────────────────────────────────────────────────
ax = axes[1]
ax.plot(workers, throughput, "s-", color=GREEN, linewidth=2, markersize=8)
ax.set_xlabel("Concurrent workers")
ax.set_ylabel("Buckets completed / hour")
ax.set_title("Throughput")
ax.set_xticks(workers)
ax.grid(True, alpha=0.3)

# Annotate throughput values
for x, y in zip(workers, throughput):
    ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(4, 4),
                textcoords="offset points", fontsize=8, color=GREEN)

# ── Subplot 3: Parallel efficiency ───────────────────────────────────────────
ax = axes[2]
eff_valid   = [e for e in efficiency if e is not None]
wk_valid    = [w for w, e in zip(workers, efficiency) if e is not None]
if eff_valid:
    ax.plot(wk_valid, eff_valid, "^-", color=RED, linewidth=2, markersize=8)
ax.axhline(y=1.0, color=GRAY, linestyle="--", linewidth=1, label="Ideal (1.0)")
ax.set_xlabel("Concurrent workers")
ax.set_ylabel("Parallel efficiency  (ideal = 1.0)")
ax.set_title("Parallel Efficiency")
ax.set_ylim(0, 1.25)
ax.set_xticks(workers)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Annotate efficiency values
for x, y in zip(wk_valid, eff_valid):
    ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(4, 4),
                textcoords="offset points", fontsize=8, color=RED)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")

# Also print a text summary
print("\n  Workers | Wall (s) | Speedup | Throughput (b/hr) | Efficiency")
print("  " + "-" * 64)
for i, w in enumerate(workers):
    eff_str = f"{efficiency[i]:.3f}" if efficiency[i] is not None else "N/A"
    print(f"  {w:7d} | {total_wall[i]:8.1f} | {speedup[i]:7.2f}× | "
          f"{throughput[i]:17.2f} | {eff_str}")
