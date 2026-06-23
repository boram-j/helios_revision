#!/usr/bin/env bash
# run_parallel_scaling.sh
# ============================================================
# Tier-3 parallel scaling demonstration for HELIOS.
#
# Runs the same bucket (256×512) N_workers times concurrently using
# background processes and measures total wall-clock for each worker
# count.  Reports speedup, throughput, and parallel efficiency.
#
# The benchmark binary is independent per-invocation (no shared state),
# so scaling is purely limited by CPU core count.
#
# Usage:
#   cd <build_dir>           # where bin/helios_bucket_bench lives
#   chmod +x ../../run_parallel_scaling.sh
#   ../../run_parallel_scaling.sh
#
# Output:
#   logs/T3b_scaling_results.csv    — machine-readable results
#   logs/T3b_scaling.log            — full console output
#   logs/T3b_scale_<N>_w<w>.txt    — per-worker log
#
# After running:
#   python3 analysis/plot_scaling.py
# ============================================================

set -euo pipefail

BENCH="${BENCH:-./bin/helios_bucket_bench}"
LOG_DIR="${LOG_DIR:-./logs}"
SCALING_LOG="$LOG_DIR/T3b_scaling.log"
RESULTS_CSV="$LOG_DIR/T3b_scaling_results.csv"

# Bucket to use for scaling measurement (must already be representable;
# 256×512 takes ~21 min on togepi — a good benchmark unit).
SCALE_N="${SCALE_N:-256}"
SCALE_M="${SCALE_M:-512}"

WORKER_COUNTS="${WORKER_COUNTS:-1 2 4 8}"

mkdir -p "$LOG_DIR"

# Helpers
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
hr()  { printf -- '=%.0s' {1..80}; echo; }

# Write CSV header
echo "# HELIOS Parallel Scaling — ${SCALE_N}×${SCALE_M} bucket, $(date '+%Y-%m-%d')" \
    > "$RESULTS_CSV"
echo "workers,total_wall_s,throughput_buckets_per_hr,parallel_efficiency" \
    >> "$RESULTS_CSV"

echo "" | tee -a "$SCALING_LOG"
hr | tee -a "$SCALING_LOG"
echo "[$(ts)]  HELIOS T3b Parallel Scaling  bucket=${SCALE_N}×${SCALE_M}" \
    | tee -a "$SCALING_LOG"
echo "  Workers to test: $WORKER_COUNTS" | tee -a "$SCALING_LOG"
hr | tee -a "$SCALING_LOG"
echo "" | tee -a "$SCALING_LOG"

# Portable nanosecond-resolution timestamp
now_ns() {
    # date +%s%N works on Linux; fallback to python3 on macOS
    if date +%s%N 2>/dev/null | grep -q '^[0-9]*[0-9]$'; then
        date +%s%N
    else
        python3 -c "import time; print(int(time.time()*1e9))"
    fi
}

WALL_1="0"   # will be set on first (1-worker) run

for N_WORKERS in $WORKER_COUNTS; do
    echo "--- N_workers=${N_WORKERS} ---" | tee -a "$SCALING_LOG"
    echo "[$(ts)]  Starting ${N_WORKERS} concurrent bucket(s)..." | tee -a "$SCALING_LOG"

    TN_START=$(now_ns)

    PIDS=()
    for ((w=1; w<=N_WORKERS; w++)); do
        WLOG="$LOG_DIR/T3b_scale_${N_WORKERS}_w${w}.txt"
        stdbuf -oL "$BENCH" "$SCALE_N" "$SCALE_M" > "$WLOG" 2>&1 &
        PIDS+=($!)
        echo "  Launched worker ${w} (pid=${PIDS[-1]}) → $WLOG" | tee -a "$SCALING_LOG"
    done

    # Wait for all workers to finish; collect exit codes
    FAILED=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "  WARNING: worker pid=${pid} exited non-zero" | tee -a "$SCALING_LOG"
            FAILED=$((FAILED+1))
        fi
    done

    TN_END=$(now_ns)

    # Compute elapsed in seconds (bc for floating-point)
    WALL_N=$(echo "scale=3; ($TN_END - $TN_START) / 1000000000" | bc)

    # Throughput = buckets completed / hours elapsed
    THROUGHPUT=$(echo "scale=4; $N_WORKERS * 3600 / $WALL_N" | bc)

    # Parallel efficiency = (1-worker wall) / (N × N-worker wall)
    if [ "$N_WORKERS" -eq 1 ]; then
        WALL_1="$WALL_N"
        EFFICIENCY="1.0000"
    else
        if [ "$(echo "$WALL_N > 0" | bc)" -eq 1 ]; then
            EFFICIENCY=$(echo "scale=4; $WALL_1 / ($N_WORKERS * $WALL_N)" | bc)
        else
            EFFICIENCY="N/A"
        fi
    fi

    echo "$N_WORKERS,$WALL_N,$THROUGHPUT,$EFFICIENCY" >> "$RESULTS_CSV"

    echo "" | tee -a "$SCALING_LOG"
    echo "  N_workers  = ${N_WORKERS}" | tee -a "$SCALING_LOG"
    echo "  total_wall = ${WALL_N}s" | tee -a "$SCALING_LOG"
    echo "  throughput = ${THROUGHPUT} buckets/hr" | tee -a "$SCALING_LOG"
    echo "  efficiency = ${EFFICIENCY}" | tee -a "$SCALING_LOG"
    if [ "$FAILED" -gt 0 ]; then
        echo "  WARN: ${FAILED} worker(s) failed — check per-worker logs" \
            | tee -a "$SCALING_LOG"
    fi
    echo "" | tee -a "$SCALING_LOG"
done

hr | tee -a "$SCALING_LOG"
echo "[$(ts)]  Scaling run complete." | tee -a "$SCALING_LOG"
echo "  CSV:   $RESULTS_CSV" | tee -a "$SCALING_LOG"
echo "  Log:   $SCALING_LOG" | tee -a "$SCALING_LOG"
echo "" | tee -a "$SCALING_LOG"
echo "  Plot:  python3 analysis/plot_scaling.py" | tee -a "$SCALING_LOG"
echo "         (saves to analysis/outputs/T3b_scaling.png)" | tee -a "$SCALING_LOG"
hr | tee -a "$SCALING_LOG"
