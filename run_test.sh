#!/usr/bin/env bash
# run_helios_bench.sh
# Sequential HELIOS benchmark runs — no parallelism, clean wall-clock measurements.
#
# Usage:
#   chmod +x run_helios_bench.sh
#   cd <build_dir>
#   ../run_helios_bench.sh
#
# Output:
#   togepi/bench_quick.txt       — 1-seed correctness check (quick sanity)
#   togepi/bench_256x512.txt     — 256 x 512
#   togepi/bench_512x256.txt     — 512 x 256  (orientation swap)
#   togepi/bench_1974x1028.txt   — 1974 x 1028  (NC Bucket C)
#   helios_bench_results.csv     — appended by each run
#
# Note: stdbuf -oL forces line-buffered output from the C++ binary so progress
#       prints appear immediately even when piped through tee.

set -euo pipefail

BENCH=./bin/helios_bucket_bench
LOG_DIR=./togepi

mkdir -p "$LOG_DIR"

CSV=./helios_bench_results.csv

# ts and hr as plain functions (no () when calling inside $())
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
hr()  { printf -- '-%.0s' {1..80}; echo; }

run_bench() {
    local label="$1"
    local n_b="$2"
    local m_b="$3"
    local logfile="$LOG_DIR/bench_${n_b}x${m_b}.txt"

    hr
    echo "[$(ts)]  START  $label  (n_b=$n_b  m_b=$m_b)"
    echo "  log -> $logfile"
    hr

    local t_start
    t_start=$(date +%s)

    stdbuf -oL "$BENCH" "$n_b" "$m_b" 2>&1 | tee "$logfile"
    local bench_exit=${PIPESTATUS[0]}

    local t_end
    t_end=$(date +%s)
    local elapsed=$(( t_end - t_start ))

    hr
    echo "[$(ts)]  DONE   $label  elapsed=${elapsed}s  (~$(( elapsed / 60 ))m $(( elapsed % 60 ))s)"
    hr
    echo

    if [ "$bench_exit" -ne 0 ]; then
        echo "ERROR: bench exited with code $bench_exit"
        exit "$bench_exit"
    fi
}

# ----------------------------------------------------------------
# 0. Quick sanity check (1-seed correctness)
# ----------------------------------------------------------------
hr
echo "[$(ts)]  Quick correctness check (1 seed, n=8 m=6) ..."
hr

stdbuf -oL "$BENCH" quick 2>&1 | tee "$LOG_DIR/bench_quick.txt"
quick_exit=${PIPESTATUS[0]}

if [ "$quick_exit" -ne 0 ]; then
    echo "ERROR: quick check failed. Aborting."
    exit 1
fi
echo "[$(ts)]  Quick check PASSED."
echo

# ----------------------------------------------------------------
# 1. 256 x 512
# ----------------------------------------------------------------
run_bench "256x512" 256 512

# ----------------------------------------------------------------
# 2. 512 x 256  (orientation swap — confirms symmetry)
# ----------------------------------------------------------------
run_bench "512x256" 512 256

# ----------------------------------------------------------------
# 3. 1974 x 1028  (NC Bucket C)
# ----------------------------------------------------------------
run_bench "1974x1028" 1974 1028

# ----------------------------------------------------------------
# Summary
# ----------------------------------------------------------------
hr
echo "[$(ts)]  All runs complete."
echo "  CSV: $CSV"
if [ -f "$CSV" ]; then
    echo "  Rows in CSV: $(( $(wc -l < "$CSV") - 1 ))"
fi
hr
