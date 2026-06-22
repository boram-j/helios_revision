# HELIOS — Implementation batch for the agent

**Context:** Calibration is already done (`helios_calib.txt`) and we have three measured HELIOS runs (256×512, 512×256, 1974×1028, with HELIOS-Tile + HELIOS-Fused measured and Naive/FixedOrient analytical). **Do not redo calibration or those three runs.** Tasks below split into (A) start now and (B) hold until we lock the planner spine. **Report back after task A1 (the sweep) before launching the FHE runs.**

---

## A. Start now (parallelizable)

### A1. Cost-surface sweep — plaintext, no FHE. HIGHEST PRIORITY.
- Build NC voter bucket-size distributions at **coarse / medium / fine** blocking granularities.
- Using the existing calibration, compute predicted total runtime per granularity (sum the per-bucket cost model over all buckets).
- Then **sweep the FHE parameters**: vary slot size S (e.g., 8192 / 16384 / 32768) and the comparison cost, and recompute the predicted-best granularity at each setting.
- Output: (i) predicted runtime per granularity for NC, and (ii) a table/plot showing how the **optimal granularity shifts** as S and comparison cost change.
- This is cheap and it is the go/no-go on the paper's core claim. **Run it first and report before starting the FHE work.**

### A2. Small all-backend MEASURED baseline — FHE, ~16×16 (or 8×8 if you want it faster).
- Measure **all four backends end-to-end in real FHE**: Naive, FixedOrient, HELIOS-Tile, HELIOS-Fused. (The existing three runs measured only HELIOS; Naive/Fixed were analytical, so we currently have zero measured baseline points.)
- Use the **current synthetic band predicate + COUNT** so it is directly comparable to the analytical model.
- Goal: show measured Naive/Fixed ≈ analytical Naive/Fixed, to anchor the extrapolated 25 h / 50 h / 194 h numbers.

### A3. Kernel — add `SUM(payload)`.
- Currently only COUNT is implemented. Add a fused `SUM` over a **bounded** payload alongside COUNT, on the same execution path.

### A4. Kernel — real date-band residual.
- Replace the synthetic band (OFFSET / CMP_LO / CMP_HI) with a registration-date band predicate `ABS(A.reg_date − B.reg_date) ≤ Δ` on real NC fields. Keep a bounded synthetic payload for SUM if no real numeric payload exists.

### A5. Memory instrumentation.
- Replace the single `PeakCTs` view with four **measured** categories, applied identically to all backends: layout/input CTs, peak working CTs, materialized temporary CTs, total resident CTs.

### A6. Held-out characterization — plaintext.
- Pull a **second-state voter file** (or a meaningfully different NC blocking scheme) and produce its bucket distributions across granularities.
- Specifically flag whether its **cost-optimal granularity differs from NC's** — that is the property we want.

---

## B. Hold — do NOT run until the planner spine and held-out are locked

- Predict-then-verify planner FHE measurements.
- Measured mini-workload FHE run.
- Held-out representative / p99 / largest FHE runs.
- Non-selected-granularity validation point.
- Tiny-bucket co-packing — do not build unless we decide to claim it.
