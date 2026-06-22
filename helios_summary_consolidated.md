# HELIOS — Consolidated Summary: Framing, Challenges, Contributions, Experiments

**Working title:** HELIOS: Cost-Based Physical Planning for Encrypted Residual Join Aggregation under Packed FHE
**Target venue:** VLDB (preferred) / SIGMOD. Honest fallback if the planner stays shallow: ICDE / EDBT / PETS.

---

## 1. Position (one sentence)

HELIOS is a cost-based physical planner for encrypted residual join aggregation: given candidate buckets emitted at multiple blocking granularities, it selects the granularity *and* the per-bucket packed-FHE layout that minimize comparison-dominated execution cost under slot-utilization and leakage constraints, then compiles the chosen plan onto a packed-BFV backend.

---

## 2. Scope — what it is / is not

**Is** a physical-planning + execution layer for the operator

```sql
SELECT AGG(...) FROM A, B
WHERE candidate(A, B) AND residual(A, B);
```

where `candidate(A,B)` comes from upstream blocking / placement / PSI-style candidate generation; `residual(A,B)` is a non-equality pairwise predicate (band / range / inequality) depending on both records' attributes; `AGG ∈ {COUNT, SUM(payload)}`; and each bucket holds many tuples per side (bag semantics → an `n_b × m_b` residual grid).

**Is not** a new matching primitive (concede fuzzy / distance-aware / circuit-PSI, PSI-Sum) and not a general encrypted SQL engine.

**Boundary claim:** matching protocols and FHE compilers do not solve the *physical-planning* problem of executing many skewed residual-join buckets under packed-FHE slot constraints.

---

## 3. Challenges claimed

- **C1 — Bag-semantics residual join-agg.** Not equality PSI, not a single fuzzy-match primitive: many buckets, many tuples per side, predicate over the tuple-pair grid.
- **C2 — Non-monotonic blocking granularity (load-bearing).** Finer blocking is *not* monotonically better under packed FHE: coarse → oversized buckets, fine → fragmented under-filled ciphertexts, medium → best. Candidate-pair count is the wrong objective, and the optimum is **parameter-dependent** (shifts with slot size S and comparison cost).
- **C3 — Per-bucket orientation + tile shape dominate comparison cost.** Encrypted comparison is the runtime driver; layout decides how many comparisons fire.
- **C4 — Materialized residual grids are impractical.** One mask per pair/tile blows up temporary ciphertexts; must fuse mask + aggregation.
- **C5 — Leakage must be explicit.** Public blocking leaks bucket structure, sizes, and skew; the planner's granularity choice is itself data-dependent leakage.

---

## 4. Contributions (solutions to the challenges)

- **(Setup, not a headline) Operator formalization.** Residual join-agg over candidate buckets as a relational physical operator with bag semantics. Framing only — do not sell as a standalone contribution.
- **K1 — Cost-based GLOBAL planner (central).** Selects blocking granularity *and* per-bucket layout jointly. Its claim to being a planner (not a cost model with `argmin`) rests on two demonstrable properties:
  - the cost-optimal granularity **moves** with FHE parameters (S, comparison cost) — there is no fixed answer, so a planner is genuinely required;
  - it **beats the strongest plaintext-DB heuristic** (choose granularity by minimum candidate pairs), because FHE cost ≠ pair count.
  - Evaluated **predict-then-verify**, never post-hoc ("we tried three and medium won" is not acceptable).
- **K2 — Comparison-amortizing packed-BFV backend.** Repeats the inner side across BFV rows, batches outer records per ciphertext, uses shifted band comparisons, takes orientation/tile from the planner, fuses mask + aggregation, and never materializes the grid.

---

## 5. Baselines & positioning

- **NSHEDB-style packed nested-loop** — named (not "naive"); baseline + lineage, not the contribution.
- **Fixed-orientation** — isolates orientation from full planning.
- **Strong plaintext-planner baseline** — pick granularity by *minimum candidate pairs* (standard DB heuristic). HELIOS must beat **this**, not an "always-finest" strawman.
- **PSI family** (fuzzy / distance-aware / circuit-PSI / PSI-Sum) — conceded; boundary = private matching vs. physical planning over buckets.
- **FHE compilers** (CHET, HECO, Coyote, Porcupine, EVA, HEIR, Fhelipe) — boundary = fixed circuits / tensor dataflows vs. a relational plan space (granularity, bucket distribution, orientation, fusion).
- **MPC SQL** (SMCQL, Conclave) — different trust / execution model.

---

## 6. Current evidence (measured)

Calibration (done): isLessThan(ct,pt) ≈175 s · rot ≈112 ms · mul+relin ≈352 ms · mul_pt ≈1.1 ms · add ≈0.70 ms.

Three real-derived bucket runs — HELIOS measured; baselines analytical from calibration:

| Bucket | CMP (Naive → HELIOS) | Reduction | HELIOS measured | Naive (extrap.) |
|---|---|---|---|---|
| 256×512 | 512 → 8 | 64× | 21.4 m | ~25 h |
| 512×256 | 1024 → 8 | 128× | 21.5 m | ~50 h |
| 1974×1028 | 3948 → 130 | 30× | 5.8 h | ~194 h |

These establish the **mechanism**: comparison-count reduction, the orientation effect (256×512 vs 512×256), and that measured HELIOS runtime tracks the calibrated cost model. They validate the **backend**, not the global planner — which is exactly the gap the experiments below exist to close.

---

## 7. Final experiment set (consolidated, with status)

**Legend:** `[DONE]` · `[NOW]` assignable immediately · `[HOLD]` needs the planner spine + held-out choice locked first.

### Plaintext / analytical (no FHE)

- **P1 `[NOW]` Cost-surface sweep — highest priority.** NC bucket distributions at coarse/medium/fine × a sweep of S and comparison cost → predicted runtime per granularity. Must show (a) a non-monotonic optimum exists and (b) the optimum **moves** with parameters. This is the spine go/no-go *and* the justification for calling it a planner.
- **P2 `[NOW]` Workload characterization across granularities** (bucket counts, candidate pairs, regime fractions, work concentration, monster buckets). Feeds P1.
- **P3 `[NOW]` Held-out dataset characterization** — ideally find one whose cost-optimal granularity *differs* from NC's, so the held-out tests the *decision*, not just cost-model calibration.
- **P4 `[later, analytical]` Full-workload extrapolation** per mode/backend, backed by measured validation.

### FHE (validation points only)

- **F1 `[NOW]` Small all-backend MEASURED baseline (~16×16).** Measure Naive + Fixed + HELIOS-Tile + HELIOS-Fused end-to-end; show measured ≈ analytical → anchors the extrapolated 25 h / 50 h / 194 h. *Distinct from existing runs, which measured only HELIOS.*
- **F2 `[HOLD]` Predict-then-verify planner.** Planner predicts best granularity + per-bucket plan; measure its selected representative buckets; show predicted ≈ measured and the plaintext-heuristic choice loses.
- **F3 `[HOLD]` Measured mini-workload** (~hundreds of real buckets, selected granularity, end-to-end) — removes reliance on pure extrapolation.
- **F4 `[HOLD]` Held-out runs** — representative / p95–p99 / largest-feasible from predicted-best granularity, with real date-band residual + `SUM(payload)`.
- **F5 `[HOLD, optional]` Non-selected-granularity point** — one feasible bucket from coarse/fine to show why the planner rejects it.
- **Reuse** existing 256×512 / 512×256 / 1974×1028 for in-distribution backend + orientation evidence.

### Backend / kernel work (enables the F-runs)

- **B1 `[NOW]` Add `SUM(payload)`** aggregation (currently COUNT only); bounded payload, same fused path.
- **B2 `[NOW]` Wire a real date-band residual** (registration-date) replacing the synthetic band; bounded synthetic payload acceptable with disclosure.
- **B3 `[NOW]` Memory re-instrumentation** — layout / working / materialized / resident CT categories, measured, identical definitions across all backends.

---

## 8. Risk & venue gate

VLDB-credible **iff**: the plan space is more than `argmin` over three hand-built blockings; the optimum-moves result holds; the planner beats the strong plaintext baseline; predict-then-verify lands on a held-out with a *different* optimum; a measured mini-workload exists; the leakage model is explicit; and real residual + `SUM(payload)` is shown. If the planner stays a per-bucket compiler / three-way feasibility classifier, the honest target drops to ICDE / EDBT / PETS.
