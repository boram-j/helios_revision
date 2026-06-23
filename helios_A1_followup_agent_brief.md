# HELIOS — A1 Follow-up Agent Brief: Three Plaintext Decisions That Pick the Paper

**All plaintext / analytical. No FHE. Do not launch any FHE run, and do not lock the held-out, until this follow-up is reported.**

---

## Why we are doing this (context)

A1 was the cost-surface sweep. It established two things and broke a third:

1. **Non-monotonicity is real** on the NC soundex ladder: at S=32,768, coarse ≈ 9.1 M CMPs, medium ≈ 494 K, fine ≈ 2.76 M. Medium beats both — and beats fine *even though fine has fewer candidate pairs*. That is the FHE-specific insight: candidate-pair count is the wrong objective.
2. **The optimum does NOT move.** Medium won all 18 (S × lt_mult) cells. lt_mult is a global multiplier on the dominant comparison term, so it scales every granularity equally and can never reorder the winner — the six lt_mult columns are effectively one experiment. So "the optimum shifts with parameters → you need a planner" is empirically dead on this dataset.
3. **The absolute number is the real problem.** The *best* cell in the entire sweep — medium @ S=32,768 — is 24,130 hours ≈ 2.75 years. Naive is ~100 years. So the honest framing today is "infeasible → still infeasible," not "infeasible → feasible."

Why medium beats fine in A1: each bucket pays a **comparison floor** of 2 CMPs regardless of size. Fine creates 1.38 M tiny buckets × 2 = 2.76 M CMPs even though total *work* is lowest at fine. That floor is the only reason fine loses.

**The pivot — co-packing.** That floor is an artifact of running each tiny bucket in its own near-empty ciphertext (24 useful slots out of 16,384, still paying the full 175 s comparison). Co-packing = laying pairs from many tiny buckets into one ciphertext and firing **one** shared comparison across all of them. It is sound here because (a) the band predicate `|A.reg_date − B.reg_date| ≤ Δ` is identical for every pair, and (b) the aggregate is a **global** COUNT/SUM (no GROUP BY), so a single sum over the whole ciphertext collects the answer. Cross-bucket pairs are never placed in the slots (blocking enforces that at layout-construction time, in plaintext), so the global sum only counts valid within-bucket matches.

**The bind co-packing creates.** Turning it on does two opposite things at once:
- It removes fine's floor → fine's cost approaches (total pairs ÷ slots). Total pairs is *lowest* at fine, so the optimum likely flips medium → fine, and the absolute drops from ~years toward days/weeks. **Feasibility is rescued.**
- But once cost ≈ total pairs, "minimize cost" = "minimize candidate pairs" = the plaintext min-pairs heuristic. HELIOS would now *agree* with plaintext intuition instead of beating it. **The planner's one distinctive win (medium-over-fine, 5.6×) dissolves.**

So feasibility and planner-novelty are coupled in opposite directions, with co-packing as the hinge. Everything below exists to find out, cheaply and in plaintext, **which side of that hinge we land on** — before spending a months-long FHE campaign on the wrong paper.

A note on two candidate escape routes we already pressure-tested:
- **Orientation** (inner/outer choice) is a genuine FHE-specific decision plaintext doesn't have, but (a) it is the *already-published* HELIOS contribution, so it can't be the *new* spine; (b) it's a local two-option argmin; (c) its value concentrates in large lopsided buckets — exactly the ones that don't co-pack. So orientation is at best a supporting lever, and we need to measure how much of the workload it even touches (Experiment 3).
- **Co-packing grouping** ("which buckets to group") was proposed as the real planner. But it is only a hard optimization if co-packed buckets must preserve per-bucket structure (different orientations/shapes that can't be shared). With a uniform predicate + global sum, tiny buckets may just flatten into a pair-stream where greedy slot-fill ≈ optimal. So grouping being non-trivial is a *hypothesis to test* (Experiment 2), not an assumption.

---

## The three experiments

### A1-2 — Decide the co-packing floor question
Extend the A1 cost model with a co-packing path for tiny buckets:
- tiny buckets flattened into shared ciphertexts;
- cost = (total surviving pairs ÷ slots) × comparison cost **+ an explicit layout term** (rotations to place each co-packed bucket into shared slots, ~112 ms each; plus any per-group setup).
- **Do not assume the layout term is zero.** Model it as scaling with #buckets or #groups and report sensitivity.

Recompute coarse / medium / fine totals with co-packing ON.
Report: does fine now win, by how much, and what is the new absolute (hours) for the winner? How sensitive is the winner to the layout-cost term — is there a layout cost at which medium and fine cross back?

### A1-3 — Decide whether grouping is non-trivial at all
For the co-packed (tiny-bucket) regime, compare grouping strategies on predicted cost:
- **Greedy slot-fill:** flatten all surviving pairs, fill ciphertexts in order.
- **Shape/orientation-aware grouping:** group by compatible shape/orientation to minimize padding and shared-schedule waste.
- (Optional, small sample only) near-optimal bin-pack as a lower bound.

Report: total predicted CMPs, slot utilization, padding waste, #groups, and **the gap between greedy and shape-aware**. If the gap is ≈ 0 (plausible, given uniform predicate + global sum), say so plainly — that means grouping is **not** a planner contribution and we stop pursuing it.

### A1-4 — Quantify the orientation tail
Work-weighted across the NC distribution: what fraction of total predicted cost lives in buckets **too large to co-pack** (the lopsided tail where orientation genuinely changes comparison count)?
For that tail, also report the orientation ablation: per bucket, cost(A outer) vs cost(B outer) vs chosen; fraction of buckets where orientation flips; work-weighted speedup from orientation selection; median / p95 / max benefit.
Report: is the orientation-relevant tail ~5% of work (footnote) or ~40% (real lever)?

---

## How to read the result (decides the paper)

- If **A1-2** fine wins cleanly with a feasible absolute, **A1-3** grouping gap ≈ 0, and **A1-4** the orientation tail is small → the honest conclusion is that HELIOS is a strong FHE residual-join **backend/kernel**, not a global planner; venue target drops to ICDE/EDBT, and we reframe to "fast feasible band-join kernel."
- If any one breaks in our favor — **A1-2** co-packing carries a real (layout-driven) floor that keeps the optimum non-obvious, OR **A1-3** shape-aware grouping beats greedy by a meaningful margin, OR **A1-4** the orientation tail is a heavy share of work — then *that specific result* is the planner spine, and we build the FHE validation around it.

**Report all three numbers before we decide anything else.** No held-out lock, no FHE run, until then.
