# A1 Follow-up Decision Summary

_Answers the three questions from `helios_A1_followup_agent_brief.md` and applies the decision logic._

---

## Q1 — A1-2: Does fine win with co-packing ON?

**Yes, unambiguously.**

At the reference operating point (γ=0, no layout overhead), fine achieves **307.7 hrs** total cost versus medium's **4,477 hrs** — a **14.5× improvement** over medium under co-packing, and a **438× improvement** over the medium-without-co-packing baseline (24,130 hrs). At γ=0, fine's cost breaks down as: large-bucket cost 4.88 hrs + tiny co-pack cost 302.85 hrs. At γ=30 (heavy layout overhead), fine rises only to **310.6 hrs** while medium rises to **4,484.7 hrs**.

**There is no positive-γ crossover.** Medium requires 8,237 co-packing CTs; fine requires only 3,115 (fewer because fine's 25 large buckets consume almost no slots). Since fine has both lower large-bucket cost and fewer tiny CTs, medium's layout overhead grows faster than fine's as γ increases. The algebraic crossover point is at γ < 0 — physically impossible. Fine dominates at all achievable layout costs.

The absolute winner is **fine at 308 hrs ≈ 13 days**. Feasibility is rescued.

---

## Q2 — A1-3: Is the grouping gap ≈ 0?

**Yes. Grouping is not a planner contribution — and greedy is actually better than shape-aware.**

| Granularity | Greedy (CTs / CMPs) | Shape-aware (CTs / CMPs) | Padding waste (shape-aware) |
|------------|---------------------|--------------------------|-----------------------------|
| Medium | 8,237 / 16,474 | 8,304 / 16,608 | 0.81% |
| Fine | 3,115 / 6,230 | 3,182 / 6,364 | 2.12% |

Shape-aware grouping is consistently worse (more CTs, more padding) because partitioning by inner dimension creates 126–135 pools whose remainders waste slots. The gap between greedy and shape-aware is **+0.81% to +2.12% in greedy's favor** — essentially zero, and in the wrong direction for a planner claim. **Grouping is not a planner contribution.**

---

## Q3 — A1-4: What fraction of total cost lives in the orientation tail?

**~1.6% — not even the 5% footnote level.**

At fine granularity (the winner), only **25 of 1,380,502 buckets** are too large to co-pack. Their combined cost is **4.88 hrs out of 307.73 hrs total = 1.59%**. The final-verdict summary confirms orientation saves zero hours at fine. At medium granularity, orientation saves 4.1 hrs (1.2% flip rate). In neither case does the orientation tail reach ~40%; it is squarely a footnote phenomenon, not a real lever.

---

## Decision

**Scenario: all three break against the "planner" framing.**

- A1-2: Fine wins cleanly, absolute is feasible (308 hrs).
- A1-3: Grouping gap ≈ 0 (< 2.2%, greedy wins).
- A1-4: Orientation tail ≈ 1.6% of work (< 5% footnote threshold).

Per the brief's decision logic, the conclusion is: **HELIOS is a strong FHE band-join backend/kernel, not a global planner.** The non-monotonicity insight (medium beats both coarse and fine without co-packing) is real, but co-packing reverses the winner and makes the optimal choice obvious to any competent engineer.

**Recommended framing:** "Fast feasible FHE band-join kernel."  
**Venue target: ICDE or EDBT** (not VLDB/SIGMOD).

The co-packing scope question (in/out) should be resolved explicitly: if a principled argument exists for keeping co-packing out of scope (GROUP-BY semantics, DP noise budget, key-confidentiality), that argument can restore the VLDB/SIGMOD path on the medium-granularity non-monotonicity result, as detailed in `a1_experiments_5.py` Final Verdict (Path A).
