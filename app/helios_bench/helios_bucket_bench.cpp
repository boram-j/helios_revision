/**
 * helios_bucket_bench.cpp  v5
 *
 * Benchmarks four compiler backends for encrypted band-join |A.x - B.y| <= Delta.
 * Output semantic: global COUNT(*) — one encrypted scalar.
 *
 * A3: Also computes SUM(payload) — a synthetic bounded integer attribute of the
 *     outer record — in the same FHE pass as COUNT (no second pass).
 * A4: Optional date-band mode (pass 'dateband' CLI flag): replaces synthetic
 *     integer values with days-since-epoch dates and a 30-day window.
 * A5: Replaces scalar peak_cts with a 4-category MemStats breakdown:
 *     layout_cts / work_cts / materialized_cts / resident_cts (all analytical).
 *
 * -------------------------------------------------------------------------
 * BFV SLOT GEOMETRY
 * -------------------------------------------------------------------------
 * poly_modulus_degree N = 32768
 * benc.slot_count()    = N = 32768   (total slots; for encode/decode vectors)
 * row_slots            = N/2 = 16384  (rotate_rows wraps within each row)
 *
 * Two-row packing:
 *   p_per_row = floor(row_slots / inner_m)  = floor(16384/1028) = 15
 *   p         = 2 * p_per_row               = 30 outer elements per batch CT
 *   n_batches = ceil(outer_n / p)           = ceil(1974/30) = 66
 *
 * -------------------------------------------------------------------------
 * SHIFTED COMPARISON (avoids signed plaintext encoding pitfalls)
 * -------------------------------------------------------------------------
 * BFV comparators operate over Z_{plain_mod}.  Negative plaintexts wrap to
 * large values (e.g., -17 mod 65537 = 65520) and may not compare correctly
 * with signed semantics depending on the comparator polynomial.
 *
 * We avoid the issue by shifting the difference into a nonneg range:
 *
 *   outer_vec[valid] = OUTER[k] + OFFSET        (OFFSET = VALUE_RANGE = 64)
 *   inner_vec[slot]  = INNER[j]                 (unchanged)
 *   ct_diff          = ct_outer - ct_inner
 *                    = (OUTER[k] + 64) - INNER[j]  ∈ [1, 127]
 *
 * Equivalent predicate (both bounds positive):
 *   |A-B| <= DELTA  iff  OFFSET-DELTA <= diff <= OFFSET+DELTA
 *                   iff  48            <= diff <= 80
 *
 * isLessThan bounds:  lo = 47,  hi = 81   (both in [1, 127], never negative)
 *
 * Padding slots use outer_vec = 0:
 *   diff = 0 - INNER[j] ≡ 65537-INNER[j]  >> 81  → no false match.
 *
 * Date-band mode (A4): OFFSET=DATE_RANGE=3650, INNER values ∈ [10000,13649].
 *   Padding diff = 0-INNER ≡ 65537-INNER ∈ [51888,55537] >> DATE_CMP_HI=3681 ✓
 *
 * -------------------------------------------------------------------------
 * BACKEND NOTES
 * -------------------------------------------------------------------------
 * Backend 1/2: analytical op counts; runtime extrapolated from calibration.
 * Backend 3/4: actual FHE; decrypt+verify global COUNT against plaintext ref.
 *   Both 3 and 4 run the SAME fused-accumulation code path.  The only
 *   difference is in reported peak_cts (analytical): Backend 3 counts all
 *   n_batch mask CTs as live simultaneously (non-fused model); Backend 4
 *   counts just the running accumulator (O(1) model).  Runtime is identical.
 *
 * Build:
 *   cd NSHEDB && mkdir -p build && cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   make helios_bucket_bench -j$(sysctl -n hw.logicalcpu)
 *   ./bin/helios_bucket_bench [N_B] [M_B]
 *   ./bin/helios_bucket_bench quick          # n=8, m=6, 1-seed check
 *   ./bin/helios_bucket_bench dateband       # A4: date-band residual mode
 *   ./bin/helios_bucket_bench recalib        # force fresh calibration
 */

#include "nshedb/nshedb.h"
#include "nshedb/utils/seal_examples.h"
#include "nshedb/utils/timer.h"

#include <seal/seal.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <climits>

using namespace seal;
using namespace nshedb;
using namespace nshedb::core;
using namespace nshedb::utils;
using namespace seal_examples;

// ============================================================
//  Constants
// ============================================================
static const size_t POLY_DEG    = 32768;
static const int    VALUE_RANGE = 64;
static const int    DELTA       = VALUE_RANGE / 4;          // = 16
static const int    OFFSET      = VALUE_RANGE;              // = 64  (shift to nonneg)
static const int    CMP_LO      = OFFSET - DELTA - 1;      // = 47  (positive)
static const int    CMP_HI      = OFFSET + DELTA + 1;      // = 81  (positive)
// outer_vec padding: 0.  diff = 0 - INNER[j] wraps to ~65537 >> CMP_HI → no match.
static const int    PAD_OUTER   = 0;

// A3: payload bound — keeps individual payload values small so per-slot
//     accumulation stays well below plain_mod/n_batches for small test sizes.
static const int    PAYLOAD_BOUND = 100;

// A4: date-band residual constants (days-since-epoch representation)
static const int    BASE_DATE   = 10000;  // epoch offset in days
static const int    DATE_RANGE  = 3650;   // 10-year span in days
static const int    DELTA_DAYS  = 30;     // registration-date window (days)
// Shifted predicate for date mode:
//   diff = (a_date - b_date) + DATE_RANGE  ∈ [DATE_RANGE-3649, DATE_RANGE+3649]
//   match iff DATE_RANGE-30 <= diff <= DATE_RANGE+30
//   isLessThan bounds: DATE_RANGE-31 < diff < DATE_RANGE+31
static const int    DATE_OFFSET = DATE_RANGE;               // = 3650
static const int    DATE_CMP_LO = DATE_RANGE - DELTA_DAYS - 1;  // = 3619
static const int    DATE_CMP_HI = DATE_RANGE + DELTA_DAYS + 1;  // = 3681

static const int    CALIB_N     = 4;
static const int    CALIB_M     = 4;
static const int    SEED_CHECK_N = 8;
static const int    SEED_CHECK_M = 6;
static const int    SEED_CHECK_K = 20;

// ============================================================
//  HE operation counter
// ============================================================
struct OpStats {
    int64_t he_lt     = 0;   // isLessThan(ct, ct)
    int64_t he_ltp    = 0;   // isLessThan(ct, pt) or (pt, ct)
    int64_t he_eq     = 0;   // isEqual
    int64_t he_rot    = 0;   // rotate_rows + rotate_columns
    int64_t he_mul_ct = 0;   // ct x ct + relin
    int64_t he_mul_pt = 0;   // ct x pt
    int64_t he_add_ct = 0;   // ct + ct
    int64_t he_add_pt = 0;   // ct + pt
    int64_t he_sub_ct = 0;   // ct - ct

    int64_t total_cmp() const { return he_lt + he_ltp + he_eq; }
    void reset() {
        he_lt=he_ltp=he_eq=he_rot=he_mul_ct=he_mul_pt=
        he_add_ct=he_add_pt=he_sub_ct=0;
    }
};

// ============================================================
//  Calibrated per-op wall time
// ============================================================
struct PerOpTimes {
    double lt_s     = 0;
    double rot_s    = 0;
    double mul_ct_s = 0;
    double mul_pt_s = 0;
    double add_ct_s = 0;
};

// A5: per-backend memory breakdown (model-based, not runtime-tracked)
struct MemStats {
    int layout_cts       = 0;  // input CTs holding data before computation
    int work_cts         = 0;  // peak scratch CTs alive simultaneously
    int materialized_cts = 0;  // intermediate CTs explicitly stored (tile only)
    int resident_cts     = 0;  // total in memory at peak = layout + work
};

// A3: FHE result bundling COUNT and SUM(outer payload) from one pass
struct FheResult {
    int64_t count = 0;
    int64_t sum   = 0;  // SUM(payload[outer_k]) for all matching (outer_k, inner_j)
};

// ============================================================
//  Helpers
// ============================================================
static int ceildiv(int a, int b) { return (a + b - 1) / b; }
static int ceil_log2(int x)      { return x <= 1 ? 0 : (int)std::ceil(std::log2((double)x)); }

static size_t ct_serial_bytes(const Ciphertext& ct) {
    std::stringstream ss; ct.save(ss); return ss.str().size();
}

// ============================================================
//  Targeted Galois step list
//
//  create_galois_keys() with no args generates keys for ALL rotation
//  steps — O(N) elements at poly_degree=32768, which is very slow.
//
//  We only need:
//    (a) Row rotation steps for rotation_sum_row:
//        s = 1, 2, 4, ..., 8192  (14 steps)
//    (b) Column rotation for rotate_columns:
//        step = 0  (SEAL's special value for the column-swap key)
//  Total: 15 steps → 15 Galois keys instead of O(N).
//
//  Using SEAL's int-steps API avoids having to compute raw Galois
//  elements ourselves (the formula is 3^step mod 2N, and step 0 is
//  a special case that returns element 2N-1 for the column swap).
// ============================================================
static std::vector<int> needed_galois_steps()
{
    std::vector<int> steps;
    steps.push_back(0);  // column rotation (rotate_columns)
    int row_slots = (int)(POLY_DEG / 2);  // = 16384
    for (int s = row_slots >> 1; s >= 1; s >>= 1)
        steps.push_back(s);              // 8192, 4096, ..., 1
    return steps;
}

// ============================================================
//  Plaintext reference: global COUNT
// ============================================================
static int64_t plaintext_count(
    const std::vector<int64_t>& A,
    const std::vector<int64_t>& B,
    int delta)
{
    int64_t cnt = 0;
    for (int64_t a : A)
        for (int64_t b : B)
            if (std::abs(a - b) <= delta) cnt++;
    return cnt;
}

// A3: ground-truth SUM(outer_payload[i]) for all matching (OUTER[i], INNER[j]) pairs.
//     Uses the same orientation that helios_tiling_fhe resolves internally.
static int64_t plaintext_sum_payload(
    const std::vector<int64_t>& OUTER,
    const std::vector<int64_t>& INNER,
    int delta,
    const std::vector<int64_t>& outer_payload)
{
    int64_t s = 0;
    for (int i = 0; i < (int)OUTER.size(); i++)
        for (int j = 0; j < (int)INNER.size(); j++)
            if (std::abs(OUTER[i] - INNER[j]) <= delta)
                s += outer_payload[i];
    return s;
}

// ============================================================
//  fhe_between_shifted
//
//  Inputs:
//    ct_shifted_diff  = (A[i] - B[j]) + offset_val  per slot
//
//  Checks: cmp_lo_val < ct_shifted_diff < cmp_hi_val
//    i.e., for synthetic band: 47 < diff < 81  (|A-B| <= DELTA=16)
//    i.e., for date band:    3619 < diff < 3681 (|A-B| <= 30 days)
//
//  Both bounds are always positive — no signed-mod ambiguity.
//
//  A4: cmp_lo_val / cmp_hi_val are runtime parameters (default to the
//      compile-time constants for backward-compatible callers).
// ============================================================
static Ciphertext fhe_between_shifted(
    Ciphertext&       ct_shifted_diff,
    Comparator&       comp,
    Evaluator&        eval,
    const RelinKeys&  rk,
    BatchEncoder&     benc,
    int               total_slots,
    OpStats&          ops,
    int               cmp_lo_val = CMP_LO,  // A4: runtime band lower bound
    int               cmp_hi_val = CMP_HI)  // A4: runtime band upper bound
{
    std::vector<int64_t> lo_vec(total_slots, (int64_t)cmp_lo_val);
    std::vector<int64_t> hi_vec(total_slots, (int64_t)cmp_hi_val);
    Plaintext pt_lo, pt_hi;
    benc.encode(lo_vec, pt_lo);
    benc.encode(hi_vec, pt_hi);

    // mask_lo: cmp_lo_val < ct_shifted_diff  →  isLessThan(cmp_lo_val, ct)
    auto res_lo = comp.isLessThan(eval, rk, pt_lo, ct_shifted_diff);
    ops.he_ltp++;

    // mask_hi: ct_shifted_diff < cmp_hi_val  →  isLessThan(ct, cmp_hi_val)
    auto res_hi = comp.isLessThan(eval, rk, ct_shifted_diff, pt_hi);
    ops.he_ltp++;

    Ciphertext mask_lo = res_lo[0];
    Ciphertext mask_hi = res_hi[0];

    Ciphertext mask;
    eval.multiply(mask_lo, mask_hi, mask);
    eval.relinearize_inplace(mask, rk);
    ops.he_mul_ct++;
    return mask;
}

// ============================================================
//  rotation_sum_row
//
//  Sums all row_slots positions within each BFV row independently.
//  (rotate_rows affects both rows in lock-step, so after the loop:
//    all slots in row 0 = sum(row 0),
//    all slots in row 1 = sum(row 1).)
//  row_slots must be a power of 2.
// ============================================================
static Ciphertext rotation_sum_row(
    Evaluator&        eval,
    const GaloisKeys& gk,
    Ciphertext        ct,
    int               row_slots,
    OpStats&          ops)
{
    for (int step = row_slots >> 1; step >= 1; step >>= 1) {
        Ciphertext rot;
        eval.rotate_rows(ct, step, gk, rot);
        eval.add_inplace(ct, rot);
        ops.he_rot++;
        ops.he_add_ct++;
    }
    return ct;
}

// ============================================================
//  rotation_sum_both_rows → global scalar in slot[0]
//
//  1. rotation_sum_row   → slot[0] = sum(row0), slot[row_slots] = sum(row1)
//  2. rotate_columns     → swap rows; slot[0] now = sum(row1)
//  3. add_inplace        → slot[0] = sum(row0) + sum(row1) = global COUNT
// ============================================================
static Ciphertext rotation_sum_both_rows(
    Evaluator&        eval,
    const GaloisKeys& gk,
    Ciphertext        ct,
    int               row_slots,
    OpStats&          ops)
{
    ct = rotation_sum_row(eval, gk, ct, row_slots, ops);
    Ciphertext ct_col;
    eval.rotate_columns(ct, gk, ct_col);
    eval.add_inplace(ct, ct_col);
    ops.he_rot++;     // rotate_columns
    ops.he_add_ct++;
    return ct;
}

// ============================================================
//  Analytical op counts for Backend 1 and 2
// ============================================================
static OpStats count_naive(int n_b, int m_b, int total_slots) {
    OpStats ops;
    int b_chunks = ceildiv(m_b, total_slots);
    for (int i = 0; i < n_b; i++) {
        ops.he_rot    += ceil_log2(total_slots) + 1;   // broadcast + col-swap
        ops.he_add_ct += ceil_log2(total_slots) + 1;
        for (int c = 0; c < b_chunks; c++) {
            int active = std::min(m_b - c * total_slots, total_slots);
            ops.he_ltp    += 2;
            ops.he_mul_ct += 1;
            if (c == b_chunks - 1 && active < total_slots)
                ops.he_mul_pt += 1;
            ops.he_rot    += ceil_log2(active) + 1;    // reduce + col-swap
            ops.he_add_ct += ceil_log2(active) + 1;
        }
    }
    return ops;
}

static OpStats count_fixed(int n_b, int m_b, int total_slots) {
    int outer_n  = std::min(n_b, m_b);
    int inner_m  = std::max(n_b, m_b);
    int b_chunks = ceildiv(inner_m, total_slots);
    OpStats ops;
    for (int i = 0; i < outer_n; i++) {
        ops.he_rot    += ceil_log2(total_slots) + 1;
        ops.he_add_ct += ceil_log2(total_slots) + 1;
        for (int c = 0; c < b_chunks; c++) {
            int active = std::min(inner_m - c * total_slots, total_slots);
            ops.he_ltp    += 2;
            ops.he_mul_ct += 1;
            if (c == b_chunks - 1 && active < total_slots)
                ops.he_mul_pt += 1;
            ops.he_rot    += ceil_log2(active) + 1;
            ops.he_add_ct += ceil_log2(active) + 1;
        }
    }
    return ops;
}

// ============================================================
//  Calibration
// ============================================================
static PerOpTimes calibrate(
    Comparator&       comp,
    Evaluator&        eval,
    Encryptor&        encryptor,
    BatchEncoder&     benc,
    const RelinKeys&  rk,
    const GaloisKeys& gk,
    int               total_slots)
{
    std::cout << "  Calibrating ...\n";
    PerOpTimes T;

    auto make_ct = [&](int64_t val) -> Ciphertext {
        std::vector<int64_t> v(total_slots, val);
        Plaintext pt; benc.encode(v, pt);
        Ciphertext ct; encryptor.encrypt(pt, ct);
        return ct;
    };

    // Use a shifted diff value representative of real comparisons
    Ciphertext ct_diff = make_ct((int64_t)(OFFSET + DELTA - 2));

    // isLessThan(ct, pt)  — most expensive op
    {
        std::vector<int64_t> v(total_slots, (int64_t)CMP_HI);
        Plaintext pt; benc.encode(v, pt);
        const int REPS = 5;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < REPS; i++) {
            Ciphertext tmp = ct_diff;
            comp.isLessThan(eval, rk, tmp, pt);
        }
        T.lt_s = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count() / REPS;
        std::cout << "    isLessThan(ct,pt): "
                  << std::fixed << std::setprecision(3) << T.lt_s << " s\n";
    }

    // rotate_rows
    {
        const int REPS = 50;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < REPS; i++) {
            Ciphertext rot; eval.rotate_rows(ct_diff, 1, gk, rot);
        }
        T.rot_s = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count() / REPS;
        std::cout << "    rotate_rows:       " << T.rot_s * 1e3 << " ms\n";
    }

    // ct x ct + relin
    {
        Ciphertext ct2 = make_ct(2);
        const int REPS = 10;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < REPS; i++) {
            Ciphertext res; eval.multiply(ct_diff, ct2, res);
            eval.relinearize_inplace(res, rk);
        }
        T.mul_ct_s = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count() / REPS;
        std::cout << "    mul_ct + relin:    " << T.mul_ct_s * 1e3 << " ms\n";
    }

    // ct x pt
    {
        std::vector<int64_t> v(total_slots, 1);
        Plaintext pt_one; benc.encode(v, pt_one);
        const int REPS = 50;
        auto t0 = std::chrono::high_resolution_clock::now();
        Ciphertext acc = ct_diff;
        for (int i = 0; i < REPS; i++) eval.multiply_plain_inplace(acc, pt_one);
        T.mul_pt_s = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count() / REPS;
        std::cout << "    mul_pt:            " << T.mul_pt_s * 1e3 << " ms\n";
    }

    // ct + ct
    {
        Ciphertext ct2 = make_ct(2);
        const int REPS = 200;
        auto t0 = std::chrono::high_resolution_clock::now();
        Ciphertext acc = ct_diff;
        for (int i = 0; i < REPS; i++) eval.add_inplace(acc, ct2);
        T.add_ct_s = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count() / REPS;
        std::cout << "    add_ct:            " << T.add_ct_s * 1e6 << " us\n";
    }

    return T;
}

static double extrapolate_s(const OpStats& ops, const PerOpTimes& T) {
    return ops.total_cmp()  * T.lt_s
         + ops.he_rot       * T.rot_s
         + ops.he_mul_ct    * T.mul_ct_s
         + ops.he_mul_pt    * T.mul_pt_s
         + (ops.he_add_ct + ops.he_sub_ct + ops.he_add_pt) * T.add_ct_s;
}

// ============================================================
//  Calibration cache — save/load to avoid re-measuring every run.
//  Same machine + same SEAL params → identical values each time.
//  Pass 'recalib' on the command line to force a fresh measurement.
// ============================================================
static const std::string CALIB_CACHE = "./helios_calib.txt";

static void save_calibration(const PerOpTimes& T) {
    std::ofstream f(CALIB_CACHE);
    if (!f) { std::cerr << "  Warning: could not write " << CALIB_CACHE << "\n"; return; }
    f << std::fixed << std::setprecision(9);
    f << "lt_s="    << T.lt_s     << "\n"
      << "rot_s="   << T.rot_s    << "\n"
      << "mulct_s=" << T.mul_ct_s << "\n"
      << "mulpt_s=" << T.mul_pt_s << "\n"
      << "addct_s=" << T.add_ct_s << "\n";
    std::cout << "  Saved calibration to " << CALIB_CACHE << "\n";
}

// Returns {ok, PerOpTimes}.  ok=true only if all 5 fields were loaded.
static std::pair<bool, PerOpTimes> load_calibration() {
    std::ifstream f(CALIB_CACHE);
    if (!f) return {false, {}};
    PerOpTimes T;
    std::map<std::string, double*> fields = {
        {"lt_s",    &T.lt_s    },
        {"rot_s",   &T.rot_s   },
        {"mulct_s", &T.mul_ct_s},
        {"mulpt_s", &T.mul_pt_s},
        {"addct_s", &T.add_ct_s},
    };
    int loaded = 0;
    std::string line;
    while (std::getline(f, line)) {
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        auto it = fields.find(key);
        if (it != fields.end()) {
            try { *it->second = std::stod(line.substr(eq + 1)); loaded++; }
            catch (...) {}
        }
    }
    return {loaded == 5, T};
}

static void print_calib(const PerOpTimes& T) {
    std::cout << std::fixed;
    std::cout << "    isLessThan(ct,pt): " << std::setprecision(3) << T.lt_s       << " s\n";
    std::cout << "    rotate_rows:       " << std::setprecision(3) << T.rot_s*1e3  << " ms\n";
    std::cout << "    mul_ct + relin:    " << std::setprecision(3) << T.mul_ct_s*1e3 << " ms\n";
    std::cout << "    mul_pt:            " << std::setprecision(3) << T.mul_pt_s*1e3 << " ms\n";
    std::cout << "    add_ct:            " << std::setprecision(3) << T.add_ct_s*1e6 << " us\n";
}

// ============================================================
//  HELIOS tiling — actual FHE, two-row packing
//
//  Returns FheResult{count, sum} where:
//    count = global COUNT(*) for band-join predicate
//    sum   = SUM(outer_payload[k]) for all matching (outer_k, inner_j) pairs
//            (0 if a_payload/b_payload are both empty — seed-check path)
//
//  A3: compute SUM(payload) in the same pass as COUNT.
//      payload[k] = OUTER_PAYLOAD[k] encoded in same slot layout as outer CT.
//      ct_contribution = ct_mask * pt_payload; ct_sum += ct_contribution.
//
//  A4: offset_val / cmp_lo_val / cmp_hi_val replace compile-time OFFSET/CMP_LO/CMP_HI,
//      enabling date-band mode without changing the FHE circuit structure.
//
//  Shifted encoding:
//    outer_vec[valid k,j] = OUTER[batch_start+k] + offset_val
//    outer_vec[pad k,j]   = PAD_OUTER = 0
//    inner_vec[k,j]       = INNER[j]
//    ct_diff[slot]        = outer[slot] - inner[slot]
//                         = (A+offset_val-B) for valid  →  ∈ [1, 2*offset_val-1]
//                         = (0 - B)          for pad    →  ≡ large value >> cmp_hi_val
//
//  Both backends (fused=false and fused=true) run the same fused-accumulation
//  code path.  Difference is only in the reported analytical peak_cts (A5).
// ============================================================
static FheResult helios_tiling_fhe(
    int n_b, int m_b,
    const std::vector<int64_t>& A,
    const std::vector<int64_t>& B,
    const std::vector<int64_t>& a_payload,  // A3: payload for A records
    const std::vector<int64_t>& b_payload,  // A3: payload for B records
    Comparator&       comp,
    Evaluator&        eval,
    Encryptor&        encryptor,
    Decryptor&        decryptor,
    BatchEncoder&     benc,
    const RelinKeys&  rk,
    const GaloisKeys& gk,
    int               row_slots,
    int               total_slots,
    OpStats&          ops,
    bool              /*fused*/,   // same execution; only affects peak_cts reporting
    double&           wall_s,
    int               offset_val  = OFFSET,   // A4: runtime shift (OFFSET or DATE_OFFSET)
    int               cmp_lo_val  = CMP_LO,   // A4: runtime lower bound
    int               cmp_hi_val  = CMP_HI)   // A4: runtime upper bound
{
    // Orientation: minimise comparison count
    auto cmp_count_f = [&](int outer, int inner) -> int {
        if (inner > row_slots) return INT_MAX;
        return ceildiv(outer, 2 * (row_slots / inner));
    };
    bool swap_orient = (cmp_count_f(m_b, n_b) < cmp_count_f(n_b, m_b));

    int outer_n = swap_orient ? m_b : n_b;
    int inner_m = swap_orient ? n_b : m_b;
    const std::vector<int64_t>& OUTER = swap_orient ? B : A;
    const std::vector<int64_t>& INNER = swap_orient ? A : B;

    // A3: select payload matching the resolved outer orientation
    bool compute_sum = (!a_payload.empty() && !b_payload.empty());
    const std::vector<int64_t>& OUTER_PAYLOAD = swap_orient ? b_payload : a_payload;

    if (inner_m > row_slots) {
        std::cerr << "  ERROR: inner_m=" << inner_m << " > row_slots=" << row_slots
                  << "; chunked inner not implemented.\n";
        return {-1, 0};
    }

    int p_per_row = row_slots / inner_m;    // floor(16384/1028) = 15
    int p         = 2 * p_per_row;           // 30
    int n_batches = ceildiv(outer_n, p);     // ceil(1974/30) = 66

    // ------------------------------------------------------------------
    // Hoist inner CT (both rows, same inner tiling)
    // inner_vec[k*inner_m + j]             = INNER[j]   row 0
    // inner_vec[row_slots + k*inner_m + j] = INNER[j]   row 1
    // Leftover slots (k >= p_per_row within each half) stay 0.
    // ------------------------------------------------------------------
    std::vector<int64_t> inner_vec(total_slots, 0);
    for (int k = 0; k < p_per_row; k++)
        for (int j = 0; j < inner_m; j++) {
            inner_vec[k * inner_m + j]              = INNER[j];
            inner_vec[row_slots + k * inner_m + j]  = INNER[j];
        }
    Plaintext  pt_inner; benc.encode(inner_vec, pt_inner);
    Ciphertext ct_inner; encryptor.encrypt(pt_inner, ct_inner);

    // ------------------------------------------------------------------
    // Start timing after hoisted inner setup
    // ------------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    Ciphertext ct_acc;
    bool acc_ready = false;

    // A3: accumulator for SUM(payload)
    Ciphertext ct_sum;
    bool sum_acc_ready = false;

    for (int batch = 0; batch < n_batches; batch++) {
        int batch_start = batch * p;
        int batch_end   = std::min(batch_start + p, outer_n);
        int this_p      = batch_end - batch_start;
        int this_p0     = std::min(this_p, p_per_row);
        int this_p1     = std::max(0, this_p - p_per_row);

        // Outer CT:  valid positions = OUTER[batch_start+k] + offset_val  (A4: runtime offset)
        //            pad  positions  = PAD_OUTER = 0
        // Padding diff = 0 - INNER[j] = -(INNER[j]) mod p ≈ 65537-INNER[j] >> cmp_hi_val
        std::vector<int64_t> outer_vec(total_slots, (int64_t)PAD_OUTER);
        for (int k = 0; k < this_p0; k++)
            for (int j = 0; j < inner_m; j++)
                outer_vec[k * inner_m + j] =
                    OUTER[batch_start + k] + (int64_t)offset_val;  // A4
        for (int k = 0; k < this_p1; k++)
            for (int j = 0; j < inner_m; j++)
                outer_vec[row_slots + k * inner_m + j] =
                    OUTER[batch_start + p_per_row + k] + (int64_t)offset_val;  // A4

        Plaintext  pt_outer; benc.encode(outer_vec, pt_outer);
        Ciphertext ct_outer; encryptor.encrypt(pt_outer, ct_outer);

        // ct_diff = (OUTER+offset_val - INNER)  for valid slots
        //         = (0 - INNER)                 for pad slots
        Ciphertext ct_diff;
        eval.sub(ct_outer, ct_inner, ct_diff);
        ops.he_sub_ct++;

        // Slot-wise BETWEEN with (runtime) shifted bounds — A4: passes cmp_lo/hi_val
        Ciphertext ct_mask = fhe_between_shifted(
            ct_diff, comp, eval, rk, benc, total_slots, ops,
            cmp_lo_val, cmp_hi_val);  // A4

        // COUNT accumulation
        if (!acc_ready) {
            ct_acc    = ct_mask;
            acc_ready = true;
        } else {
            eval.add_inplace(ct_acc, ct_mask);
            ops.he_add_ct++;
        }

        // A3: SUM(payload) accumulation — ct_contribution = mask * payload_plaintext
        if (compute_sum) {
            // Build payload plaintext in the same slot layout as outer_vec
            std::vector<int64_t> payload_vec(total_slots, 0);
            for (int k = 0; k < this_p0; k++)
                for (int j = 0; j < inner_m; j++)
                    payload_vec[k * inner_m + j] =
                        OUTER_PAYLOAD[batch_start + k];
            for (int k = 0; k < this_p1; k++)
                for (int j = 0; j < inner_m; j++)
                    payload_vec[row_slots + k * inner_m + j] =
                        OUTER_PAYLOAD[batch_start + p_per_row + k];

            Plaintext  pt_payload; benc.encode(payload_vec, pt_payload);
            Ciphertext ct_contrib;
            eval.multiply_plain(ct_mask, pt_payload, ct_contrib);
            ops.he_mul_pt++;  // A3

            if (!sum_acc_ready) {
                ct_sum        = ct_contrib;
                sum_acc_ready = true;
            } else {
                eval.add_inplace(ct_sum, ct_contrib);
                ops.he_add_ct++;
            }
        }
        // end A3
    }

    // Reduce both rows to global scalar in slot[0]
    ct_acc = rotation_sum_both_rows(eval, gk, ct_acc, row_slots, ops);

    // A3: also reduce SUM CT to global scalar
    if (compute_sum && sum_acc_ready)
        ct_sum = rotation_sum_both_rows(eval, gk, ct_sum, row_slots, ops);

    wall_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // Decrypt and return
    // BFV BatchEncoder decodes in signed Z_{plain_mod}: values > plain_mod/2
    // appear negative (e.g., count=61238 > 32768 → decoded as 61238-65537=-4299).
    // Apply unsigned interpretation so the caller can compare with plaintext gt.
    // For large n×m where gt may exceed plain_mod, the returned value is
    // count mod plain_mod (exact comparison not possible without a larger modulus).
    static const int64_t PLAIN_MOD = 65537;

    Plaintext pt_result; decryptor.decrypt(ct_acc, pt_result);
    std::vector<int64_t> decoded; benc.decode(pt_result, decoded);
    int64_t raw = decoded[0];
    int64_t count_val = ((raw % PLAIN_MOD) + PLAIN_MOD) % PLAIN_MOD;

    // A3: decode SUM
    int64_t sum_val = 0;
    if (compute_sum && sum_acc_ready) {
        Plaintext pt_sum; decryptor.decrypt(ct_sum, pt_sum);
        std::vector<int64_t> dec_sum; benc.decode(pt_sum, dec_sum);
        int64_t raw_sum = dec_sum[0];
        sum_val = ((raw_sum % PLAIN_MOD) + PLAIN_MOD) % PLAIN_MOD;
    }

    return FheResult{count_val, sum_val};
}

// ============================================================
//  Multi-seed correctness check
//  Runs SEED_CHECK_K quick FHE trials (n=8, m=6) with different seeds.
//  Returns true only if all pass.  Catches comparator boundary bugs early.
//  Uses synthetic-band mode (default OFFSET/CMP_LO/CMP_HI) regardless of
//  the outer dateband flag — this tests the core circuit only.
// ============================================================
static bool multi_seed_check(
    Comparator& comp, Evaluator& eval,
    Encryptor& encryptor, Decryptor& decryptor,
    BatchEncoder& benc,
    const RelinKeys& rk, const GaloisKeys& gk,
    int row_slots, int total_slots,
    int n_seeds = SEED_CHECK_K)
{
    std::cout << "[Correctness] " << n_seeds << " seed(s)"
              << " (n=" << SEED_CHECK_N << " m=" << SEED_CHECK_M << ") ...\n";
    // A3: empty payload → skip sum computation in seed check
    const std::vector<int64_t> no_payload;
    int failures = 0;
    for (int seed = 0; seed < n_seeds; seed++) {
        std::cout << "  seed " << (seed + 1) << "/" << n_seeds << " ..." << std::flush;
        std::mt19937_64 rng((uint64_t)seed * 1000003 + 7);
        std::uniform_int_distribution<int64_t> dist(0, VALUE_RANGE - 1);
        std::vector<int64_t> A(SEED_CHECK_N), B(SEED_CHECK_M);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);
        int64_t gt = plaintext_count(A, B, DELTA);
        OpStats ops; ops.reset();
        double wall = 0;
        // A3/A4: pass empty payload and default band params
        FheResult res = helios_tiling_fhe(
            SEED_CHECK_N, SEED_CHECK_M, A, B,
            no_payload, no_payload,
            comp, eval, encryptor, decryptor, benc, rk, gk,
            row_slots, total_slots, ops, /*fused=*/true, wall);
        if (res.count != gt) {
            std::cout << " FAIL (gt=" << gt << " fhe=" << res.count << ")\n";
            std::cout << "    A=["; for (auto v:A) std::cout<<v<<",";
            std::cout << "]\n    B=["; for (auto v:B) std::cout<<v<<",";
            std::cout << "]\n";
            failures++;
        } else {
            std::cout << " PASS  (" << std::fixed << std::setprecision(1)
                      << wall << "s  gt=" << gt << ")\n";
        }
    }
    if (failures == 0)
        std::cout << "  ALL PASS (" << n_seeds << "/" << n_seeds << " seeds correct)\n\n";
    else
        std::cout << "  FAIL (" << failures << " seed(s) incorrect out of " << n_seeds << ")\n\n";
    return failures == 0;
}

// ============================================================
//  Results
// ============================================================
struct BenchResult {
    std::string name;
    OpStats     ops;
    double      wall_actual_s = 0;
    double      wall_extrap_s = 0;
    int64_t     peak_cts      = 0;
    double      peak_mem_mb   = 0;
    bool        correct       = false;
    bool        ran_actual    = false;
    // A3: SUM(payload) results
    int64_t     sum_fhe       = 0;   // FHE-computed sum (mod plain_mod)
    int64_t     sum_gt        = 0;   // ground-truth sum (mod plain_mod)
    bool        sum_correct   = false;
    // A5: detailed memory breakdown
    MemStats    mem_stats     = {};
};

static void print_sep(int w = 134) { std::cout << std::string(w, '-') << "\n"; }

static void write_csv(
    const std::vector<BenchResult>& R,
    int n_b, int m_b,
    int row_slots, int total_slots,
    double ct_mb,
    int64_t gt_count,
    int64_t gt_sum,    // A3
    bool dateband)     // A4
{
    std::string path = "./helios_bench_results.csv";
    std::ofstream f(path, std::ios::app);
    if (!f.is_open()) { std::cerr << "  Warning: could not write CSV\n"; return; }
    // Header on first write (check if file empty)
    f.seekp(0, std::ios::end);
    if (f.tellp() == 0)
        f << "n_b,m_b,total_slots,row_slots,ct_mb,gt_count,dateband,"
          << "backend,cmp_tot,he_rot,he_mul_ct,he_mul_pt,he_add_ct,he_sub_ct,"
          << "wall_actual_s,wall_extrap_s,peak_cts,peak_mem_mb,correct,"
          // A3
          << "sum_fhe,sum_gt,"
          // A5
          << "layout_cts,work_cts,mat_cts,resident_cts\n";
    for (const auto& r : R) {
        f << n_b << "," << m_b << "," << total_slots << "," << row_slots << ","
          << std::fixed << std::setprecision(2) << ct_mb << ","
          << gt_count << ","
          << (dateband ? "date" : "synth") << ","    // A4
          << "\"" << r.name << "\","
          << r.ops.total_cmp() << ","
          << r.ops.he_rot << ","
          << r.ops.he_mul_ct << ","
          << r.ops.he_mul_pt << ","
          << r.ops.he_add_ct << ","
          << r.ops.he_sub_ct << ","
          << std::fixed << std::setprecision(4) << r.wall_actual_s << ","
          << r.wall_extrap_s << ","
          << r.peak_cts << ","
          << std::setprecision(2) << r.peak_mem_mb << ","
          << (r.ran_actual ? (r.correct ? "PASS" : "FAIL") : "analytic") << ","
          // A3: sum_gt stored as gt_sum%PM for all backends; always output mod-reduced
          << r.sum_fhe << "," << r.sum_gt << ","
          // A5
          << r.mem_stats.layout_cts << ","
          << r.mem_stats.work_cts << ","
          << r.mem_stats.materialized_cts << ","
          << r.mem_stats.resident_cts
          << "\n";
    }
    std::cout << "  CSV appended to " << path << "\n\n";
}

static void print_results(
    const std::vector<BenchResult>& R,
    int n_b, int m_b,
    int64_t gt_count,
    int64_t gt_sum,     // A3
    int row_slots, int total_slots,
    double ct_mb,
    bool dateband,      // A4
    int active_delta)   // A4
{
    double naive_cmp = R.empty() ? 1.0 :
        (double)std::max((int64_t)1, R[0].ops.total_cmp());

    std::cout << "\n" << std::string(134, '=') << "\n";
    std::cout << "  HELIOS Bench  n_b=" << n_b << "  m_b=" << m_b
              << "  row_slots=" << row_slots << "  total_slots=" << total_slots
              << "  Delta=" << active_delta                       // A4: show active delta
              << (dateband ? "  [DATE-BAND]" : "  [SYNTHETIC]") // A4: mode indicator
              << "  gt=" << gt_count
              << "  gt_sum=" << gt_sum                           // A3
              << "  ct=" << std::fixed << std::setprecision(2) << ct_mb << " MB\n";
    std::cout << std::string(134, '=') << "\n";

    std::cout << "\n[HE Operation Counts]   (A) = analytical; no FHE executed\n";
    print_sep();
    std::cout << std::left  << std::setw(24) << "Backend"
              << std::right
              << std::setw(8)  << "LT"
              << std::setw(8)  << "EQ"
              << std::setw(8)  << "ROT"
              << std::setw(8)  << "MULct"
              << std::setw(8)  << "MULpt"
              << std::setw(8)  << "ADDct"
              << std::setw(8)  << "SUBct"
              << std::setw(10) << "CMP"
              << std::setw(12) << "Reduction"
              << "\n";
    print_sep();
    for (const auto& r : R) {
        double red = naive_cmp / (double)std::max((int64_t)1, r.ops.total_cmp());
        std::string lbl = r.name + (r.ran_actual ? "" : " (A)");
        std::cout << std::left  << std::setw(24) << lbl
                  << std::right
                  << std::setw(8)  << r.ops.he_lt + r.ops.he_ltp
                  << std::setw(8)  << r.ops.he_eq
                  << std::setw(8)  << r.ops.he_rot
                  << std::setw(8)  << r.ops.he_mul_ct
                  << std::setw(8)  << r.ops.he_mul_pt
                  << std::setw(8)  << r.ops.he_add_ct
                  << std::setw(8)  << r.ops.he_sub_ct
                  << std::setw(10) << r.ops.total_cmp()
                  << std::setw(11) << std::fixed << std::setprecision(1) << red << "x"
                  << "\n";
    }
    print_sep();

    std::cout << "\n[Wall Time & Memory]   Backend 3+4 share same exec path; peak_cts differs analytically\n";
    print_sep();
    std::cout << std::left  << std::setw(24) << "Backend"
              << std::right
              << std::setw(16) << "WallActual"
              << std::setw(16) << "WallExtrap"
              << std::setw(12) << "PeakCTs"
              << std::setw(14) << "PeakMem(MB)"
              << std::setw(12) << "Correct?"
              << "\n";
    print_sep();
    for (const auto& r : R) {
        auto fmt_s = [](double s) {
            std::ostringstream ss;
            if      (s >= 3600) ss << std::fixed << std::setprecision(1) << s/3600 << "h";
            else if (s >= 60)   ss << std::fixed << std::setprecision(1) << s/60   << "m";
            else                ss << std::fixed << std::setprecision(2) << s      << "s";
            return ss.str();
        };
        std::string corr = r.ran_actual ? (r.correct ? "PASS" : "FAIL") : "(analytic)";
        std::cout << std::left  << std::setw(24) << r.name
                  << std::right
                  << std::setw(16) << (r.ran_actual ? fmt_s(r.wall_actual_s) : "-")
                  << std::setw(16) << fmt_s(r.wall_extrap_s)
                  << std::setw(12) << r.peak_cts
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.peak_mem_mb
                  << std::setw(12) << corr
                  << "\n";
    }
    print_sep();

    // A3: SUM(payload) results section
    std::cout << "\n[SUM(payload) Results]   A3 — outer payload[k] = (k % " << PAYLOAD_BOUND << ") + 1\n";
    print_sep(134);
    std::cout << std::left  << std::setw(24) << "Backend"
              << std::right
              << std::setw(20) << "SUM_FHE"
              << std::setw(20) << "SUM_GT(mod)"
              << std::setw(12) << "SUM OK?"
              << "\n";
    print_sep(134);
    static const int64_t PM_PRINT = 65537;
    for (const auto& r : R) {
        if (!r.ran_actual) {
            std::cout << std::left  << std::setw(24) << (r.name + " (A)")
                      << std::right
                      << std::setw(20) << "-"
                      << std::setw(20) << gt_sum % PM_PRINT
                      << std::setw(12) << "(analytic)"
                      << "\n";
        } else {
            std::string s_ok = r.sum_correct ? "PASS" : "FAIL";
            std::cout << std::left  << std::setw(24) << r.name
                      << std::right
                      << std::setw(20) << r.sum_fhe
                      << std::setw(20) << (gt_sum % PM_PRINT)
                      << std::setw(12) << s_ok
                      << "\n";
        }
    }
    print_sep(134);

    // A5: Memory breakdown section
    std::cout << "\n[Memory Breakdown]   A5 — analytical model (all values in CTs)\n";
    print_sep(134);
    std::cout << std::left  << std::setw(24) << "Backend"
              << std::right
              << std::setw(14) << "layout_cts"
              << std::setw(14) << "work_cts"
              << std::setw(16) << "mat_cts"
              << std::setw(16) << "resident_cts"
              << std::setw(14) << "resident_MB"
              << "\n";
    print_sep(134);
    for (const auto& r : R) {
        const auto& m = r.mem_stats;
        std::cout << std::left  << std::setw(24) << r.name
                  << std::right
                  << std::setw(14) << m.layout_cts
                  << std::setw(14) << m.work_cts
                  << std::setw(16) << m.materialized_cts
                  << std::setw(16) << m.resident_cts
                  << std::setw(14) << std::fixed << std::setprecision(1)
                  << (m.resident_cts * ct_mb)
                  << "\n";
    }
    print_sep(134);

    std::cout << "\n[Decision Gate]\n";
    if (R.size() >= 3) {
        double sp = naive_cmp / (double)std::max((int64_t)1, R[2].ops.total_cmp());
        std::cout << "  CMP reduction (Naive vs HELIOS):  "
                  << std::fixed << std::setprecision(1) << sp << "x\n";
        std::cout << "  Naive extrap:    " << [&](){
            std::ostringstream s;
            s << std::fixed << std::setprecision(0) << R[0].wall_extrap_s
              << "s  (~" << std::setprecision(2) << R[0].wall_extrap_s/3600 << "h)";
            return s.str(); }() << "\n";
        if (R[2].ran_actual)
            std::cout << "  HELIOS actual:   "
                      << std::fixed << std::setprecision(2) << R[2].wall_actual_s << "s\n";
        if (R.size() >= 4 && R[2].ran_actual && R[3].ran_actual) {
            std::cout << "  Backend 3 peak:  " << R[2].peak_cts << " CTs  ("
                      << std::fixed << std::setprecision(1) << R[2].peak_mem_mb << " MB)\n";
            std::cout << "  Backend 4 peak:  " << R[3].peak_cts << " CTs  ("
                      << R[3].peak_mem_mb << " MB)  [O(1) fused model]\n";
        }
        std::cout << "  Verdict: "
                  << (sp >= 20 ? "STRONG (>=20x)" :
                      sp >= 10 ? "GOOD   (>=10x)" :
                      sp >=  5 ? "OK     (>=5x)"  :
                      sp >=  2 ? "MARGINAL (>=2x)" : "WEAK (<2x)")
                  << "\n";
    }
    std::cout << std::string(134, '=') << "\n\n";
}

// ============================================================
//  main
// ============================================================
int main(int argc, char* argv[])
{
    // Flush every write immediately so progress shows through tee/pipes on Linux.
    std::cout.setf(std::ios::unitbuf);

    int  N_B      = 1974;
    int  M_B      = 1028;
    bool quick    = false;
    bool recalib  = false;   // force fresh calibration (ignores cache)
    bool dateband = false;   // A4: use date-band residual instead of synthetic band

    int pos_arg = 0;
    for (int i = 1; i < argc; i++) {
        if      (std::strcmp(argv[i], "quick")    == 0) quick    = true;
        else if (std::strcmp(argv[i], "recalib")  == 0) recalib  = true;
        else if (std::strcmp(argv[i], "dateband") == 0) dateband = true;  // A4
        else if (std::isdigit((unsigned char)argv[i][0])) {
            if      (pos_arg == 0) N_B = std::atoi(argv[i]);
            else if (pos_arg == 1) M_B = std::atoi(argv[i]);
            pos_arg++;
        }
    }
    if (quick) { N_B = SEED_CHECK_N; M_B = SEED_CHECK_M; }

    // A4: select active band parameters based on mode
    int active_offset = dateband ? DATE_OFFSET : OFFSET;
    int active_cmp_lo = dateband ? DATE_CMP_LO : CMP_LO;
    int active_cmp_hi = dateband ? DATE_CMP_HI : CMP_HI;
    int active_delta  = dateband ? DELTA_DAYS  : DELTA;

    std::cout << "\n" << std::string(134, '=') << "\n";
    std::cout << "  HELIOS FHE Performance Gate  n_b=" << N_B << "  m_b=" << M_B
              << "  OFFSET=" << active_offset
              << "  CMP_LO=" << active_cmp_lo
              << "  CMP_HI=" << active_cmp_hi
              << (dateband ? "  [DATE-BAND mode]" : "  [SYNTHETIC mode]")  // A4
              << "\n";
    std::cout << std::string(134, '=') << "\n\n";

    // ---- BFV setup -------------------------------------------------------
    std::cout << "[Setup] Generating BFV keys ...\n";
    auto t_s0 = std::chrono::high_resolution_clock::now();

    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(POLY_DEG);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(POLY_DEG));
    parms.set_plain_modulus(65537);

    SEALContext  context(parms);
    KeyGenerator keygen(context);
    SecretKey    sk = keygen.secret_key();
    PublicKey    pk; keygen.create_public_key(pk);
    RelinKeys    rlk; keygen.create_relin_keys(rlk);
    GaloisKeys   gk;  keygen.create_galois_keys(needed_galois_steps(), gk);

    Encryptor    encryptor(context, pk);
    Evaluator    evaluator(context);
    Comparator   comparator(context);
    BatchEncoder benc(context);
    Decryptor    decryptor(context, sk);

    int total_slots = (int)benc.slot_count();  // N   = 32768
    int row_slots   = total_slots / 2;          // N/2 = 16384

    double setup_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_s0).count();
    std::cout << "  Done (" << std::fixed << std::setprecision(1) << setup_s << "s)"
              << "  total_slots=" << total_slots << "  row_slots=" << row_slots << "\n";

    double ct_mb = 0;
    {
        std::vector<int64_t> v(total_slots, 1);
        Plaintext pt; benc.encode(v, pt);
        Ciphertext cs; encryptor.encrypt(pt, cs);
        size_t sz = ct_serial_bytes(cs);
        ct_mb = sz / 1e6;
        std::cout << "  CT size (serialized): " << sz << " bytes  ("
                  << std::fixed << std::setprecision(2) << ct_mb << " MB)\n\n";
    }

    // ---- Quick mode: 1-seed correctness check, then exit ----------------
    // Full bench skips the seed check entirely — run './bench quick' first.
    // (20-seed check costs ~107 min at 320s/seed; verified once is enough.)
    if (quick) {
        bool seed_ok = multi_seed_check(
            comparator, evaluator, encryptor, decryptor, benc, rlk, gk,
            row_slots, total_slots, 1);
        if (!seed_ok) {
            std::cerr << "ERROR: 1-seed check failed.\n";
            return 1;
        }
        std::cout << "Quick mode: 1-seed check passed. Exiting.\n";
        return 0;
    }

    // ---- Calibration (full bench mode only) ------------------------------
    // Load from cache if available; otherwise measure and save.
    // Run with 'recalib' argument to force a fresh measurement.
    std::cout << "[Calibration]\n";
    PerOpTimes T;
    {
        auto [cache_ok, T_cached] = load_calibration();
        if (cache_ok && !recalib) {
            T = T_cached;
            std::cout << "  Loaded from " << CALIB_CACHE
                      << "  (pass 'recalib' to remeasure)\n";
            print_calib(T);
        } else {
            if (recalib) std::cout << "  Forcing fresh calibration (recalib)\n";
            T = calibrate(comparator, evaluator, encryptor, benc, rlk, gk, total_slots);
            save_calibration(T);
        }
    }
    std::cout << "\n";

    // ---- Tiling preview --------------------------------------------------
    int inner_m_d  = std::min(N_B, M_B);
    int p_per_row_d = row_slots / inner_m_d;
    int p_d         = 2 * p_per_row_d;
    int nb_d        = ceildiv(std::max(N_B, M_B), p_d);
    // Naive CMP = N_B outer elements × ceildiv(M_B, total_slots) chunks × 2 LT
    // This matches count_naive(N_B, M_B, total_slots).total_cmp().
    int naive_cmp_d = 2 * N_B * ceildiv(M_B, total_slots);
    std::cout << "[Tiling — Bucket C]\n"
              << "  inner_m=" << inner_m_d
              << "  p_per_row=" << p_per_row_d
              << "  p=" << p_d
              << "  n_batches=" << nb_d << "\n"
              << "  HELIOS CMP: 2 * " << nb_d << " = " << 2*nb_d << "\n"
              << "  Naive   CMP: 2 * " << N_B * ceildiv(M_B, total_slots)
              <<                    " = " << naive_cmp_d << "\n"
              << "  Expected reduction: ~"
              << std::fixed << std::setprecision(1)
              << (double)naive_cmp_d / (double)(2*nb_d) << "x\n\n";

    // ---- Data generation -------------------------------------------------
    // A4: date-band mode uses deterministic day-offset values instead of random.
    std::vector<int64_t> A(N_B), B(M_B);
    if (dateband) {
        // A4: synthetic but realistic date distribution (days-since-epoch)
        for (int i = 0; i < N_B; i++)
            A[i] = BASE_DATE + (int64_t)(i * 7)  % DATE_RANGE;
        for (int j = 0; j < M_B; j++)
            B[j] = BASE_DATE + (int64_t)(j * 11) % DATE_RANGE;
        std::cout << "[Data] DATE-BAND mode  n_b=" << N_B << "  m_b=" << M_B
                  << "  BASE_DATE=" << BASE_DATE
                  << "  DATE_RANGE=" << DATE_RANGE
                  << "  DELTA_DAYS=" << DELTA_DAYS << "\n";
    } else {
        std::mt19937_64 rng(42);
        std::uniform_int_distribution<int64_t> dist(0, VALUE_RANGE - 1);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);
    }

    int64_t gt_count = plaintext_count(A, B, active_delta);

    // A3: synthetic payloads — outer_payload[k] = (k % PAYLOAD_BOUND) + 1
    //     Generate for both A and B; helios_tiling_fhe selects the right one
    //     based on its internal swap_orient decision.
    std::vector<int64_t> a_payload(N_B), b_payload(M_B);
    for (int i = 0; i < N_B; i++) a_payload[i] = (int64_t)(i % PAYLOAD_BOUND) + 1;
    for (int j = 0; j < M_B; j++) b_payload[j] = (int64_t)(j % PAYLOAD_BOUND) + 1;

    // A3: ground-truth SUM — replicate helios_tiling_fhe's swap_orient decision
    //     so we compute SUM of the same payload set as the FHE circuit uses.
    auto orient_cmp = [&](int outer, int inner) -> int {
        if (inner > row_slots) return INT_MAX;
        return ceildiv(outer, 2 * (row_slots / inner));
    };
    bool will_swap = (orient_cmp(M_B, N_B) < orient_cmp(N_B, M_B));
    const std::vector<int64_t>& OUTER_GT    = will_swap ? B : A;
    const std::vector<int64_t>& INNER_GT    = will_swap ? A : B;
    const std::vector<int64_t>& OUTER_PAY_GT = will_swap ? b_payload : a_payload;
    int64_t gt_sum = plaintext_sum_payload(OUTER_GT, INNER_GT, active_delta, OUTER_PAY_GT);

    if (!dateband) {
        std::cout << "[Data] n_b=" << N_B << "  m_b=" << M_B
                  << "  gt_count=" << gt_count
                  << "  gt_sum=" << gt_sum
                  << "  (" << std::fixed << std::setprecision(1)
                  << 100.0 * gt_count / ((int64_t)N_B * M_B) << "% match)\n\n";
    } else {
        std::cout << "  gt_count=" << gt_count
                  << "  gt_sum=" << gt_sum
                  << "  (" << std::fixed << std::setprecision(1)
                  << 100.0 * gt_count / ((int64_t)N_B * M_B) << "% match)\n\n";
    }

    // A5: analytical memory breakdown for each backend
    //
    // Formulas (model-based; see paper §Memory):
    //   Naive:        layout = N_B + ceil(M_B/S),  work = 6, mat = 0
    //   FixedOrient:  layout = min + ceil(max/S),  work = 6, mat = 0
    //   HELIOS-Fused: layout = 1 + n_batches,      work = 6, mat = 0
    //   HELIOS-Tile:  layout = 1 + n_batches,      work = 6 + n_batches, mat = n_batches
    //
    // resident = layout + work for all backends.
    MemStats mem_naive, mem_fixed, mem_tile, mem_fused;
    {
        // Naive: outer=N_B (no orientation swap), inner=M_B
        mem_naive.layout_cts       = N_B + ceildiv(M_B, total_slots);
        mem_naive.work_cts         = 6;
        mem_naive.materialized_cts = 0;
        mem_naive.resident_cts     = mem_naive.layout_cts + mem_naive.work_cts;

        // FixedOrient: outer=min(N_B,M_B), inner=max(N_B,M_B)
        mem_fixed.layout_cts       = std::min(N_B,M_B) + ceildiv(std::max(N_B,M_B), total_slots);
        mem_fixed.work_cts         = 6;
        mem_fixed.materialized_cts = 0;
        mem_fixed.resident_cts     = mem_fixed.layout_cts + mem_fixed.work_cts;

        // HELIOS shared geometry (matches helios_tiling_fhe's swap_orient)
        int h_outer_n = will_swap ? M_B : N_B;
        int h_inner_m = will_swap ? N_B : M_B;
        int h_p       = 2 * (row_slots / h_inner_m);
        int h_nb      = ceildiv(h_outer_n, h_p);

        // HELIOS-Tile: materialized tile grid = n_batches extra work CTs
        mem_tile.layout_cts        = 1 + h_nb;
        mem_tile.work_cts          = 6 + h_nb;
        mem_tile.materialized_cts  = h_nb;
        mem_tile.resident_cts      = mem_tile.layout_cts + mem_tile.work_cts;

        // HELIOS-Fused: O(1) scratch (running accumulator only)
        mem_fused.layout_cts       = 1 + h_nb;
        mem_fused.work_cts         = 6;
        mem_fused.materialized_cts = 0;
        mem_fused.resident_cts     = mem_fused.layout_cts + mem_fused.work_cts;
    }

    std::vector<BenchResult> results;
    static const int64_t PM = 65537;  // plaintext modulus

    // Backend 1: Naive (analytical)
    {
        std::cout << "[Backend 1] Naive (analytical) ...\n";
        OpStats ops = count_naive(N_B, M_B, total_slots);
        double  ext = extrapolate_s(ops, T);
        int64_t pk  = (int64_t)mem_naive.resident_cts;
        std::cout << "  CMP=" << ops.total_cmp() << "  ROT=" << ops.he_rot
                  << "  extrap=" << std::fixed << std::setprecision(0)
                  << ext << "s (~" << ext/3600 << "h)\n\n";
        results.push_back({"1-Naive", ops, 0, ext, pk, pk*ct_mb, true, false,
                            0, gt_sum % PM, false, mem_naive});
    }

    // Backend 2: Fixed orientation (analytical)
    {
        std::cout << "[Backend 2] FixedOrient (analytical) ...\n";
        OpStats ops = count_fixed(N_B, M_B, total_slots);
        double  ext = extrapolate_s(ops, T);
        int64_t pk  = (int64_t)mem_fixed.resident_cts;
        std::cout << "  CMP=" << ops.total_cmp() << "  ROT=" << ops.he_rot
                  << "  extrap=" << std::fixed << std::setprecision(0)
                  << ext << "s (~" << ext/3600 << "h)\n\n";
        results.push_back({"2-FixedOrient", ops, 0, ext, pk, pk*ct_mb, true, false,
                            0, gt_sum % PM, false, mem_fixed});
    }

    // Backend 3: HELIOS tile (non-fused memory model; fused execution)
    {
        std::cout << "[Backend 3] HELIOS-Tile (actual FHE; non-fused peak_cts model) ...\n";
        OpStats ops; ops.reset();
        double wall = 0;
        // A3/A4: pass payloads and active band params
        FheResult fhe3 = helios_tiling_fhe(
            N_B, M_B, A, B,
            a_payload, b_payload,
            comparator, evaluator, encryptor, decryptor,
            benc, rlk, gk, row_slots, total_slots, ops, /*fused=*/false, wall,
            active_offset, active_cmp_lo, active_cmp_hi);
        bool ok3      = (fhe3.count == gt_count % PM);
        bool sum_ok3  = (fhe3.sum   == gt_sum   % PM);  // A3
        int64_t pk    = (int64_t)mem_tile.resident_cts;
        double ext    = extrapolate_s(ops, T);
        std::cout << "  CMP=" << ops.total_cmp() << "  ROT=" << ops.he_rot
                  << "  wall=" << std::fixed << std::setprecision(1) << wall << "s"
                  << "  fhe_count=" << fhe3.count
                  << " gt_count=" << gt_count << " (mod=" << gt_count % PM << ")"
                  << "  " << (ok3 ? "PASS" : "FAIL")
                  << "  fhe_sum=" << fhe3.sum                          // A3
                  << " gt_sum(mod)=" << gt_sum % PM                   // A3
                  << "  sum:" << (sum_ok3 ? "PASS" : "FAIL")          // A3
                  << "\n\n";
        results.push_back({"3-HELIOS-Tile", ops, wall, ext, pk, pk*ct_mb, ok3, true,
                            fhe3.sum, gt_sum % PM, sum_ok3, mem_tile});
    }

    // Backend 4: HELIOS fused (fused memory model; same execution)
    {
        std::cout << "[Backend 4] HELIOS-Fused (actual FHE; O(1) peak_cts model) ...\n";
        OpStats ops; ops.reset();
        double wall = 0;
        // A3/A4: pass payloads and active band params
        FheResult fhe4 = helios_tiling_fhe(
            N_B, M_B, A, B,
            a_payload, b_payload,
            comparator, evaluator, encryptor, decryptor,
            benc, rlk, gk, row_slots, total_slots, ops, /*fused=*/true, wall,
            active_offset, active_cmp_lo, active_cmp_hi);
        bool ok4      = (fhe4.count == gt_count % PM);
        bool sum_ok4  = (fhe4.sum   == gt_sum   % PM);  // A3
        int64_t pk    = (int64_t)mem_fused.resident_cts;
        double ext    = extrapolate_s(ops, T);
        std::cout << "  CMP=" << ops.total_cmp() << "  ROT=" << ops.he_rot
                  << "  wall=" << std::fixed << std::setprecision(1) << wall << "s"
                  << "  fhe_count=" << fhe4.count
                  << " gt_count=" << gt_count << " (mod=" << gt_count % PM << ")"
                  << "  " << (ok4 ? "PASS" : "FAIL")
                  << "  fhe_sum=" << fhe4.sum                          // A3
                  << " gt_sum(mod)=" << gt_sum % PM                   // A3
                  << "  sum:" << (sum_ok4 ? "PASS" : "FAIL")          // A3
                  << "\n\n";
        results.push_back({"4-HELIOS-Fused", ops, wall, ext, pk, pk*ct_mb, ok4, true,
                            fhe4.sum, gt_sum % PM, sum_ok4, mem_fused});
    }

    print_results(results, N_B, M_B, gt_count, gt_sum,
                  row_slots, total_slots, ct_mb, dateband, active_delta);
    write_csv(results, N_B, M_B, row_slots, total_slots, ct_mb,
              gt_count, gt_sum, dateband);
    return 0;
}
