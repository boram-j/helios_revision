/**
 * helios_bucket_bench.cpp  v5
 *
 * Benchmarks four compiler backends for encrypted band-join |A.x - B.y| <= Delta.
 * Output semantic: global COUNT(*) — one encrypted scalar.
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
 *   ./bin/helios_bucket_bench quick          # n=20, m=10, 20-seed check
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

// ============================================================
//  fhe_between_shifted
//
//  Inputs:
//    ct_shifted_diff  = (A[i] - B[j]) + OFFSET  per slot
//                     = OUTER+OFFSET - INNER ∈ [1, 127] for valid slots
//
//  Checks: CMP_LO < ct_shifted_diff < CMP_HI
//    i.e., 47 < diff < 81
//    i.e., 48 ≤ diff ≤ 80
//    i.e., |A[i] - B[j]| ≤ DELTA
//
//  Both bounds (47, 81) are positive — no signed-mod ambiguity.
//
//  API note: comp.isLessThan(eval, rk, A, B) returns vector<Ciphertext>;
//  [0] is the boolean indicator.  Adjust if API returns plain Ciphertext.
// ============================================================
static Ciphertext fhe_between_shifted(
    Ciphertext&       ct_shifted_diff,
    Comparator&       comp,
    Evaluator&        eval,
    const RelinKeys&  rk,
    BatchEncoder&     benc,
    int               total_slots,
    OpStats&          ops)
{
    std::vector<int64_t> lo_vec(total_slots, (int64_t)CMP_LO);
    std::vector<int64_t> hi_vec(total_slots, (int64_t)CMP_HI);
    Plaintext pt_lo, pt_hi;
    benc.encode(lo_vec, pt_lo);
    benc.encode(hi_vec, pt_hi);

    // mask_lo: CMP_LO < ct_shifted_diff  →  isLessThan(CMP_LO, ct)
    auto res_lo = comp.isLessThan(eval, rk, pt_lo, ct_shifted_diff);
    ops.he_ltp++;

    // mask_hi: ct_shifted_diff < CMP_HI  →  isLessThan(ct, CMP_HI)
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
//  HELIOS tiling — actual FHE, two-row packing, global COUNT
//
//  Shifted encoding:
//    outer_vec[valid k,j] = OUTER[batch_start+k] + OFFSET
//    outer_vec[pad k,j]   = PAD_OUTER = 0
//    inner_vec[k,j]       = INNER[j]
//    ct_diff[slot]        = outer[slot] - inner[slot]
//                         = (A+OFFSET-B) for valid  →  ∈ [1,127]
//                         = (0 - B)      for pad    →  ≡ large value >> CMP_HI
//
//  Comparison: CMP_LO < ct_diff < CMP_HI  (bounds 47,81 — both positive)
//
//  Both backends (fused=false and fused=true) run the same fused-accumulation
//  code path.  Difference is only in the reported analytical peak_cts.
// ============================================================
static int64_t helios_tiling_fhe(
    int n_b, int m_b,
    const std::vector<int64_t>& A,
    const std::vector<int64_t>& B,
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
    double&           wall_s)
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

    if (inner_m > row_slots) {
        std::cerr << "  ERROR: inner_m=" << inner_m << " > row_slots=" << row_slots
                  << "; chunked inner not implemented.\n";
        return -1;
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

    for (int batch = 0; batch < n_batches; batch++) {
        int batch_start = batch * p;
        int batch_end   = std::min(batch_start + p, outer_n);
        int this_p      = batch_end - batch_start;
        int this_p0     = std::min(this_p, p_per_row);
        int this_p1     = std::max(0, this_p - p_per_row);

        // Outer CT:  valid positions = OUTER[batch_start+k] + OFFSET
        //            pad  positions  = PAD_OUTER = 0
        // Padding diff = 0 - INNER[j] = -(INNER[j]) mod p ≈ 65537-INNER[j] >> CMP_HI
        std::vector<int64_t> outer_vec(total_slots, (int64_t)PAD_OUTER);
        for (int k = 0; k < this_p0; k++)
            for (int j = 0; j < inner_m; j++)
                outer_vec[k * inner_m + j] = OUTER[batch_start + k] + OFFSET;
        for (int k = 0; k < this_p1; k++)
            for (int j = 0; j < inner_m; j++)
                outer_vec[row_slots + k * inner_m + j] =
                    OUTER[batch_start + p_per_row + k] + OFFSET;

        Plaintext  pt_outer; benc.encode(outer_vec, pt_outer);
        Ciphertext ct_outer; encryptor.encrypt(pt_outer, ct_outer);

        // ct_diff = (OUTER+OFFSET - INNER)  for valid slots
        //         = (0 - INNER)             for pad slots
        Ciphertext ct_diff;
        eval.sub(ct_outer, ct_inner, ct_diff);
        ops.he_sub_ct++;

        // Slot-wise BETWEEN with shifted bounds (no negative plaintexts)
        Ciphertext ct_mask = fhe_between_shifted(
            ct_diff, comp, eval, rk, benc, total_slots, ops);

        if (!acc_ready) {
            ct_acc    = ct_mask;
            acc_ready = true;
        } else {
            eval.add_inplace(ct_acc, ct_mask);
            ops.he_add_ct++;
        }
    }

    // Reduce both rows to global scalar in slot[0]
    ct_acc = rotation_sum_both_rows(eval, gk, ct_acc, row_slots, ops);

    wall_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // Decrypt and return
    Plaintext pt_result; decryptor.decrypt(ct_acc, pt_result);
    std::vector<int64_t> decoded; benc.decode(pt_result, decoded);
    return decoded[0];
}

// ============================================================
//  Multi-seed correctness check
//  Runs SEED_CHECK_K quick FHE trials (n=8, m=6) with different seeds.
//  Returns true only if all pass.  Catches comparator boundary bugs early.
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
        int64_t fhe = helios_tiling_fhe(
            SEED_CHECK_N, SEED_CHECK_M, A, B,
            comp, eval, encryptor, decryptor, benc, rk, gk,
            row_slots, total_slots, ops, /*fused=*/true, wall);
        if (fhe != gt) {
            std::cout << " FAIL (gt=" << gt << " fhe=" << fhe << ")\n";
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
};

static void print_sep(int w = 134) { std::cout << std::string(w, '-') << "\n"; }

static void write_csv(
    const std::vector<BenchResult>& R,
    int n_b, int m_b,
    int row_slots, int total_slots,
    double ct_mb,
    int64_t gt_count)
{
    std::string path = "./helios_bench_results.csv";
    std::ofstream f(path, std::ios::app);
    if (!f.is_open()) { std::cerr << "  Warning: could not write CSV\n"; return; }
    // Header on first write (check if file empty)
    f.seekp(0, std::ios::end);
    if (f.tellp() == 0)
        f << "n_b,m_b,total_slots,row_slots,ct_mb,gt_count,"
          << "backend,cmp_tot,he_rot,he_mul_ct,he_mul_pt,he_add_ct,he_sub_ct,"
          << "wall_actual_s,wall_extrap_s,peak_cts,peak_mem_mb,correct\n";
    for (const auto& r : R) {
        f << n_b << "," << m_b << "," << total_slots << "," << row_slots << ","
          << std::fixed << std::setprecision(2) << ct_mb << "," << gt_count << ","
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
          << (r.ran_actual ? (r.correct ? "PASS" : "FAIL") : "analytic")
          << "\n";
    }
    std::cout << "  CSV appended to " << path << "\n\n";
}

static void print_results(
    const std::vector<BenchResult>& R,
    int n_b, int m_b,
    int64_t gt_count,
    int row_slots, int total_slots,
    double ct_mb)
{
    double naive_cmp = R.empty() ? 1.0 :
        (double)std::max((int64_t)1, R[0].ops.total_cmp());

    std::cout << "\n" << std::string(134, '=') << "\n";
    std::cout << "  HELIOS Bench  n_b=" << n_b << "  m_b=" << m_b
              << "  row_slots=" << row_slots << "  total_slots=" << total_slots
              << "  Delta=" << DELTA << "  OFFSET=" << OFFSET
              << "  gt=" << gt_count
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

    int  N_B   = 1974;
    int  M_B   = 1028;
    bool quick = false;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "quick") == 0) {
            quick = true;
        } else if (i == 1 && std::isdigit((unsigned char)argv[i][0])) {
            N_B = std::atoi(argv[i]);
        } else if (i == 2 && std::isdigit((unsigned char)argv[i][0])) {
            M_B = std::atoi(argv[i]);
        }
    }
    if (quick) { N_B = SEED_CHECK_N; M_B = SEED_CHECK_M; }

    std::cout << "\n" << std::string(134, '=') << "\n";
    std::cout << "  HELIOS FHE Performance Gate  n_b=" << N_B << "  m_b=" << M_B
              << "  OFFSET=" << OFFSET << "  CMP_LO=" << CMP_LO << "  CMP_HI=" << CMP_HI << "\n";
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

    // ---- Correctness check (1 seed in quick mode, 20 in full mode) ------
    // Run BEFORE calibration so 'quick' exits without paying calibration cost.
    int n_seeds = quick ? 1 : SEED_CHECK_K;
    bool seed_ok = multi_seed_check(
        comparator, evaluator, encryptor, decryptor, benc, rlk, gk,
        row_slots, total_slots, n_seeds);
    if (!seed_ok) {
        std::cerr << "ERROR: correctness check failed. Fix comparator before full run.\n";
        return 1;
    }

    if (quick) {
        std::cout << "Quick mode: 1-seed check passed. Exiting.\n";
        return 0;
    }

    // ---- Calibration (full bench mode only) ------------------------------
    std::cout << "[Calibration]\n";
    PerOpTimes T = calibrate(comparator, evaluator, encryptor, benc, rlk, gk, total_slots);
    std::cout << "\n";

    // ---- Tiling preview --------------------------------------------------
    int inner_m_d  = std::min(N_B, M_B);
    int p_per_row_d = row_slots / inner_m_d;
    int p_d         = 2 * p_per_row_d;
    int nb_d        = ceildiv(std::max(N_B, M_B), p_d);
    std::cout << "[Tiling — Bucket C]\n"
              << "  inner_m=" << inner_m_d
              << "  p_per_row=" << p_per_row_d
              << "  p=" << p_d
              << "  n_batches=" << nb_d << "\n"
              << "  HELIOS CMP: 2 * " << nb_d << " = " << 2*nb_d << "\n"
              << "  Naive   CMP: 2 * " << std::max(N_B,M_B) << " = " << 2*std::max(N_B,M_B) << "\n"
              << "  Expected reduction: ~"
              << std::fixed << std::setprecision(1)
              << (double)(2*std::max(N_B,M_B)) / (double)(2*nb_d) << "x\n\n";

    // ---- Synthetic data --------------------------------------------------
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> dist(0, VALUE_RANGE - 1);
    std::vector<int64_t> A(N_B), B(M_B);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    int64_t gt_count = plaintext_count(A, B, DELTA);
    std::cout << "[Data] n_b=" << N_B << "  m_b=" << M_B
              << "  gt_count=" << gt_count
              << "  (" << std::fixed << std::setprecision(1)
              << 100.0 * gt_count / ((int64_t)N_B * M_B) << "% match)\n\n";

    std::vector<BenchResult> results;

    // Backend 1: Naive (analytical)
    {
        std::cout << "[Backend 1] Naive (analytical) ...\n";
        OpStats ops = count_naive(N_B, M_B, total_slots);
        double  ext = extrapolate_s(ops, T);
        int64_t pk  = 1 + ceildiv(M_B, total_slots) + 4;
        std::cout << "  CMP=" << ops.total_cmp() << "  ROT=" << ops.he_rot
                  << "  extrap=" << std::fixed << std::setprecision(0)
                  << ext << "s (~" << ext/3600 << "h)\n\n";
        results.push_back({"1-Naive", ops, 0, ext, pk, pk*ct_mb, true, false});
    }

    // Backend 2: Fixed orientation (analytical)
    {
        std::cout << "[Backend 2] FixedOrient (analytical) ...\n";
        OpStats ops = count_fixed(N_B, M_B, total_slots);
        double  ext = extrapolate_s(ops, T);
        int64_t pk  = 1 + ceildiv(std::max(N_B, M_B), total_slots) + 4;
        std::cout << "  CMP=" << ops.total_cmp() << "  ROT=" << ops.he_rot
                  << "  extrap=" << std::fixed << std::setprecision(0)
                  << ext << "s (~" << ext/3600 << "h)\n\n";
        results.push_back({"2-FixedOrient", ops, 0, ext, pk, pk*ct_mb, true, false});
    }

    // Backend 3: HELIOS tile (non-fused memory model; fused execution)
    {
        std::cout << "[Backend 3] HELIOS-Tile (actual FHE; non-fused peak_cts model) ...\n";
        OpStats ops; ops.reset();
        double wall = 0;
        int64_t fhe_count = helios_tiling_fhe(
            N_B, M_B, A, B, comparator, evaluator, encryptor, decryptor,
            benc, rlk, gk, row_slots, total_slots, ops, /*fused=*/false, wall);
        bool ok = (fhe_count == gt_count);
        int  p3 = 2 * (row_slots / std::min(N_B, M_B));
        int  nb3= ceildiv(std::max(N_B, M_B), p3);
        int64_t pk = (int64_t)nb3 + 2 + 4;
        double ext = extrapolate_s(ops, T);
        std::cout << "  CMP=" << ops.total_cmp() << "  ROT=" << ops.he_rot
                  << "  wall=" << std::fixed << std::setprecision(1) << wall << "s"
                  << "  fhe=" << fhe_count << " gt=" << gt_count
                  << "  " << (ok ? "PASS" : "FAIL") << "\n\n";
        results.push_back({"3-HELIOS-Tile", ops, wall, ext, pk, pk*ct_mb, ok, true});
    }

    // Backend 4: HELIOS fused (fused memory model; same execution)
    {
        std::cout << "[Backend 4] HELIOS-Fused (actual FHE; O(1) peak_cts model) ...\n";
        OpStats ops; ops.reset();
        double wall = 0;
        int64_t fhe_count = helios_tiling_fhe(
            N_B, M_B, A, B, comparator, evaluator, encryptor, decryptor,
            benc, rlk, gk, row_slots, total_slots, ops, /*fused=*/true, wall);
        bool ok = (fhe_count == gt_count);
        int64_t pk = 2 + 4;
        double ext = extrapolate_s(ops, T);
        std::cout << "  CMP=" << ops.total_cmp() << "  ROT=" << ops.he_rot
                  << "  wall=" << std::fixed << std::setprecision(1) << wall << "s"
                  << "  fhe=" << fhe_count << " gt=" << gt_count
                  << "  " << (ok ? "PASS" : "FAIL") << "\n\n";
        results.push_back({"4-HELIOS-Fused", ops, wall, ext, pk, pk*ct_mb, ok, true});
    }

    print_results(results, N_B, M_B, gt_count, row_slots, total_slots, ct_mb);
    write_csv(results, N_B, M_B, row_slots, total_slots, ct_mb, gt_count);
    return 0;
}
