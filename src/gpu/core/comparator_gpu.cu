// comparator_gpu.cu — GPU implementation of GpuComparator.
//
// Ports nshedb::core::Comparator from SEAL to Phantom's BFV free-function API.
//
// All polynomial-coefficient arithmetic (initCoeff, initPolyParam) is performed
// on the host (CPU) during construction — the code is identical to the CPU
// version since it is pure integer math over Z_p.
//
// The GPU polynomial evaluation mirrors the CPU Paterson-Stockmeyer strategy
// but calls Phantom free functions (namespace phantom) instead of SEAL Evaluator
// methods.
//
// HECache integration (optional, runtime-gated):
//   When hecache::HECache::is_initialized(), getPower() registers computed baby-
//   and giant-step powers with HECache in DISCARD mode.  DISCARD means HECache
//   can evict them under VRAM pressure without a write-back (they are
//   recomputable from the input ciphertext).  The HECache key per power is:
//       "pow_<thread_local_prefix>_<exponent>"
//   where the prefix is unique per evalPoly() invocation (set via a thread-
//   local string).  evalPoly() cleans up its prefix via erase_prefix() after
//   the polynomial evaluation completes.

#include "nshedb_gpu/core/comparator_gpu.cuh"

#include <cmath>
#include <stdexcept>
#include <atomic>
#include <string>

// HECache integration — pulled in only when the header is reachable.
// The integration is entirely runtime-gated via HECache::is_initialized(),
// so the binary works identically with or without HECache present at startup.
#include "hecache/hecache.h"

// Phantom free functions are in namespace phantom
using namespace phantom;

namespace {

// ── HECache power-table helpers ──────────────────────────────────────────────
//
// Each evalPoly() invocation generates a unique prefix string so that HECache
// keys do not collide across concurrent or sequential evaluations.  The prefix
// is stored in a thread-local variable so helper functions (getPower, etc.) can
// access it without needing an extra parameter.

static std::atomic<uint64_t> s_eval_call_counter{0};

// Thread-local prefix for the current evalPoly() invocation.
// Set by evalPoly() at the start of each call; cleared (and HECache cleaned)
// at the end of each call via erase_prefix().
thread_local std::string tl_power_prefix;

} // anonymous namespace

namespace nshedb_gpu {
namespace core {

// ============================================================
// Constructor — reads p from context, computes poly on CPU
// ============================================================

GpuComparator::GpuComparator(const PhantomContext    &ctx,
                              const PhantomBatchEncoder &encoder)
    : ctx_(ctx), encoder_(encoder)
{
    // Read plain_modulus from the key-level context data (same as CPU)
    p_ = static_cast<int64_t>(
        ctx_.key_context_data().parms().plain_modulus().value());

    initCoeff();
    initPolyParam();
}

// ============================================================
// makeCoefPlain — encode a scalar coefficient as a constant plaintext
// ============================================================

PhantomPlaintext GpuComparator::makeCoefPlain(int64_t coef) const
{
    // Lift signed coefficient into Z_p representative as uint64_t
    uint64_t uval = (coef < 0)
                    ? static_cast<uint64_t>(p_ + coef)
                    : static_cast<uint64_t>(coef);

    size_t n = encoder_.slot_count();
    std::vector<uint64_t> vec(n, uval);
    return encoder_.encode(ctx_, vec);
}

// ============================================================
// initCoeff — compute comparison polynomial coefficients over Z_p
// (identical to CPU Comparator::initCoeff)
// ============================================================

void GpuComparator::initCoeff()
{
    poly_.resize(static_cast<size_t>((p_ - 1) / 2), 0);
    int64_t coeff;
    for (int64_t idx = 1; idx < p_ - 1; idx += 2) {
        coeff = 1;
        for (int64_t a = 2; a <= ((p_ - 1) >> 1); a++) {
            int64_t power = utils::powMod(a, p_ - 1 - idx, p_);
            coeff += power;
            if (coeff >= p_) coeff %= p_;
        }
        poly_[static_cast<size_t>((idx - 1) >> 1)] = coeff;
    }
}

// ============================================================
// initPolyParam — set Paterson-Stockmeyer parameters
// (identical to CPU Comparator::initPolyParam)
// ============================================================

void GpuComparator::initPolyParam()
{
    d_comp_ = utils::getDegree(poly_);
    kk_     = static_cast<int64_t>(std::sqrt(d_comp_ / 2.0));
    k_      = 1LL << utils::nextPowerOf2(kk_);

    if ((k_ == 16 && d_comp_ > 167) || (k_ > 16 && k_ > (1.44 * kk_)))
        k_ /= 2;

    m_ = (d_comp_ + k_ - 1) / k_;

    top_coef_  = poly_[d_comp_];
    topInv_    = p_ - 8;
    extra_coef_ = 0;
    divisible_ = (m_ * k_) == d_comp_;

    if (m_ != (1LL << utils::nextPowerOf2(m_))) {
        if (!divisible_) {
            top_coef_   = 1;
            topInv_     = 1;
            extra_coef_ = (top_coef_ - poly_[m_ * k_]) % p_;
            poly_[static_cast<size_t>(m_ * k_)] = 1;
        }
        if (top_coef_ != 1) {
            for (size_t i = 0; i < poly_.size(); i++)
                poly_[i] *= topInv_;
            for (int64_t i = 0; i <= m_ * k_; i++)
                poly_[static_cast<size_t>(i)] %= p_;
            poly_.resize(static_cast<size_t>(utils::getDegree(poly_) + 1));
        }
    }

    int64_t top_deg = (p_ - 1) >> 1;
    baby_idx_  = top_deg % k_;
    giant_idx_ = top_deg / k_;
    if (baby_idx_ == 0) {
        baby_idx_ = k_;
        giant_idx_--;
    }
}

// ============================================================
// Public: isLessThan (ct - ct)
// ============================================================

std::vector<PhantomCiphertext>
GpuComparator::isLessThan(const PhantomRelinKey &rk,
                           PhantomCiphertext a,
                           const PhantomCiphertext &b) const
{
    // cipher_Z = a - b
    PhantomCiphertext cipher_Z = sub(ctx_, a, b);
    return evalPoly(rk, cipher_Z);
}

// ============================================================
// Public: isLessThan (ct - plain)
// ============================================================

std::vector<PhantomCiphertext>
GpuComparator::isLessThan(const PhantomRelinKey &rk,
                           PhantomCiphertext a,
                           const PhantomPlaintext &b_plain) const
{
    PhantomCiphertext cipher_Z = sub_plain(ctx_, a, b_plain);
    return evalPoly(rk, cipher_Z);
}

// ============================================================
// Public: isEqual (ct - ct)
// Computes 1 - (a-b)^{p-1} = indicator that a==b in Z_p.
// ============================================================

PhantomCiphertext
GpuComparator::isEqual(const PhantomRelinKey &rk,
                        PhantomCiphertext a,
                        const PhantomCiphertext &b) const
{
    PhantomCiphertext res = sub(ctx_, a, b);

    // Raise to (p-1)-th power via repeated squaring: floor(log2(p-1)) squarings
    int sq_rounds = static_cast<int>(std::log2(static_cast<double>(p_ - 1)));
    for (int i = 0; i < sq_rounds; i++) {
        multiply_inplace(ctx_, res, res);
        relinearize_inplace(ctx_, res, rk);
    }

    // res = 1 - res
    negate_inplace(ctx_, res);
    PhantomPlaintext one = makeCoefPlain(1);
    add_plain_inplace(ctx_, res, one);
    return res;
}

// ============================================================
// Private: getPower — memoised exponentiation table lookup
//
// x[e-1]  stores  (x^2)^e  (i.e. the e-th entry for baby/giant steps).
// computed[e-1] tracks whether x[e-1] has been filled in LOCAL memory.
//
// HECache integration (runtime-gated, DISCARD mode):
//   When HECache is initialised and tl_power_prefix is non-empty (set by
//   evalPoly()), each computed power is also inserted into HECache with
//   ItemCacheMode::DISCARD.  HECache may evict these copies under VRAM
//   pressure without any write-back cost (they are recomputable).
//
//   Before computing, we consult HECache: if the power was previously
//   computed and NOT yet evicted, we restore it from there (avoiding a
//   redundant multiply + relinearize).  If it WAS evicted (DISCARD →
//   freed), get_ct() returns nullptr and we recompute normally.
//
//   The local x/computed vectors are the primary cache; HECache is a
//   secondary, budget-aware copy that the system may discard at will.
// ============================================================

PhantomCiphertext
GpuComparator::getPower(const PhantomRelinKey        &rk,
                         std::vector<PhantomCiphertext> &x,
                         std::vector<bool>              &computed,
                         int64_t e) const
{
    const size_t idx = static_cast<size_t>(e - 1);

    if (!computed[idx]) {
        // ── HECache secondary lookup ─────────────────────────────────────────
        // Only attempt if HECache is live AND this evalPoly() set a prefix.
        //
        // Key includes the address of the power vector (x.data()) to
        // distinguish baby-step from giant-step tables: both use exponents
        // starting at 1, and without a table discriminator they would alias
        // (babyStep[e=k] and giantStep[e=k] would share the same HECache
        // slot).  x.data() is unique per vector, since babyStep and
        // giantStep are separate heap allocations within the same evalPoly().
        if (hecache::HECache::is_initialized() && !tl_power_prefix.empty()) {
            const std::string key = tl_power_prefix
                + "_t" + std::to_string(reinterpret_cast<uintptr_t>(x.data()))
                + "_e" + std::to_string(e);
            auto ct_ptr = hecache::HECache::instance().get_ct(key);
            if (ct_ptr) {
                // Restore from HECache (GPU-resident shared_ptr copy)
                x[idx]       = *ct_ptr;
                computed[idx] = true;
                return x[idx];
            }
        }

        // ── Compute ──────────────────────────────────────────────────────────
        int64_t k   = 1LL << (utils::nextPowerOf2(e) - 1);
        // Recursively ensure both sub-powers are computed
        PhantomCiphertext lo = getPower(rk, x, computed, e - k);
        PhantomCiphertext hi = getPower(rk, x, computed, k);
        x[idx] = multiply(ctx_, lo, hi);
        relinearize_inplace(ctx_, x[idx], rk);
        computed[idx] = true;

        // ── HECache secondary insertion (DISCARD mode) ────────────────────────
        // Insert a GPU clone so HECache can account for and optionally evict it.
        // DISCARD means eviction = free; no D2H write-back needed (recomputable).
        // Key is scoped to (evalPoly invocation, table address, exponent).
        if (hecache::HECache::is_initialized() && !tl_power_prefix.empty()) {
            const std::string key = tl_power_prefix
                + "_t" + std::to_string(reinterpret_cast<uintptr_t>(x.data()))
                + "_e" + std::to_string(e);
            // clone_ciphertext copies GPU memory; HECache owns the copy.
            hecache::HECache::instance().insert_ct(
                key,
                hecache::clone_ciphertext(x[idx]),
                hecache::ItemCacheMode::DISCARD);
        }
    }
    return x[idx];
}

// ============================================================
// Private: evalPoly — main polynomial evaluation (Paterson-Stockmeyer)
//
// Mirrors CPU Comparator::evalPoly.
// Input  x : cipher of (a - b)
// Output : {lt_result, eq_result}
// ============================================================

std::vector<PhantomCiphertext>
GpuComparator::evalPoly(const PhantomRelinKey &rk,
                         PhantomCiphertext &x) const
{
    // ── HECache: assign a unique prefix for this invocation ──────────────────
    // tl_power_prefix identifies all powers belonging to this evalPoly() call
    // so that HECache keys never collide across calls (sequential or concurrent
    // on different threads).  The prefix is cleared (and HECache entries pruned)
    // at the end of this function via a scope-guard lambda.
    if (hecache::HECache::is_initialized()) {
        uint64_t cid = s_eval_call_counter.fetch_add(1, std::memory_order_relaxed);
        tl_power_prefix = "pow_c" + std::to_string(cid);
    }

    // Scope-guard: always erase the HECache entries and reset the prefix on exit,
    // regardless of whether evalPoly() succeeds or throws.
    struct PrefixGuard {
        ~PrefixGuard() {
            if (hecache::HECache::is_initialized() && !tl_power_prefix.empty()) {
                hecache::HECache::instance().erase_prefix(tl_power_prefix);
                tl_power_prefix.clear();
            }
        }
    } prefix_guard;

    // x2 = x^2
    PhantomCiphertext x2 = multiply(ctx_, x, x);
    relinearize_inplace(ctx_, x2, rk);

    // babyStep[i] stores x^{2*(i+1)}, computed on demand.
    // babyStep[0] = x^2 is pre-filled.
    std::vector<PhantomCiphertext> babyStep(static_cast<size_t>(k_));
    std::vector<bool>              babyComp(static_cast<size_t>(k_), false);
    babyStep[0] = x2;
    babyComp[0] = true;

    // giantStep[j] stores (x^{2k})^{j+1}, computed on demand.
    // giantStep[0] = x^{2k} is pre-filled.
    PhantomCiphertext x2k = getPower(rk, babyStep, babyComp, k_);
    std::vector<PhantomCiphertext> giantStep(static_cast<size_t>(m_));
    std::vector<bool>              giantComp(static_cast<size_t>(m_), false);
    giantStep[0] = x2k;
    giantComp[0] = true;

    // lt accumulates the odd-degree polynomial evaluation
    PhantomCiphertext lt = x;

    if (m_ == (1LL << utils::nextPowerOf2(m_))) {
        evalPolyHelperPowerOf2(rk, lt, poly_, babyStep, babyComp,
                               giantStep, giantComp);
    } else {
        // Make a mutable copy of poly_ for evalPolyHelper
        std::vector<int64_t> poly_copy = poly_;
        evalPolyHelper(rk, lt, poly_copy, babyStep, babyComp,
                       giantStep, giantComp);

        if (top_coef_ != 1) {
            PhantomPlaintext plain_top = makeCoefPlain(top_coef_);
            multiply_plain_inplace(ctx_, lt, plain_top);
        }
        if (extra_coef_ != 0) {
            PhantomCiphertext topTerm =
                getPower(rk, giantStep, giantComp, m_);
            PhantomPlaintext plain_extra = makeCoefPlain(extra_coef_);
            multiply_plain_inplace(ctx_, topTerm, plain_extra);
            sub_inplace(ctx_, lt, topTerm);
        }
    }

    // lt = lt * x
    multiply_inplace(ctx_, lt, x);
    relinearize_inplace(ctx_, lt, rk);

    // topTerm = x^{2*baby_idx} * (x^{2k})^{giant_idx}
    PhantomCiphertext topTerm =
        multiply(ctx_,
                 getPower(rk, babyStep, babyComp, baby_idx_),
                 getPower(rk, giantStep, giantComp, giant_idx_));
    relinearize_inplace(ctx_, topTerm, rk);

    // eq = 1 - topTerm
    PhantomCiphertext eq = topTerm;
    negate_inplace(ctx_, eq);
    add_plain_inplace(ctx_, eq, makeCoefPlain(1));

    // lt += topTerm * alpha, where alpha = (p+1)/2
    int64_t alpha = (p_ + 1) >> 1;
    PhantomCiphertext topCopy = topTerm;
    multiply_plain_inplace(ctx_, topCopy, makeCoefPlain(alpha));
    add_inplace(ctx_, lt, topCopy);

    return {lt, eq};
}

// ============================================================
// Private: evalPolyHelper
// Mirrors CPU Comparator::evalPolyHelper.
// ============================================================

void GpuComparator::evalPolyHelper(
    const PhantomRelinKey          &rk,
    PhantomCiphertext              &x,
    std::vector<int64_t>            poly,          // by value (modified here)
    std::vector<PhantomCiphertext> &babyStep,
    std::vector<bool>              &babyComp,
    std::vector<PhantomCiphertext> &giantStep,
    std::vector<bool>              &giantComp) const
{
    int64_t deg = utils::getDegree(poly);

    if (deg <= static_cast<int64_t>(babyStep.size())) {
        evalPolySimple(rk, x, poly, babyStep, babyComp);
        return;
    }

    int64_t delta = deg % k_;
    int64_t n     = (deg + k_ - 1) / k_;
    int64_t t     = 1LL << utils::nextPowerOf2(n);

    if (n == t) {
        evalPolyHelperPowerOf2(rk, x, poly, babyStep, babyComp,
                               giantStep, giantComp);
        return;
    }

    if (n == t - 1 && delta == 0) {
        evalPolyPS(rk, x, poly, babyStep, babyComp,
                   giantStep, giantComp, t / 2, delta);
        return;
    }

    t         = t / 2;
    int64_t u = deg - k_ * (t - 1);

    // Build monic divisor x^u
    std::vector<int64_t> uu(static_cast<size_t>(u + 1), 0);
    uu.back() = 1;

    utils::PolyDivResult divqr = utils::dividePoly(poly, uu, p_);
    std::vector<int64_t> r = divqr.r;
    std::vector<int64_t> q = divqr.q;

    q[0]--;
    if (u >= static_cast<int64_t>(r.size()))
        r.resize(static_cast<size_t>(u + 1), 0);
    r[static_cast<size_t>(u)] = 1;

    evalPolyPS(rk, x, q, babyStep, babyComp, giantStep, giantComp, t / 2, 0);

    // tmp = x^{2*(u/k)} * (optionally) x^{2*delta}
    PhantomCiphertext tmp =
        getPower(rk, giantStep, giantComp, u / k_);
    if (delta != 0) {
        multiply_inplace(ctx_, tmp,
                         getPower(rk, babyStep, babyComp, delta));
        relinearize_inplace(ctx_, tmp, rk);
    }
    multiply_inplace(ctx_, x, tmp);
    relinearize_inplace(ctx_, x, rk);

    evalPolyHelper(rk, tmp, r, babyStep, babyComp, giantStep, giantComp);
    add_inplace(ctx_, x, tmp);
}

// ============================================================
// Private: evalPolyHelperPowerOf2
// Mirrors CPU Comparator::evalPolyHelperPowerOf2.
// ============================================================

void GpuComparator::evalPolyHelperPowerOf2(
    const PhantomRelinKey          &rk,
    PhantomCiphertext              &x,
    std::vector<int64_t>           &poly,
    std::vector<PhantomCiphertext> &babyStep,
    std::vector<bool>              &babyComp,
    std::vector<PhantomCiphertext> &giantStep,
    std::vector<bool>              &giantComp) const
{
    int64_t deg = utils::getDegree(poly);

    if (deg <= static_cast<int64_t>(babyStep.size())) {
        evalPolySimple(rk, x, poly, babyStep, babyComp);
        return;
    }

    int64_t n = (deg + k_ - 1) / k_;
    n = 1LL << utils::nextPowerOf2(n);

    // Monic divisor x^{(n-1)*k}
    std::vector<int64_t> uu(static_cast<size_t>((n - 1) * k_ + 1), 0);
    uu.back() = 1;

    utils::PolyDivResult divqr = utils::dividePoly(poly, uu, p_);
    std::vector<int64_t> r = divqr.r;
    std::vector<int64_t> q = divqr.q;

    if ((n - 1) * k_ >= static_cast<int64_t>(r.size()))
        r.resize(static_cast<size_t>((n - 1) * k_ + 1), 0);
    r[static_cast<size_t>((n - 1) * k_)] = 1;
    q[0]--;

    evalPolyPS(rk, x, r, babyStep, babyComp, giantStep, giantComp, n / 2, 0);

    PhantomCiphertext tmp;
    evalPolySimple(rk, tmp, q, babyStep, babyComp);

    for (int64_t i = 1; i < n; i *= 2) {
        multiply_inplace(ctx_, tmp,
                         getPower(rk, giantStep, giantComp, i));
        relinearize_inplace(ctx_, tmp, rk);
    }
    add_inplace(ctx_, x, tmp);
}

// ============================================================
// Private: evalPolyPS (Paterson-Stockmeyer core)
// Mirrors CPU Comparator::evalPolyPS.
// ============================================================

void GpuComparator::evalPolyPS(
    const PhantomRelinKey          &rk,
    PhantomCiphertext              &x,
    std::vector<int64_t>           &poly,
    std::vector<PhantomCiphertext> &babyStep,
    std::vector<bool>              &babyComp,
    std::vector<PhantomCiphertext> &giantStep,
    std::vector<bool>              &giantComp,
    int64_t t,
    int64_t delta) const
{
    int64_t deg = utils::getDegree(poly);

    if (deg <= static_cast<int64_t>(babyStep.size())) {
        evalPolySimple(rk, x, poly, babyStep, babyComp);
        return;
    }

    // Divide by x^{k*t}
    std::vector<int64_t> uu(static_cast<size_t>(k_ * t + 1), 0);
    uu.back() = 1;

    utils::PolyDivResult divqr = utils::dividePoly(poly, uu, p_);
    std::vector<int64_t> r = divqr.r;
    std::vector<int64_t> q = divqr.q;

    deg = utils::getDegree(q);
    int64_t coef = r[static_cast<size_t>(deg)];
    r[static_cast<size_t>(deg)]--;

    utils::PolyDivResult divcs = utils::dividePoly(r, q, p_);
    std::vector<int64_t> s = divcs.r;
    std::vector<int64_t> c = divcs.q;

    if (deg >= static_cast<int64_t>(s.size()))
        s.resize(static_cast<size_t>(deg + 1), 0);
    s[static_cast<size_t>(deg)] = 1;

    for (size_t i = 0; i < c.size(); i++) c[i] %= p_;
    c.resize(static_cast<size_t>(utils::getDegree(c) + 1));
    for (size_t i = 0; i < s.size(); i++) s[i] %= p_;
    s.resize(static_cast<size_t>(utils::getDegree(s) + 1));

    evalPolyPS(rk, x, q, babyStep, babyComp, giantStep, giantComp, t / 2, delta);

    PhantomCiphertext tmp;
    evalPolySimple(rk, tmp, c, babyStep, babyComp);
    add_inplace(ctx_, tmp,
                getPower(rk, giantStep, giantComp, t));
    multiply_inplace(ctx_, x, tmp);
    relinearize_inplace(ctx_, x, rk);

    evalPolyPS(rk, tmp, s, babyStep, babyComp, giantStep, giantComp, t / 2, delta);
    add_inplace(ctx_, x, tmp);
}

// ============================================================
// Private: evalPolySimple — evaluate degree-≤k polynomial using baby steps
//
// GPU version fixes the CPU bug: zero coefficients are skipped entirely
// instead of computing multiply_plain_inplace(tmp, plaintext(0)).
// ============================================================

void GpuComparator::evalPolySimple(
    const PhantomRelinKey          &rk,
    PhantomCiphertext              &x,
    std::vector<int64_t>           &poly,
    std::vector<PhantomCiphertext> &babyStep,
    std::vector<bool>              &babyComp) const
{
    bool x_initialized = false;

    for (int64_t i = 1; i < static_cast<int64_t>(poly.size()); i++) {
        int64_t coef = poly[static_cast<size_t>(i)] % p_;
        if (coef > p_ / 2) coef -= p_;
        if (coef == 0) continue;          // skip: no contribution

        PhantomCiphertext tmp =
            getPower(rk, babyStep, babyComp, i);
        multiply_plain_inplace(ctx_, tmp, makeCoefPlain(coef));

        if (!x_initialized) {
            x = std::move(tmp);
            x_initialized = true;
        } else {
            add_inplace(ctx_, x, tmp);
        }
    }

    // Ensure x is a valid ciphertext even if all higher coefficients were 0
    if (!x_initialized) {
        // Return zero ciphertext: use babyStep[0] * 0 as a zero CT
        x = babyStep[0];
        multiply_plain_inplace(ctx_, x, makeCoefPlain(0));
    }

    // Add constant term (poly[0])
    int64_t coef0 = poly[0] % p_;
    if (coef0 > p_ / 2) coef0 -= p_;
    // Always add, even if coef0 == 0 (add_plain_inplace of zero is a no-op)
    add_plain_inplace(ctx_, x, makeCoefPlain(coef0));
}

} // namespace core
} // namespace nshedb_gpu
