#pragma once
// comparator_gpu.cuh — GpuComparator class declaration.
//
// Mirrors the CPU nshedb::core::Comparator but uses Phantom free functions
// (evaluate.cuh) instead of SEAL's Evaluator methods.
//
// Key API mapping  (SEAL method → Phantom free function):
//   evaluator.sub(a,b,c)                → c = sub(ctx,a,b)
//   evaluator.sub_plain(ct,pt,res)      → res = sub_plain(ctx,ct,pt)
//   evaluator.negate_inplace(ct)        → negate_inplace(ctx,ct)
//   evaluator.multiply_inplace(a,b)     → multiply_inplace(ctx,a,b)
//   evaluator.relinearize_inplace(ct,rk)→ relinearize_inplace(ctx,ct,rk)
//   evaluator.multiply_plain_inplace(ct,pt) → multiply_plain_inplace(ctx,ct,pt)
//   evaluator.add_inplace(a,b)          → add_inplace(ctx,a,b)
//   evaluator.add_plain_inplace(ct,pt)  → add_plain_inplace(ctx,ct,pt)
//   evaluator.sub_inplace(a,b)          → sub_inplace(ctx,a,b)
//   evaluator.rotate_rows(ct,s,gk,res)  → rotate_inplace(ctx,ct,s,gk)
//   evaluator.rotate_columns(ct,gk,res) → apply_galois_inplace(ctx,ct,2N-1,gk)
//
// Polynomial coefficient computation (initCoeff / initPolyParam) is pure
// integer arithmetic — identical to the CPU version, performed on the host
// during construction.

#include <phantom.h>
#include <vector>
#include <cstdint>
#include "nshedb_gpu/utils/math_utils_gpu.h"
#include "nshedb_gpu/utils/conversion_gpu.h"

namespace nshedb_gpu {
namespace core {

using nshedb_gpu::utils::PolyDivResult;
using nshedb_gpu::utils::dividePoly;
using nshedb_gpu::utils::getDegree;
using nshedb_gpu::utils::nextPowerOf2;
using nshedb_gpu::utils::powMod;

class GpuComparator {
public:
    // ctx and encoder must outlive this object (stored as references).
    // The constructor reads plain_modulus from ctx and precomputes the
    // comparison polynomial coefficients entirely on the CPU.
    GpuComparator(const PhantomContext &ctx, const PhantomBatchEncoder &encoder);

    // -----------------------------------------------------------------------
    // Public comparison interface
    // -----------------------------------------------------------------------

    // Returns {lt_result, eq_result}: two GPU ciphertexts.
    //   lt_result[i] = 1 iff a[i] < b[i]  (mod p, with signed interpretation)
    //   eq_result[i] = 1 iff a[i] == b[i]
    std::vector<PhantomCiphertext> isLessThan(
        const PhantomRelinKey &rk,
        PhantomCiphertext a,              // passed by value — modified internally
        const PhantomCiphertext &b) const;

    std::vector<PhantomCiphertext> isLessThan(
        const PhantomRelinKey &rk,
        PhantomCiphertext a,
        const PhantomPlaintext &b_plain) const;

    PhantomCiphertext isEqual(
        const PhantomRelinKey &rk,
        PhantomCiphertext a,
        const PhantomCiphertext &b) const;

    // -----------------------------------------------------------------------
    // Getters (match CPU Comparator interface)
    // -----------------------------------------------------------------------
    int64_t getK() const noexcept { return k_; }
    int64_t getM() const noexcept { return m_; }

private:
    // -----------------------------------------------------------------------
    // Stored references (must outlive this object)
    // -----------------------------------------------------------------------
    const PhantomContext      &ctx_;
    const PhantomBatchEncoder &encoder_;

    // -----------------------------------------------------------------------
    // Polynomial parameters — computed once in the constructor on the CPU
    // -----------------------------------------------------------------------
    std::vector<int64_t> poly_;   // comparison polynomial coefficients over Z_p
    int64_t p_;                   // plain modulus value
    int64_t top_coef_;
    int64_t topInv_;
    int64_t extra_coef_;
    int64_t m_, k_, d_comp_;
    int64_t baby_idx_, giant_idx_, kk_;
    bool    divisible_;

    // -----------------------------------------------------------------------
    // Helper: encode scalar coefficient as a constant-slot PhantomPlaintext
    // coef is in the signed range (-p/2, p/2]; negative values are lifted
    // to their Z_p representatives before encoding.
    // -----------------------------------------------------------------------
    PhantomPlaintext makeCoefPlain(int64_t coef) const;

    // -----------------------------------------------------------------------
    // CPU-side init (identical to CPU Comparator)
    // -----------------------------------------------------------------------
    void initCoeff();
    void initPolyParam();

    // -----------------------------------------------------------------------
    // GPU polynomial evaluation helpers
    // babyStep[i] caches (x^2)^{i+1}, giantStep[j] caches (x^2k)^{j+1}.
    // computed_baby[i] / computed_giant[j] track whether index i/j is live.
    // -----------------------------------------------------------------------
    PhantomCiphertext getPower(
        const PhantomRelinKey &rk,
        std::vector<PhantomCiphertext> &x,
        std::vector<bool>              &computed,
        int64_t e) const;

    std::vector<PhantomCiphertext> evalPoly(
        const PhantomRelinKey &rk,
        PhantomCiphertext &x) const;

    void evalPolyHelper(
        const PhantomRelinKey &rk,
        PhantomCiphertext &x,
        std::vector<int64_t> poly,                 // passed by value (modified)
        std::vector<PhantomCiphertext> &babyStep,
        std::vector<bool>              &babyComp,
        std::vector<PhantomCiphertext> &giantStep,
        std::vector<bool>              &giantComp) const;

    void evalPolyHelperPowerOf2(
        const PhantomRelinKey &rk,
        PhantomCiphertext &x,
        std::vector<int64_t> &poly,
        std::vector<PhantomCiphertext> &babyStep,
        std::vector<bool>              &babyComp,
        std::vector<PhantomCiphertext> &giantStep,
        std::vector<bool>              &giantComp) const;

    void evalPolyPS(
        const PhantomRelinKey &rk,
        PhantomCiphertext &x,
        std::vector<int64_t> &poly,
        std::vector<PhantomCiphertext> &babyStep,
        std::vector<bool>              &babyComp,
        std::vector<PhantomCiphertext> &giantStep,
        std::vector<bool>              &giantComp,
        int64_t t,
        int64_t delta) const;

    void evalPolySimple(
        const PhantomRelinKey &rk,
        PhantomCiphertext &x,
        std::vector<int64_t> &poly,
        std::vector<PhantomCiphertext> &babyStep,
        std::vector<bool>              &babyComp) const;
};

} // namespace core
} // namespace nshedb_gpu
