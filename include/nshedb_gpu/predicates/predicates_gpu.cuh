#pragma once
// predicates_gpu.cuh — GPU predicate templates mirroring CPU predicates.h
//
// Uses Phantom free functions (namespace phantom) instead of SEAL Evaluator.
// COUNT_GPU / SUM_GPU are free functions (defined in predicates_gpu.cu).
// LT_GPU, GT_GPU, LTE_GPU, GTE_GPU, BETWEEN_GPU, IN_GPU, GROUPBY_GPU are
// header-only templates.
//
// COUNT_GPU / SUM_GPU accept a BfvRotationKeyStream& instead of
// PhantomGaloisKey& so that the rotation path can be managed (and potentially
// streamed / cached) through HECache in the future.  For callers that have
// only a PhantomGaloisKey, wrap it first:
//
//   BfvRotationKeyStream rks(ctx, gk);
//   COUNT_GPU(ctx, ct, slot_count, rks);

#include <phantom.h>
#include <vector>
#include "nshedb_gpu/core/comparator_gpu.cuh"
#include "nshedb_gpu/utils/math_utils_gpu.h"
#include "nshedb_gpu/utils/cache_bridge.h"

namespace nshedb_gpu {
namespace predicates {

using nshedb_gpu::core::GpuComparator;
using nshedb_gpu::utils::generateCombinations;
using nshedb_gpu::utils::BfvRotationKeyStream;

// ---------------------------------------------------------------------------
// Non-templated free functions (implementations in predicates_gpu.cu)
// ---------------------------------------------------------------------------

// Sum all slots into the first slot via log-depth rotation + addition.
// Works for BFV batch encoding with slot_count slots (= poly_degree).
//   slot_count must equal encoder.slot_count() used when building the context.
// Internally uses key_stream.rotate_inplace / key_stream.apply_galois_inplace_streaming.
PhantomCiphertext COUNT_GPU(
    const PhantomContext    &ctx,
    const PhantomCiphertext &a,
    size_t                   slot_count,
    BfvRotationKeyStream    &key_stream);

PhantomCiphertext SUM_GPU(
    const PhantomContext    &ctx,
    PhantomCiphertext        a,           // passed by value — modified in-place
    size_t                   slot_count,
    BfvRotationKeyStream    &key_stream);

// ---------------------------------------------------------------------------
// Templated predicates (inline, header-only)
// ---------------------------------------------------------------------------

// LT_GPU: returns ct where slot[i] = 1 iff a[i] < b[i]
template <typename T1, typename T2>
PhantomCiphertext LT_GPU(GpuComparator &comp,
                         const PhantomRelinKey &rk,
                         T1 a, T2 b)
{
    return comp.isLessThan(rk, a, b)[0];
}

// GT_GPU: a > b  ⟺  b < a
template <typename T1, typename T2>
PhantomCiphertext GT_GPU(GpuComparator &comp,
                         const PhantomRelinKey &rk,
                         T1 a, T2 b)
{
    return comp.isLessThan(rk, b, a)[0];
}

// LTE_GPU: a <= b  ⟺  lt OR eq
template <typename T1, typename T2>
PhantomCiphertext LTE_GPU(GpuComparator &comp,
                          const PhantomContext  &ctx,
                          const PhantomRelinKey &rk,
                          T1 a, T2 b)
{
    auto cmp = comp.isLessThan(rk, a, b);
    PhantomCiphertext res = cmp[0];
    add_inplace(ctx, res, cmp[1]);
    return res;
}

// GTE_GPU: a >= b  ⟺  b <= a
template <typename T1, typename T2>
PhantomCiphertext GTE_GPU(GpuComparator &comp,
                          const PhantomContext  &ctx,
                          const PhantomRelinKey &rk,
                          T1 a, T2 b)
{
    auto cmp = comp.isLessThan(rk, b, a);
    PhantomCiphertext res = cmp[0];
    add_inplace(ctx, res, cmp[1]);
    return res;
}

// BETWEEN_GPU: cond1 <= a <= cond2
template <typename T1, typename T2>
PhantomCiphertext BETWEEN_GPU(GpuComparator &comp,
                               const PhantomContext  &ctx,
                               const PhantomRelinKey &rk,
                               PhantomCiphertext &a,
                               T1 &cond1, T2 &cond2)
{
    PhantomCiphertext gte = GTE_GPU(comp, ctx, rk, a, cond1);
    PhantomCiphertext lte = LTE_GPU(comp, ctx, rk, a, cond2);
    multiply_inplace(ctx, gte, lte);
    relinearize_inplace(ctx, gte, rk);
    return gte;
}

// IN_GPU: returns 1 if a is in set (equality test for each member)
template <typename T>
PhantomCiphertext IN_GPU(GpuComparator &comp,
                         const PhantomContext  &ctx,
                         const PhantomRelinKey &rk,
                         PhantomCiphertext &a,
                         std::vector<T> &set)
{
    if (set.empty()) return PhantomCiphertext{};
    std::vector<PhantomCiphertext> res(set.size());
    for (size_t i = 0; i < set.size(); i++) {
        res[i] = comp.isEqual(rk, a, set[i]);
    }
    for (size_t i = 1; i < set.size(); i++) {
        add_inplace(ctx, res[0], res[i]);
    }
    return res[0];
}

// GROUPBY_GPU (single column): returns one indicator ciphertext per group value.
template <typename T>
std::vector<PhantomCiphertext> GROUPBY_GPU(GpuComparator &comp,
                                            const PhantomContext  &ctx,
                                            const PhantomRelinKey &rk,
                                            PhantomCiphertext &ctxt_group,
                                            std::vector<T> &groups)
{
    std::vector<PhantomCiphertext> result(groups.size());
    for (size_t i = 0; i < groups.size(); i++) {
        result[i] = comp.isEqual(rk, ctxt_group, groups[i]);
    }
    return result;
}

// GROUPBY_GPU (multi-column): cross-product of per-column group indicators.
template <typename T>
std::vector<PhantomCiphertext> GROUPBY_GPU(
    GpuComparator &comp,
    const PhantomContext  &ctx,
    const PhantomRelinKey &rk,
    std::vector<PhantomCiphertext> &ctxt_group,
    std::vector<std::vector<T>>    &groups)
{
    std::vector<std::vector<PhantomCiphertext>> per_col(ctxt_group.size());
    for (size_t i = 0; i < ctxt_group.size(); i++) {
        per_col[i] = GROUPBY_GPU(comp, ctx, rk, ctxt_group[i], groups[i]);
    }

    auto combos = generateCombinations<PhantomCiphertext>(per_col);
    std::vector<PhantomCiphertext> result(combos.size());
    for (size_t i = 0; i < combos.size(); i++) {
        result[i] = combos[i][0];
        for (size_t j = 1; j < combos[i].size(); j++) {
            multiply_inplace(ctx, result[i], combos[i][j]);
            relinearize_inplace(ctx, result[i], rk);
        }
    }
    return result;
}

} // namespace predicates
} // namespace nshedb_gpu
