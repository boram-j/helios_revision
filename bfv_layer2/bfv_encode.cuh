#pragma once
// =============================================================================
// bfv_encode.cuh — BFV batch encoder / decoder
//
// ENCODE:
//   m(X) = sum_i values[i] * X^i  (coeff = values[i] mod t, slot i = coeff i)
//   Lift to RNS: m mod q_l for each l  (no reduction needed since values[i] < t < q_l)
//   Forward NTT on each limb → pt.encoded in NTT domain.
//
// DECODE:
//   INTT limb 0 → coeff representation mod q_0
//   Since coefficients < t < q_0, CRT reconstruction is exact.
//   Reduce each coeff mod t → original values.
// =============================================================================

#include "bfv_context.cuh"
#include "bfv_types.cuh"
#include <vector>
#include <cstdint>

namespace bfv {

// Encode N integers (mod t) into BfvPlaintext.
// pt must already be allocated: pt.alloc(ctx).
void bfv_encode(const BfvContext&           ctx,
                const std::vector<uint64_t>& values,
                BfvPlaintext&               pt);

// Decode BfvPlaintext back to N integers mod t.
void bfv_decode(const BfvContext&      ctx,
                const BfvPlaintext&    pt,
                std::vector<uint64_t>& values_out);

} // namespace bfv
