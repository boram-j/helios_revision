#pragma once
// =============================================================================
// bfv_rotate.cuh — BFV Galois rotation: automorphism + key-switch
// =============================================================================

#include "bfv_context.cuh"
#include "bfv_types.cuh"

namespace bfv {

// Full BFV rotation: galois automorphism on ct + key-switch using gke.
//
// galois_elt = (2*step + 1) % (2*N)  for left rotation by `step` slots.
//
// Steps (see bfv_rotate.cu):
//   1. σ_{galois_elt}(c0), σ_{galois_elt}(c1)  — INTT → permute → NTT
//   2. Key-switch σ(c1) via simple RNS digit decomposition + ks_ip_single
//   3. ModDown (L+K) → L limbs
//   4. ct_out = (σ(c0) + acc_b_down,  acc_a_down)
//
// Preconditions:
//   ct_in is in NTT domain (L limbs).
//   gke was generated for this galois_elt (bfv_galois_keygen).
//   ctx.K == 1  (single special prime; required by rns_moddown).
//
// ct_out is allocated inside this function (caller must call ct_out.free()).
void bfv_rotate(const BfvContext&    ctx,
                const GaloisKeyEntry& gke,
                uint32_t             galois_elt,
                const BfvCiphertext& ct_in,
                BfvCiphertext&       ct_out);

} // namespace bfv
