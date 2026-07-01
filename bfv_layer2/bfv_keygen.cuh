#pragma once
// =============================================================================
// bfv_keygen.cuh — BFV Layer 2 key generation (secret, public, Galois)
// =============================================================================

#include "bfv_context.cuh"
#include "bfv_types.cuh"

namespace bfv {

// Generate a ternary secret key s ∈ {-1,0,1}^N in NTT form (L+K limbs).
void bfv_secret_keygen(const BfvContext& ctx, BfvSecretKey& sk);

// Generate public key (b, a) where b = -(a*s + e) mod Q, NTT form (L limbs).
void bfv_public_keygen(const BfvContext& ctx, const BfvSecretKey& sk,
                       BfvPublicKey& pk);

// Generate one Galois key entry for the given galois_elt.
// gke receives beta=L pairs (b[j], a[j]), each L+K limbs in NTT form.
void bfv_galois_keygen(const BfvContext& ctx, const BfvSecretKey& sk,
                       uint32_t galois_elt, GaloisKeyEntry& gke);

} // namespace bfv
