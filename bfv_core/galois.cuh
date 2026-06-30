#pragma once
// =============================================================================
// galois.cuh — Galois automorphism (ciphertext rotation permutation)
//
// A Galois automorphism σ_k acts on a polynomial f(X) ∈ Z[X]/(X^N + 1) as:
//   σ_k(f)(X) = f(X^k)  where k is odd, 1 ≤ k < 2N, gcd(k, 2N) = 1.
//
// In coefficient representation, this is a permutation of indices:
//   out[i] = ± in[j]  where j*k ≡ i (mod 2N)
//   equivalently: out[(k*j) mod 2N] = in[j]  (with sign if (k*j)/2N is odd)
//
// The sign: if the mapped index k*j falls in [N, 2N), the coefficient is
// negated (because X^N ≡ -1 in the quotient ring Z[X]/(X^N + 1)).
//
// This permutation is applied independently to each RNS limb.
//
// For BFV rotations:
//   Row rotation by step s: galois_elt = 5^s mod 2N  (for BFV with specific structure)
//   Column swap:            galois_elt = 2N - 1
//   The actual galois elements depend on the slot structure and generator.
//
// API
// ---
//   apply_galois(in, galois_elt, out, N, L, d_primes, stream)
//     in  : device [L * N] polynomial (coefficient domain)
//     out : device [L * N] (may equal in for in-place if a temp is used internally)
//     galois_elt : the k in σ_k, must be odd, 1 ≤ k < 2N
// =============================================================================

#include "bfv_params.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace bfv_core {

// ---------------------------------------------------------------------------
// Apply Galois automorphism σ_{galois_elt}.
//
// Input polynomial in coefficient domain, L RNS limbs.
// in and out may NOT alias — use a temporary if in-place is needed.
// ---------------------------------------------------------------------------
void apply_galois(const uint64_t* in,
                  uint32_t        galois_elt,
                  uint64_t*       out,
                  int             N,
                  int             L,
                  const uint64_t* d_primes,
                  cudaStream_t    stream = 0);

// ---------------------------------------------------------------------------
// In-place version: uses an internal temporary buffer.
// ---------------------------------------------------------------------------
void apply_galois_inplace(uint64_t*       poly,
                           uint32_t        galois_elt,
                           int             N,
                           int             L,
                           const uint64_t* d_primes,
                           cudaStream_t    stream = 0);

// ---------------------------------------------------------------------------
// Compute the inverse Galois element: k^{-1} mod 2N.
// If σ_k is applied followed by σ_{k_inv}, the result is the identity.
// ---------------------------------------------------------------------------
__host__ inline
uint32_t galois_inverse(uint32_t k, int N) {
    // Find k^{-1} mod 2N using extended Euclidean / brute force for small N
    uint64_t mod = (uint64_t)2 * N;
    // Since gcd(k, 2N) = 1 (required), use Euler's theorem or brute force
    for (uint64_t inv = 1; inv < mod; inv++) {
        if ((k * inv) % mod == 1) return (uint32_t)inv;
    }
    return 0;  // should never happen if k is valid
}

} // namespace bfv_core
