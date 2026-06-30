#pragma once
// =============================================================================
// rns.cuh — RNS/CRT ModUp and ModDown for BFV key-switching
//
// Terminology (following BFV literature):
//   Q  = q_0 * q_1 * ... * q_{L-1}   (base modulus, L primes)
//   P  = p_0 * p_1 * ... * p_{K-1}   (special modulus, K primes)
//   QP = Q * P                        (extended modulus)
//
// ModUp (base extension Q → QP):
//   Input:  poly in RNS-Q layout, size [L * N]
//   Output: poly in RNS-QP layout, size [(L+K) * N]
//   The first L slices are copied unchanged; the last K slices are the
//   extension values computed by FastBConv (Garner's algorithm).
//
// ModDown (scale-and-round P out of QP → Q):
//   Input:  poly in RNS-QP layout, size [(L+K) * N]
//   Output: poly in RNS-Q layout, size [L * N]
//   Implements: b_i = round(a / P) mod q_i for i = 0..L-1.
//   With K=1: b_i = (a_i - a_P) * P^{-1} mod q_i.
//
// RnsModUpParams / RnsModDownParams hold all precomputed constants.
// Call rns_params_create() once and pass to ModUp/ModDown at runtime.
//
// NOTE: The current implementation supports K = 1 (single special prime).
//       Multi-P ModDown is a straightforward extension; stub noted below.
// =============================================================================

#include "bfv_params.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace bfv_core {

// ---------------------------------------------------------------------------
// FastBConv (ModUp) precomputed constants
//
// To compute a mod p_j from {a mod q_i, i=0..L-1}:
//
//   For each element position n:
//     t_i = a_i * hat_q_i_inv_qi mod q_i        (step 1, for each i)
//     b_j = sum_i( t_i * hat_q_i_mod_pj ) mod p_j  (step 2, for each j)
//
//   where:
//     hat_q_i_inv_qi = (Q / q_i)^{-1} mod q_i
//     hat_q_i_mod_pj = (Q / q_i) mod p_j
//
// We store: modup_mat[j * L + i] = hat_q_i_inv_qi * hat_q_i_mod_pj mod p_j
//           (folded into a single multiply per (i,j) pair during ModUp step 2).
//
// Actually we keep them separate so the step 1 normalization is absorbed into
// a scalar multiply per slice:
//   modup_hat_inv_qi[i] = hat_q_i_inv_qi  (L values, mod q_i)
//   modup_hat_mod_pj[j*L + i] = hat_q_i mod p_j  (L*K values, all mod p_j)
// ---------------------------------------------------------------------------
struct RnsModUpParams {
    int L;  // number of base primes
    int K;  // number of special primes
    // Device arrays:
    uint64_t* d_hat_inv_qi;    // size L: hat_q_i_inv mod q_i
    uint64_t* d_hat_mod_pj;    // size K*L: hat_q_i mod p_j, layout [j*L + i]
    uint64_t* d_q_primes;      // size L: the q primes
    uint64_t* d_p_primes;      // size K: the p primes
};

// ---------------------------------------------------------------------------
// ModDown (K=1 case: single special prime P) precomputed constants
//
//   b_i = (a_i - a_P * (P mod q_i)) * P^{-1} mod q_i
//       = (a_i - a_P_lift_i) * P_inv_qi mod q_i
//
//   where a_P_lift_i = a_P * (P mod q_i) mod q_i ... but actually:
//   b_i = ((a_i - a_P_mod_qi) mod q_i) * P_inv_qi mod q_i
//
//   Subtlety: a_P (the coefficient under prime P) must be lifted to an integer
//   in [-P/2, P/2) first (center-lift), then reduced mod q_i.
//
//   So:
//     a_P_centered = a_P > P/2 ? a_P - P : a_P     (integer in [-P/2, P/2))
//     delta_i = (a_P_centered % q_i + q_i) % q_i   (positive residue mod q_i)
//     b_i = (a_i - delta_i + q_i) % q_i * P_inv_qi mod q_i
// ---------------------------------------------------------------------------
struct RnsModDownParams {
    int L;
    int K;    // must be 1 for this implementation
    uint64_t  P;          // the single special prime
    // Device arrays:
    uint64_t* d_P_inv_qi; // size L: P^{-1} mod q_i
    uint64_t* d_P_mod_qi; // size L: P mod q_i (for center-lift delta)
    uint64_t* d_q_primes; // size L
};

// ---------------------------------------------------------------------------
// Allocation / free
// ---------------------------------------------------------------------------

// q_primes[0..L-1]: base primes, p_primes[0..K-1]: special primes
RnsModUpParams   rns_modup_params_create(const uint64_t* q_primes, int L,
                                          const uint64_t* p_primes, int K);
RnsModDownParams rns_moddown_params_create(const uint64_t* q_primes, int L,
                                            uint64_t P);

void rns_modup_params_free(RnsModUpParams&   p);
void rns_moddown_params_free(RnsModDownParams& p);

// ---------------------------------------------------------------------------
// ModUp:  in [L * N] → out [(L+K) * N]
// The first L*N coefficients of out are copied from in.
// The last K*N coefficients are the base-extended values.
// ---------------------------------------------------------------------------
void rns_modup(uint64_t*              out,    // device, (L+K)*N
               const uint64_t*        in,     // device, L*N
               int                    N,
               const RnsModUpParams&  params,
               cudaStream_t           stream = 0);

// ---------------------------------------------------------------------------
// ModDown (K=1):  in [(L+1)*N] → out [L*N]
// ---------------------------------------------------------------------------
void rns_moddown(uint64_t*                out,   // device, L*N
                 const uint64_t*          in,    // device, (L+1)*N
                 int                      N,
                 const RnsModDownParams&  params,
                 cudaStream_t             stream = 0);

} // namespace bfv_core
