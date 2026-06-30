#pragma once
// =============================================================================
// ntt.cuh — Negacyclic NTT / INTT for BFV (Z[X]/(X^N + 1))
//
// ALGORITHM
// ---------
// Forward NTT  — Cooley-Tukey (CT) butterfly, natural input → CT-ordered output:
//   For m = 1, 2, 4, ..., N/2  (ascending):
//     t = N / (2*m)
//     For tid = 0 .. N/2-1 in parallel:
//       i = tid / t,  j_in = tid % t
//       j = 2*i*t + j_in
//       S = root_table[m + i]
//       U = a[j], V = a[j+t] * S mod p
//       a[j] = U+V, a[j+t] = U-V
//
// Inverse NTT — Gentleman-Sande (GS) butterfly, with N^{-1} scaling:
//   For h = N/2, N/4, ..., 1  (descending); t = 1, 2, ..., N/2 (ascending):
//     For tid = 0 .. N/2-1 in parallel:
//       i = tid / t,  j_in = tid % t
//       j = 2*i*t + j_in
//       S = root_inv_table[h + i]
//       U = a[j], V = a[j+t]
//       a[j] = U+V, a[j+t] = (U-V) * S mod p
//   Then: a[i] *= N^{-1} mod p for all i.
//
// ROOT TABLE (SEAL-compatible negacyclic layout)
// -----------------------------------------------
//   table_fwd[idx] = psi^{bit_rev(idx, logN+1)} mod p,  idx = 1 .. N-1
//   table_inv[idx] = psi_inv^{bit_rev(idx, logN+1)} mod p,  idx = 1 .. N-1
//
//   where psi is the primitive 2N-th root of unity mod p.
//   Index 0 is unused (set to 0).
//
// ARRAY LAYOUT
// ------------
//   poly: device array of size L * N, prime-major:
//     poly[l * N + i]  = coefficient i under prime index l
//
// VERIFIED: NTT then INTT gives back the original polynomial.
// See verify/verify_layer1.cu for test.
// =============================================================================

#include "bfv_params.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace bfv_core {

// ---------------------------------------------------------------------------
// NttTable: per-prime NTT precomputed data
// ---------------------------------------------------------------------------
struct NttTable {
    uint64_t  p;          // prime
    uint64_t  psi;        // primitive 2N-th root of unity mod p
    uint64_t  psi_inv;    // psi^{-1} mod p
    uint64_t  N_inv;      // N^{-1} mod p (for INTT normalization)
    int       N;          // polynomial degree this table is built for
    uint64_t* d_roots;    // device: d_roots[idx] = psi^{bit_rev(idx, logN+1)},  size N
    uint64_t* d_roots_inv;// device: d_roots_inv[idx] = psi_inv^{bit_rev(idx,logN+1)}, size N
};

// ---------------------------------------------------------------------------
// Host: create / free NttTable
//   psi must be the primitive 2N-th root of unity mod p (psi^{2N} ≡ 1, psi^N ≡ -1)
// ---------------------------------------------------------------------------
NttTable ntt_table_create(uint64_t p, uint64_t psi, int N);
void     ntt_table_free(NttTable& tbl);

// ---------------------------------------------------------------------------
// Single-prime NTT / INTT
// ---------------------------------------------------------------------------
void ntt_forward_single(uint64_t* poly_l, const NttTable& tbl, cudaStream_t s = 0);
void ntt_inverse_single(uint64_t* poly_l, const NttTable& tbl, cudaStream_t s = 0);

// ---------------------------------------------------------------------------
// Multi-prime NTT / INTT: poly is L*N array, prime-major
// ---------------------------------------------------------------------------
void ntt_forward(uint64_t* poly, int N, int L,
                 const NttTable* tbls, cudaStream_t s = 0);
void ntt_inverse(uint64_t* poly, int N, int L,
                 const NttTable* tbls, cudaStream_t s = 0);

} // namespace bfv_core
