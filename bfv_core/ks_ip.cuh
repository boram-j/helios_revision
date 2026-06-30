#pragma once
// =============================================================================
// ks_ip.cuh — Key-Switching Inner Product (GeminiHE-style batched KS-IP)
//
// MATH OVERVIEW
// -------------
// BFV key-switching decomposes a ciphertext component c1 into "key digits"
// c1 = sum_d  c1_d * P^d  (RNS digit decomposition over the special prime basis)
// and computes the inner product with the evaluation key:
//   acc += c1_d ⊙ evk_d   (pointwise multiply in NTT domain, accumulate)
//
// GeminiHE's insight (from cpu_keyswitch.h math pattern):
//   Load one key digit evk_d once, multiply against B different ciphertexts
//   simultaneously. This amortizes the key-loading cost over B CTs.
//
// KERNEL SIGNATURE (as specified)
// ---------------------------------
//   ks_ip_batch(
//     key_digit  : device ptr to [num_limbs * N] NTT-domain key polynomial
//     ct_batch   : device ptr array [B] of ct polynomials, each [num_limbs * N]
//     acc_batch  : device ptr array [B] of accumulator polys, each [num_limbs * N]
//     B          : batch size (number of ciphertexts)
//     N          : polynomial degree
//     num_limbs  : number of RNS primes (= L or L+K depending on context)
//     d_primes   : device ptr to [num_limbs] prime values
//     stream     : CUDA stream
//   )
//
// Each call processes ONE key digit: acc_batch[b] += key_digit ⊙ ct_batch[b]
// for all b in [0, B). Caller loops over key digits (d = 0..D-1).
//
// For a complete key-switch: call ks_ip_batch D times (once per key digit),
// accumulating into acc_batch. Then run ModDown on each acc.
//
// TILING PATTERN (GeminiHE GPU style)
// ------------------------------------
//   gridDim.x = B   (one CTA column per ciphertext in batch)
//   gridDim.y = ceil(num_limbs * N / BLK_Y)  (tiles over coefficients)
//   blockDim.x = 1  (unused; we flatten below)
//
//   Or more practically:
//   gridDim.x = B
//   gridDim.y = num_limbs
//   blockDim.x = BLK  (covers N elements)
//
//   Key digit is loaded into shared memory (or L2 cache) once per CTA column,
//   then each thread handles one coefficient across all B CTs.
//
// NOTE on performance: The optimal block configuration depends on L2/shared-mem
// capacity. For num_limbs=6, N=32768, key_digit = 6*32768*8 = 1.57 MB — too
// large for shared memory. Rely on L2 cache; profile to tune block size.
// =============================================================================

#include "bfv_params.cuh"
#include <cuda_runtime.h>
#include <vector>

namespace bfv_core {

// ---------------------------------------------------------------------------
// Key-digit inner-product accumulation over a batch of B ciphertexts.
//
// acc_batch[b][l*N + n] += key_digit[l*N + n] * ct_batch[b][l*N + n] mod p_l
//
// All polynomials must already be in NTT domain.
// acc_batch is updated in-place (caller initializes to zero before first digit).
//
// key_digit  : device [num_limbs * N]
// ct_batch   : device array of B pointers, each [num_limbs * N]
// acc_batch  : device array of B pointers, each [num_limbs * N]
// ---------------------------------------------------------------------------
void ks_ip_batch(const uint64_t*  key_digit,
                 const uint64_t** ct_batch,
                 uint64_t**       acc_batch,
                 int              B,
                 int              N,
                 int              num_limbs,
                 const uint64_t*  d_primes,
                 cudaStream_t     stream = 0);

// ---------------------------------------------------------------------------
// Convenience: scalar (non-batched) key-digit inner product.
// acc[l*N + n] += key_digit[l*N + n] * ct_poly[l*N + n] mod p_l
// ---------------------------------------------------------------------------
void ks_ip_single(const uint64_t* key_digit,
                  const uint64_t* ct_poly,
                  uint64_t*       acc,
                  int             N,
                  int             num_limbs,
                  const uint64_t* d_primes,
                  cudaStream_t    stream = 0);

// ---------------------------------------------------------------------------
// Device pointer array helper: allocate a device array of B device pointers
// from a host-side vector of device pointers.
// Caller must free with cudaFree(result).
// ---------------------------------------------------------------------------
const uint64_t** make_d_ptr_array(const std::vector<const uint64_t*>& h_ptrs);
uint64_t**       make_d_ptr_array_mutable(const std::vector<uint64_t*>& h_ptrs);

} // namespace bfv_core
