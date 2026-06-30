#pragma once
// =============================================================================
// poly_ops.cuh — Polynomial arithmetic kernels mod RNS primes
//
// All arrays are device pointers in prime-major flat layout:
//   arr[l * N + i]  =  coefficient i of polynomial under prime index l
//
// All operations are element-wise (coefficient-by-coefficient), so they work
// identically on coefficient-domain and NTT-domain polynomials.
// =============================================================================

#include "bfv_params.cuh"
#include <cuda_runtime.h>

namespace bfv_core {

// ---------------------------------------------------------------------------
// Single-prime kernels (l = 0 only, stride-1 arrays of length N)
// ---------------------------------------------------------------------------

// dst[i] = (a[i] + b[i]) mod p
void poly_add_single(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                     int N, uint64_t p, cudaStream_t s = 0);

// dst[i] = (a[i] - b[i]) mod p
void poly_sub_single(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                     int N, uint64_t p, cudaStream_t s = 0);

// dst[i] = (-a[i]) mod p
void poly_neg_single(uint64_t* dst, const uint64_t* a,
                     int N, uint64_t p, cudaStream_t s = 0);

// dst[i] = (a[i] * b[i]) mod p  (NTT-domain pointwise multiply)
void poly_mul_single(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                     int N, uint64_t p, cudaStream_t s = 0);

// dst[i] = (a[i] * scalar) mod p
void poly_scalar_mul_single(uint64_t* dst, const uint64_t* a,
                             uint64_t scalar, int N, uint64_t p,
                             cudaStream_t s = 0);

// ---------------------------------------------------------------------------
// RNS-parallel kernels: operate on L primes simultaneously.
// dst, a, b are device pointers to L*N arrays in prime-major layout.
// primes: device pointer to L uint64_t prime values.
// ---------------------------------------------------------------------------

// dst[l*N+i] = (a[l*N+i] + b[l*N+i]) mod primes[l]
void poly_add_rns(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                  int N, int L, const uint64_t* d_primes,
                  cudaStream_t s = 0);

// dst[l*N+i] = (a[l*N+i] - b[l*N+i]) mod primes[l]
void poly_sub_rns(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                  int N, int L, const uint64_t* d_primes,
                  cudaStream_t s = 0);

// dst[l*N+i] = (-a[l*N+i]) mod primes[l]
void poly_neg_rns(uint64_t* dst, const uint64_t* a,
                  int N, int L, const uint64_t* d_primes,
                  cudaStream_t s = 0);

// dst[l*N+i] = (a[l*N+i] * b[l*N+i]) mod primes[l]
void poly_mul_rns(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                  int N, int L, const uint64_t* d_primes,
                  cudaStream_t s = 0);

// dst[l*N+i] += (a[l*N+i] * b[l*N+i]) mod primes[l]  (fused multiply-add)
void poly_fma_rns(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                  int N, int L, const uint64_t* d_primes,
                  cudaStream_t s = 0);

// dst[l*N+i] = (a[l*N+i] * scalars[l]) mod primes[l]
// scalars: device pointer to L uint64_t values, one per prime
void poly_scalar_mul_rns(uint64_t* dst, const uint64_t* a,
                          const uint64_t* d_scalars,
                          int N, int L, const uint64_t* d_primes,
                          cudaStream_t s = 0);

} // namespace bfv_core
