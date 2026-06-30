// =============================================================================
// ks_ip.cu — Key-Switching Inner Product kernels
// =============================================================================

#include "ks_ip.cuh"
#include <vector>

namespace bfv_core {

// ---------------------------------------------------------------------------
// Core kernel: for one key digit, accumulate over B ciphertexts.
//
// gridDim.x  = B (one column per ciphertext)
// gridDim.y  = num_limbs (one row per RNS prime)
// blockDim.x = BLK (threads covering N coefficients)
//
// GeminiHE tiling: key_digit is broadcast across all B ciphertexts.
// L2 cache will naturally hold key_digit if B is modest and key fits in L2.
// For very large B or large num_limbs, consider streaming from HBM explicitly
// (future HECache integration, Layer 3).
// ---------------------------------------------------------------------------
__global__
static void k_ks_ip_batch(
    const uint64_t*  __restrict__ key_digit,   // [num_limbs * N]
    const uint64_t** __restrict__ ct_batch,    // B pointers, each [num_limbs * N]
    uint64_t**       __restrict__ acc_batch,   // B pointers, each [num_limbs * N]
    int N,
    int num_limbs,
    const uint64_t* __restrict__ primes)
{
    int b    = blockIdx.x;                         // ciphertext index
    int l    = blockIdx.y;                         // RNS prime index
    int n    = (int)(blockIdx.z * blockDim.x + threadIdx.x);  // coeff index
    if (n >= N) return;

    uint64_t p   = primes[l];
    size_t   off = (size_t)l * N + n;

    uint64_t kd  = key_digit[off];
    uint64_t ct  = ct_batch[b][off];
    uint64_t acc = acc_batch[b][off];

    acc_batch[b][off] = addmod(acc, mulmod(kd, ct, p), p);
}

// ---------------------------------------------------------------------------
// Single (non-batched) variant: same math, no ct_batch indirection.
// ---------------------------------------------------------------------------
__global__
static void k_ks_ip_single(
    const uint64_t* __restrict__ key_digit,
    const uint64_t* __restrict__ ct_poly,
    uint64_t*       __restrict__ acc,
    int N, int num_limbs,
    const uint64_t* __restrict__ primes)
{
    int l = blockIdx.y;
    int n = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (n >= N || l >= num_limbs) return;

    uint64_t p   = primes[l];
    size_t   off = (size_t)l * N + n;

    acc[off] = addmod(acc[off], mulmod(key_digit[off], ct_poly[off], p), p);
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

void ks_ip_batch(const uint64_t*  key_digit,
                 const uint64_t** ct_batch,
                 uint64_t**       acc_batch,
                 int B, int N, int num_limbs,
                 const uint64_t*  d_primes,
                 cudaStream_t     stream)
{
    constexpr int BLK = 256;
    // gridDim: (B, num_limbs, ceil(N/BLK))
    dim3 grid(B, num_limbs, (N + BLK - 1) / BLK);
    dim3 block(BLK);
    k_ks_ip_batch<<<grid, block, 0, stream>>>(
        key_digit, ct_batch, acc_batch, N, num_limbs, d_primes);
}

void ks_ip_single(const uint64_t* key_digit,
                  const uint64_t* ct_poly,
                  uint64_t*       acc,
                  int N, int num_limbs,
                  const uint64_t* d_primes,
                  cudaStream_t    stream)
{
    constexpr int BLK = 256;
    dim3 grid((N + BLK - 1) / BLK, num_limbs);
    dim3 block(BLK);
    k_ks_ip_single<<<grid, block, 0, stream>>>(
        key_digit, ct_poly, acc, N, num_limbs, d_primes);
}

// ---------------------------------------------------------------------------
// Device pointer array helpers
// ---------------------------------------------------------------------------

const uint64_t** make_d_ptr_array(const std::vector<const uint64_t*>& h_ptrs)
{
    const uint64_t** d;
    cudaMalloc(&d, h_ptrs.size() * sizeof(const uint64_t*));
    cudaMemcpy(d, h_ptrs.data(), h_ptrs.size() * sizeof(const uint64_t*),
               cudaMemcpyHostToDevice);
    return d;
}

uint64_t** make_d_ptr_array_mutable(const std::vector<uint64_t*>& h_ptrs)
{
    uint64_t** d;
    cudaMalloc(&d, h_ptrs.size() * sizeof(uint64_t*));
    cudaMemcpy(d, h_ptrs.data(), h_ptrs.size() * sizeof(uint64_t*),
               cudaMemcpyHostToDevice);
    return d;
}

} // namespace bfv_core
