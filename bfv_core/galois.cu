// =============================================================================
// galois.cu — Galois automorphism CUDA implementation
// =============================================================================

#include "galois.cuh"

namespace bfv_core {

// ---------------------------------------------------------------------------
// Gather kernel: for each output position i, read from source j = k^{-1}*i mod 2N.
//
// Chosen over a scatter approach because reads from `in` are coalesced
// (sequential output indices i → sequential source indices when k_inv is small).
// For large N a precomputed permutation table would make both directions fully
// coalesced — noted as a tuning opportunity.
// This requires k_inv = k^{-1} mod 2N passed in.
// Coalesced reads if stride is 1 (which it is for sequential i).
// ---------------------------------------------------------------------------
__global__
static void k_galois_gather(
    const uint64_t* __restrict__ in,
    uint64_t*       __restrict__ out,
    int             N,
    uint32_t        k_inv,      // galois_elt^{-1} mod 2N
    const uint64_t* __restrict__ primes)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int l = (int)blockIdx.y;
    if (i >= N) return;

    uint64_t p       = primes[l];
    // Source index: j such that k*j ≡ i (mod 2N) → j = k_inv * i mod 2N
    uint64_t src_mapped = ((uint64_t)k_inv * (uint64_t)i) % ((uint64_t)2 * N);

    size_t dest = (size_t)l * N + (size_t)i;
    if (src_mapped < (uint64_t)N) {
        out[dest] = in[(size_t)l * N + src_mapped];
    } else {
        // Source is in X^N..X^{2N-1} region → coefficient was negated
        // in[j] = in[src_mapped - N] and the sign flipped
        uint64_t val = in[(size_t)l * N + src_mapped - N];
        out[dest] = negmod(val, p);
    }
}

// ---------------------------------------------------------------------------
// Host wrapper: apply_galois (out-of-place)
// ---------------------------------------------------------------------------
void apply_galois(const uint64_t* in,
                  uint32_t        galois_elt,
                  uint64_t*       out,
                  int             N,
                  int             L,
                  const uint64_t* d_primes,
                  cudaStream_t    stream)
{
    // Use gather kernel for coalesced reads from 'in'.
    uint32_t k_inv = galois_inverse(galois_elt, N);

    constexpr int BLK = 256;
    dim3 grid((N + BLK - 1) / BLK, L);
    dim3 block(BLK);

    k_galois_gather<<<grid, block, 0, stream>>>(in, out, N, k_inv, d_primes);
}

// ---------------------------------------------------------------------------
// In-place version: alloc temp, apply, copy back, free
// ---------------------------------------------------------------------------
void apply_galois_inplace(uint64_t*       poly,
                           uint32_t        galois_elt,
                           int             N,
                           int             L,
                           const uint64_t* d_primes,
                           cudaStream_t    stream)
{
    uint64_t* tmp;
    cudaMallocAsync(&tmp, (size_t)L * N * sizeof(uint64_t), stream);
    apply_galois(poly, galois_elt, tmp, N, L, d_primes, stream);
    cudaMemcpyAsync(poly, tmp, (size_t)L * N * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, stream);
    cudaFreeAsync(tmp, stream);
}

} // namespace bfv_core
