// =============================================================================
// rns.cu — RNS ModUp and ModDown CUDA implementation
// =============================================================================

#include "rns.cuh"
#include <vector>
#include <cstring>
#include <stdexcept>

namespace bfv_core {

// ---------------------------------------------------------------------------
// ModUp kernels
// ---------------------------------------------------------------------------

// Step 1: normalize each base slice by hat_q_i_inv_qi.
// Result stored in a temporary "t" array: t[i*N + n] = a_i[n] * hat_inv_i mod q_i.
// We do this in-place on a scratch copy (caller allocates temp).
__global__
static void k_modup_step1(const uint64_t* __restrict__ in,     // [L*N] input
                           uint64_t*       __restrict__ tmp,    // [L*N] output (normalized)
                           int N, int L,
                           const uint64_t* __restrict__ hat_inv_qi,
                           const uint64_t* __restrict__ q_primes)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;  // base prime index
    if (n >= N || i >= L) return;

    uint64_t qi  = q_primes[i];
    uint64_t inv = hat_inv_qi[i];
    uint64_t ai  = in[(size_t)i * N + n];
    tmp[(size_t)i * N + n] = mulmod(ai, inv, qi);
}

// Step 2: for each special prime p_j, accumulate the sum over base primes.
// out[j*N + n] = sum_i( tmp[i*N + n] * hat_q_i_mod_pj[j*L + i] ) mod p_j
__global__
static void k_modup_step2(const uint64_t* __restrict__ tmp,    // [L*N]
                           uint64_t*       __restrict__ ext,    // [K*N] extension part
                           int N, int L, int K,
                           const uint64_t* __restrict__ hat_mod_pj, // [K*L]
                           const uint64_t* __restrict__ p_primes)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;   // special prime index
    if (n >= N || j >= K) return;

    uint64_t pj  = p_primes[j];
    uint64_t acc = 0;
    for (int i = 0; i < L; i++) {
        uint64_t t_in  = tmp[(size_t)i * N + n];
        uint64_t coeff = hat_mod_pj[(size_t)j * L + i];
        acc = addmod(acc, mulmod(t_in, coeff, pj), pj);
    }
    ext[(size_t)j * N + n] = acc;
}

// ---------------------------------------------------------------------------
// ModDown kernel (K=1: single special prime P)
//
// in layout: [0 .. L-1] slices = base Q, slice L = special P
// out layout: [0 .. L-1] slices
//
// For each coefficient position n and each base prime q_i:
//   a_P  = in[L*N + n]   (residue under P)
//   a_i  = in[i*N + n]   (residue under q_i)
//   Center-lift a_P: if a_P > P/2 → a_P_c = a_P - P (else a_P_c = a_P)
//   delta_i = (a_P_c mod q_i + q_i) mod q_i
//   out[i*N + n] = (a_i - delta_i + q_i) mod q_i * P_inv_qi mod q_i
// ---------------------------------------------------------------------------
__global__
static void k_moddown_k1(const uint64_t* __restrict__ in,   // [(L+1)*N]
                          uint64_t*       __restrict__ out,  // [L*N]
                          int N, int L,
                          uint64_t P,
                          const uint64_t* __restrict__ P_inv_qi, // [L]
                          const uint64_t* __restrict__ P_mod_qi, // [L]
                          const uint64_t* __restrict__ q_primes) // [L]
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;   // base prime index
    if (n >= N || i >= L) return;

    // a_P under special prime P
    uint64_t aP = in[(size_t)L * N + n];

    // Center-lift: a_P ∈ [0, P) → a_P_c ∈ (-P/2, P/2]
    // Represent center-lift as: if aP > P/2, treat as negative (aP - P)
    // We compute delta_i = aP mod q_i (no sign, then subtract separately)
    // Correct formula: (aP - P) mod qi when aP > P/2:
    //   delta_i = aP mod qi - P mod qi (= aP mod qi - P_mod_qi[i])
    // This handles the sign via modular arithmetic.

    uint64_t qi        = q_primes[i];
    uint64_t P_inv     = P_inv_qi[i];
    uint64_t Pmod      = P_mod_qi[i];   // P mod q_i
    uint64_t ai        = in[(size_t)i * N + n];

    // delta_i = aP mod qi.
    // If aP > P/2, we need to subtract P mod qi from the correction.
    // Combined: delta_i = (aP mod qi) - (P/2 > aP ? 0 : Pmod) ... simplify:
    //
    // Equivalent clean formulation used in SEAL/HElib:
    //   b_i = ((ai - aP * Pmod^{-1} ... )) -- this doesn't work cleanly.
    //
    // Correct approach: center the special residue, then lift.
    //   If aP <= P/2:   correction_i = aP mod qi   (positive)
    //   If aP > P/2:    correction_i = (aP - P) mod qi = (aP mod qi - Pmod + qi) mod qi
    //
    // Compute correction_i:
    uint64_t aP_mod_qi = (uint64_t)((__uint128_t)aP % qi);  // aP mod qi
    uint64_t correction;
    if (aP <= P / 2) {
        correction = aP_mod_qi;
    } else {
        // (aP - P) mod qi: since aP > P/2, aP - P is negative integer.
        // In mod qi: (aP mod qi - P mod qi + qi) mod qi
        correction = submod(aP_mod_qi, Pmod, qi);
    }

    // b_i = (ai - correction) * P_inv mod qi
    uint64_t diff = submod(ai, correction, qi);
    out[(size_t)i * N + n] = mulmod(diff, P_inv, qi);
}

// ---------------------------------------------------------------------------
// Param creation / free
// ---------------------------------------------------------------------------

RnsModUpParams rns_modup_params_create(const uint64_t* q_primes, int L,
                                        const uint64_t* p_primes, int K)
{
    RnsModUpParams p;
    p.L = L;
    p.K = K;

    // Compute hat_q_i_inv_qi: (Q/q_i)^{-1} mod q_i
    // Q/q_i = product of all q_j, j ≠ i
    std::vector<uint64_t> hat_inv_qi(L);
    for (int i = 0; i < L; i++) {
        uint64_t qi  = q_primes[i];
        // Compute product of all q_j (j≠i) mod q_i
        // Use __uint128_t to avoid overflow
        uint64_t prod = 1;
        for (int j = 0; j < L; j++) {
            if (j == i) continue;
            prod = (uint64_t)((__uint128_t)prod * (q_primes[j] % qi) % qi);
        }
        hat_inv_qi[i] = invmod(prod, qi);
    }

    // Compute hat_q_i mod p_j for each (j,i) pair
    // hat_q_i mod p_j = (Q/q_i) mod p_j = product of all q_k (k≠i) mod p_j
    std::vector<uint64_t> hat_mod_pj(K * L);
    for (int j = 0; j < K; j++) {
        uint64_t pj = p_primes[j];
        for (int i = 0; i < L; i++) {
            uint64_t prod = 1;
            for (int k = 0; k < L; k++) {
                if (k == i) continue;
                prod = (uint64_t)((__uint128_t)prod * (q_primes[k] % pj) % pj);
            }
            hat_mod_pj[j * L + i] = prod;
        }
    }

    // Upload to device
    cudaMalloc(&p.d_hat_inv_qi,  L * sizeof(uint64_t));
    cudaMalloc(&p.d_hat_mod_pj,  (size_t)K * L * sizeof(uint64_t));
    cudaMalloc(&p.d_q_primes,    L * sizeof(uint64_t));
    cudaMalloc(&p.d_p_primes,    K * sizeof(uint64_t));

    cudaMemcpy(p.d_hat_inv_qi, hat_inv_qi.data(),  L * sizeof(uint64_t),         cudaMemcpyHostToDevice);
    cudaMemcpy(p.d_hat_mod_pj, hat_mod_pj.data(),  (size_t)K*L*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(p.d_q_primes,   q_primes,            L * sizeof(uint64_t),         cudaMemcpyHostToDevice);
    cudaMemcpy(p.d_p_primes,   p_primes,            K * sizeof(uint64_t),         cudaMemcpyHostToDevice);

    return p;
}

void rns_modup_params_free(RnsModUpParams& p) {
    cudaFree(p.d_hat_inv_qi);  p.d_hat_inv_qi = nullptr;
    cudaFree(p.d_hat_mod_pj);  p.d_hat_mod_pj = nullptr;
    cudaFree(p.d_q_primes);    p.d_q_primes   = nullptr;
    cudaFree(p.d_p_primes);    p.d_p_primes   = nullptr;
}

RnsModDownParams rns_moddown_params_create(const uint64_t* q_primes, int L, uint64_t P)
{
    RnsModDownParams p;
    p.L = L;
    p.K = 1;
    p.P = P;

    std::vector<uint64_t> P_inv_qi(L), P_mod_qi(L);
    for (int i = 0; i < L; i++) {
        uint64_t qi  = q_primes[i];
        P_mod_qi[i]  = (uint64_t)((__uint128_t)P % qi);
        P_inv_qi[i]  = invmod(P_mod_qi[i], qi);
    }

    cudaMalloc(&p.d_P_inv_qi, L * sizeof(uint64_t));
    cudaMalloc(&p.d_P_mod_qi, L * sizeof(uint64_t));
    cudaMalloc(&p.d_q_primes, L * sizeof(uint64_t));

    cudaMemcpy(p.d_P_inv_qi, P_inv_qi.data(), L * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(p.d_P_mod_qi, P_mod_qi.data(), L * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(p.d_q_primes, q_primes,         L * sizeof(uint64_t), cudaMemcpyHostToDevice);

    return p;
}

void rns_moddown_params_free(RnsModDownParams& p) {
    cudaFree(p.d_P_inv_qi);  p.d_P_inv_qi = nullptr;
    cudaFree(p.d_P_mod_qi);  p.d_P_mod_qi = nullptr;
    cudaFree(p.d_q_primes);  p.d_q_primes = nullptr;
}

// ---------------------------------------------------------------------------
// ModUp host wrapper
// ---------------------------------------------------------------------------
void rns_modup(uint64_t* out, const uint64_t* in, int N,
               const RnsModUpParams& params, cudaStream_t stream)
{
    int L = params.L;
    int K = params.K;
    constexpr int BLK = 256;

    // Copy the base Q portion of in to out unchanged
    cudaMemcpyAsync(out, in, (size_t)L * N * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, stream);

    // Temporary buffer for normalized base slices
    uint64_t* tmp;
    cudaMallocAsync(&tmp, (size_t)L * N * sizeof(uint64_t), stream);

    // Step 1: normalize base slices
    {
        dim3 grid((N + BLK - 1) / BLK, L);
        k_modup_step1<<<grid, dim3(BLK), 0, stream>>>(
            in, tmp, N, L, params.d_hat_inv_qi, params.d_q_primes);
    }

    // Step 2: extend to special primes (result goes after the base Q part)
    {
        uint64_t* ext = out + (size_t)L * N;
        dim3 grid((N + BLK - 1) / BLK, K);
        k_modup_step2<<<grid, dim3(BLK), 0, stream>>>(
            tmp, ext, N, L, K, params.d_hat_mod_pj, params.d_p_primes);
    }

    cudaFreeAsync(tmp, stream);
}

// ---------------------------------------------------------------------------
// ModDown host wrapper (K=1)
// ---------------------------------------------------------------------------
void rns_moddown(uint64_t* out, const uint64_t* in, int N,
                 const RnsModDownParams& params, cudaStream_t stream)
{
    int L = params.L;
    constexpr int BLK = 256;

    dim3 grid((N + BLK - 1) / BLK, L);
    k_moddown_k1<<<grid, dim3(BLK), 0, stream>>>(
        in, out, N, L, params.P,
        params.d_P_inv_qi, params.d_P_mod_qi, params.d_q_primes);
}

} // namespace bfv_core
