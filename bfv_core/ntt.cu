// =============================================================================
// ntt.cu — Negacyclic NTT / INTT CUDA implementation
//
// Algorithm: SEAL-compatible negacyclic NTT in Z[X]/(X^N + 1).
// Root table: table[idx] = psi^{bit_rev(idx, logN+1)} mod p, idx = 1..N-1.
// Forward: CT butterfly, ascending m.  Inverse: GS butterfly, descending h.
// Verified by roundtrip test in verify/verify_layer1.cu.
// =============================================================================

#include "ntt.cuh"
#include <vector>
#include <cstring>

namespace bfv_core {

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

// Forward NTT stage (Cooley-Tukey butterfly):
//   m     : current outer loop value (1, 2, 4, ..., N/2)
//   N     : polynomial degree
//   p     : prime
//   roots : precomputed table, roots[m+i] is the twiddle for group i
__global__
static void k_ntt_fwd_stage(uint64_t* __restrict__ a,
                              int N, int m,
                              uint64_t p,
                              const uint64_t* __restrict__ roots)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= N / 2) return;

    int t    = N / (2 * m);
    int i    = tid / t;
    int j_in = tid % t;
    int j    = 2 * i * t + j_in;

    uint64_t S = roots[m + i];
    uint64_t U = a[j];
    uint64_t V = mulmod(a[j + t], S, p);

    a[j]     = addmod(U, V, p);
    a[j + t] = submod(U, V, p);
}

// Inverse NTT stage (Gentleman-Sande butterfly):
//   h     : current half-length (N/2, N/4, ..., 1)
//   t     : stride (1, 2, ..., N/2) — paired with h: t = N/(2h)
//   p     : prime
//   rinv  : inverse root table, rinv[h+i] is the twiddle for group i
__global__
static void k_ntt_inv_stage(uint64_t* __restrict__ a,
                              int N, int h, int t,
                              uint64_t p,
                              const uint64_t* __restrict__ rinv)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= N / 2) return;

    int i    = tid / t;
    int j_in = tid % t;
    int j    = 2 * i * t + j_in;

    uint64_t S = rinv[h + i];
    uint64_t U = a[j];
    uint64_t V = a[j + t];

    a[j]     = addmod(U, V, p);
    a[j + t] = mulmod(submod(U, V, p), S, p);
}

// Scale all elements by N_inv (final INTT step)
__global__
static void k_scale(uint64_t* __restrict__ a, int N, uint64_t N_inv, uint64_t p)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < N) a[i] = mulmod(a[i], N_inv, p);
}

// ---------------------------------------------------------------------------
// Multi-prime wrappers (gridDim.y = L primes)
// ---------------------------------------------------------------------------

__global__
static void k_ntt_fwd_stage_multi(uint64_t*        poly,
                                    int N, int m,
                                    const uint64_t*  primes,   // [L]
                                    const uint64_t** roots_arr) // [L] device ptrs
{
    int tid  = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int lidx = (int)blockIdx.y;
    if (tid >= N / 2) return;

    uint64_t p     = primes[lidx];
    const uint64_t* roots = roots_arr[lidx];
    uint64_t* a    = poly + (size_t)lidx * N;

    int t    = N / (2 * m);
    int i    = tid / t;
    int j_in = tid % t;
    int j    = 2 * i * t + j_in;

    uint64_t S = roots[m + i];
    uint64_t U = a[j];
    uint64_t V = mulmod(a[j + t], S, p);

    a[j]     = addmod(U, V, p);
    a[j + t] = submod(U, V, p);
}

__global__
static void k_ntt_inv_stage_multi(uint64_t*        poly,
                                    int N, int h, int t,
                                    const uint64_t*  primes,
                                    const uint64_t** rinv_arr)
{
    int tid  = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int lidx = (int)blockIdx.y;
    if (tid >= N / 2) return;

    uint64_t p      = primes[lidx];
    const uint64_t* rinv = rinv_arr[lidx];
    uint64_t* a     = poly + (size_t)lidx * N;

    int i    = tid / t;
    int j_in = tid % t;
    int j    = 2 * i * t + j_in;

    uint64_t S = rinv[h + i];
    uint64_t U = a[j];
    uint64_t V = a[j + t];

    a[j]     = addmod(U, V, p);
    a[j + t] = mulmod(submod(U, V, p), S, p);
}

__global__
static void k_scale_multi(uint64_t*       poly,
                            int N,
                            const uint64_t* N_invs,
                            const uint64_t* primes)
{
    int i    = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int lidx = (int)blockIdx.y;
    if (i >= N) return;
    uint64_t* a = poly + (size_t)lidx * N;
    a[i] = mulmod(a[i], N_invs[lidx], primes[lidx]);
}

// ---------------------------------------------------------------------------
// Host: root table construction
//
// table[idx] = psi^{bit_rev(idx, logN+1)} mod p,  for idx = 1 .. N-1
// table[0] = 0 (unused sentinel)
// ---------------------------------------------------------------------------
static void build_ntt_table(uint64_t* h_table, int N, uint64_t psi, uint64_t p)
{
    int logN = 0;
    { int tmp = N; while (tmp > 1) { logN++; tmp >>= 1; } }
    int bits = logN + 1;   // bit-width for the reversal

    h_table[0] = 0;
    for (int idx = 1; idx < N; idx++) {
        uint32_t br = bit_reverse((uint32_t)idx, bits);
        h_table[idx] = powmod(psi, (uint64_t)br, p);
    }
}

NttTable ntt_table_create(uint64_t p, uint64_t psi, int N)
{
    NttTable tbl;
    tbl.p       = p;
    tbl.psi     = psi;
    tbl.psi_inv = invmod(psi, p);
    tbl.N_inv   = invmod((uint64_t)N, p);
    tbl.N       = N;

    std::vector<uint64_t> h_fwd(N), h_inv(N);
    build_ntt_table(h_fwd.data(), N, psi,        p);
    build_ntt_table(h_inv.data(), N, tbl.psi_inv, p);

    cudaMalloc(&tbl.d_roots,     N * sizeof(uint64_t));
    cudaMalloc(&tbl.d_roots_inv, N * sizeof(uint64_t));
    cudaMemcpy(tbl.d_roots,     h_fwd.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(tbl.d_roots_inv, h_inv.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);

    return tbl;
}

void ntt_table_free(NttTable& tbl)
{
    if (tbl.d_roots)     { cudaFree(tbl.d_roots);     tbl.d_roots     = nullptr; }
    if (tbl.d_roots_inv) { cudaFree(tbl.d_roots_inv); tbl.d_roots_inv = nullptr; }
}

// ---------------------------------------------------------------------------
// Single-prime NTT / INTT
// ---------------------------------------------------------------------------
void ntt_forward_single(uint64_t* poly_l, const NttTable& tbl, cudaStream_t s)
{
    int N = tbl.N;
    constexpr int BLK = 256;
    dim3 block(BLK);
    dim3 grid((N / 2 + BLK - 1) / BLK);

    // Ascending m: 1, 2, 4, ..., N/2
    for (int m = 1; m < N; m <<= 1) {
        k_ntt_fwd_stage<<<grid, block, 0, s>>>(poly_l, N, m, tbl.p, tbl.d_roots);
    }
}

void ntt_inverse_single(uint64_t* poly_l, const NttTable& tbl, cudaStream_t s)
{
    int N = tbl.N;
    constexpr int BLK = 256;
    dim3 block(BLK);
    dim3 grid_bt((N / 2 + BLK - 1) / BLK);
    dim3 grid_sc((N     + BLK - 1) / BLK);

    // Descending h: N/2, N/4, ..., 1; ascending t: 1, 2, ..., N/2
    int t = 1;
    for (int h = N / 2; h >= 1; h >>= 1) {
        k_ntt_inv_stage<<<grid_bt, block, 0, s>>>(poly_l, N, h, t, tbl.p, tbl.d_roots_inv);
        t <<= 1;
    }
    // Multiply by N^{-1}
    k_scale<<<grid_sc, block, 0, s>>>(poly_l, N, tbl.N_inv, tbl.p);
}

// ---------------------------------------------------------------------------
// Multi-prime helpers: build device pointer arrays
// ---------------------------------------------------------------------------
static const uint64_t** make_dptr_arr(const NttTable* tbls, int L, bool use_inv)
{
    std::vector<const uint64_t*> h(L);
    for (int l = 0; l < L; l++)
        h[l] = use_inv ? tbls[l].d_roots_inv : tbls[l].d_roots;
    const uint64_t** d;
    cudaMalloc(&d, L * sizeof(const uint64_t*));
    cudaMemcpy(d, h.data(), L * sizeof(const uint64_t*), cudaMemcpyHostToDevice);
    return d;
}

// ---------------------------------------------------------------------------
// Multi-prime NTT / INTT
// ---------------------------------------------------------------------------
void ntt_forward(uint64_t* poly, int N, int L,
                 const NttTable* tbls, cudaStream_t s)
{
    constexpr int BLK = 256;
    dim3 block(BLK);

    // Build device arrays of primes and root pointers
    std::vector<uint64_t> h_primes(L);
    for (int l = 0; l < L; l++) h_primes[l] = tbls[l].p;

    uint64_t* d_primes;
    cudaMalloc(&d_primes, L * sizeof(uint64_t));
    cudaMemcpy(d_primes, h_primes.data(), L * sizeof(uint64_t), cudaMemcpyHostToDevice);

    const uint64_t** d_roots = make_dptr_arr(tbls, L, false);

    for (int m = 1; m < N; m <<= 1) {
        dim3 grid((N / 2 + BLK - 1) / BLK, L);
        k_ntt_fwd_stage_multi<<<grid, block, 0, s>>>(poly, N, m, d_primes, d_roots);
    }

    cudaFree(d_primes);
    cudaFree(d_roots);
}

void ntt_inverse(uint64_t* poly, int N, int L,
                 const NttTable* tbls, cudaStream_t s)
{
    constexpr int BLK = 256;
    dim3 block(BLK);

    std::vector<uint64_t> h_primes(L), h_N_invs(L);
    for (int l = 0; l < L; l++) {
        h_primes[l] = tbls[l].p;
        h_N_invs[l] = tbls[l].N_inv;
    }

    uint64_t* d_primes;
    uint64_t* d_N_invs;
    cudaMalloc(&d_primes, L * sizeof(uint64_t));
    cudaMalloc(&d_N_invs, L * sizeof(uint64_t));
    cudaMemcpy(d_primes, h_primes.data(), L * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N_invs, h_N_invs.data(), L * sizeof(uint64_t), cudaMemcpyHostToDevice);

    const uint64_t** d_rinv = make_dptr_arr(tbls, L, true);

    int t = 1;
    for (int h = N / 2; h >= 1; h >>= 1) {
        dim3 grid((N / 2 + BLK - 1) / BLK, L);
        k_ntt_inv_stage_multi<<<grid, block, 0, s>>>(poly, N, h, t, d_primes, d_rinv);
        t <<= 1;
    }
    {
        dim3 grid((N + BLK - 1) / BLK, L);
        k_scale_multi<<<grid, block, 0, s>>>(poly, N, d_N_invs, d_primes);
    }

    cudaFree(d_primes);
    cudaFree(d_N_invs);
    cudaFree(d_rinv);
}

} // namespace bfv_core
