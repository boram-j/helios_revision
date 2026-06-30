// =============================================================================
// poly_ops.cu — Polynomial arithmetic CUDA kernels
// =============================================================================

#include "poly_ops.cuh"

namespace bfv_core {

// ---------------------------------------------------------------------------
// Single-prime kernels
// ---------------------------------------------------------------------------

__global__
static void k_add(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                  int N, uint64_t p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = addmod(a[i], b[i], p);
}

__global__
static void k_sub(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                  int N, uint64_t p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = submod(a[i], b[i], p);
}

__global__
static void k_neg(uint64_t* dst, const uint64_t* a, int N, uint64_t p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = negmod(a[i], p);
}

__global__
static void k_mul(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                  int N, uint64_t p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = mulmod(a[i], b[i], p);
}

__global__
static void k_scalar_mul(uint64_t* dst, const uint64_t* a,
                          uint64_t scalar, int N, uint64_t p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = mulmod(a[i], scalar, p);
}

static inline dim3 grid1d(int N, int blk = 256) {
    return dim3((N + blk - 1) / blk);
}

void poly_add_single(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                     int N, uint64_t p, cudaStream_t s) {
    k_add<<<grid1d(N), 256, 0, s>>>(dst, a, b, N, p);
}
void poly_sub_single(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                     int N, uint64_t p, cudaStream_t s) {
    k_sub<<<grid1d(N), 256, 0, s>>>(dst, a, b, N, p);
}
void poly_neg_single(uint64_t* dst, const uint64_t* a,
                     int N, uint64_t p, cudaStream_t s) {
    k_neg<<<grid1d(N), 256, 0, s>>>(dst, a, N, p);
}
void poly_mul_single(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                     int N, uint64_t p, cudaStream_t s) {
    k_mul<<<grid1d(N), 256, 0, s>>>(dst, a, b, N, p);
}
void poly_scalar_mul_single(uint64_t* dst, const uint64_t* a,
                             uint64_t scalar, int N, uint64_t p,
                             cudaStream_t s) {
    k_scalar_mul<<<grid1d(N), 256, 0, s>>>(dst, a, scalar, N, p);
}

// ---------------------------------------------------------------------------
// RNS-parallel kernels (gridDim.y = L, gridDim.x covers N elements)
// ---------------------------------------------------------------------------

// Helper: prime-indexed coeff offset
__device__ __forceinline__
static size_t rns_idx(int l, int i, int N) { return (size_t)l * N + i; }

__global__
static void k_add_rns(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                      int N, const uint64_t* primes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y;
    if (i >= N) return;
    size_t idx = rns_idx(l, i, N);
    dst[idx] = addmod(a[idx], b[idx], primes[l]);
}

__global__
static void k_sub_rns(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                      int N, const uint64_t* primes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y;
    if (i >= N) return;
    size_t idx = rns_idx(l, i, N);
    dst[idx] = submod(a[idx], b[idx], primes[l]);
}

__global__
static void k_neg_rns(uint64_t* dst, const uint64_t* a,
                      int N, const uint64_t* primes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y;
    if (i >= N) return;
    size_t idx = rns_idx(l, i, N);
    dst[idx] = negmod(a[idx], primes[l]);
}

__global__
static void k_mul_rns(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                      int N, const uint64_t* primes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y;
    if (i >= N) return;
    size_t idx = rns_idx(l, i, N);
    dst[idx] = mulmod(a[idx], b[idx], primes[l]);
}

__global__
static void k_fma_rns(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                      int N, const uint64_t* primes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y;
    if (i >= N) return;
    size_t idx = rns_idx(l, i, N);
    uint64_t p   = primes[l];
    uint64_t acc = dst[idx];
    uint64_t val = mulmod(a[idx], b[idx], p);
    dst[idx] = addmod(acc, val, p);
}

__global__
static void k_scalar_mul_rns(uint64_t* dst, const uint64_t* a,
                               const uint64_t* scalars,
                               int N, const uint64_t* primes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y;
    if (i >= N) return;
    size_t idx = rns_idx(l, i, N);
    dst[idx] = mulmod(a[idx], scalars[l], primes[l]);
}

// ---------------------------------------------------------------------------
// Host wrappers for RNS kernels
// ---------------------------------------------------------------------------
void poly_add_rns(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                  int N, int L, const uint64_t* d_primes, cudaStream_t s) {
    constexpr int BLK = 256;
    dim3 grid((N + BLK - 1) / BLK, L);
    k_add_rns<<<grid, dim3(BLK), 0, s>>>(dst, a, b, N, d_primes);
}
void poly_sub_rns(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                  int N, int L, const uint64_t* d_primes, cudaStream_t s) {
    constexpr int BLK = 256;
    dim3 grid((N + BLK - 1) / BLK, L);
    k_sub_rns<<<grid, dim3(BLK), 0, s>>>(dst, a, b, N, d_primes);
}
void poly_neg_rns(uint64_t* dst, const uint64_t* a,
                  int N, int L, const uint64_t* d_primes, cudaStream_t s) {
    constexpr int BLK = 256;
    dim3 grid((N + BLK - 1) / BLK, L);
    k_neg_rns<<<grid, dim3(BLK), 0, s>>>(dst, a, N, d_primes);
}
void poly_mul_rns(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                  int N, int L, const uint64_t* d_primes, cudaStream_t s) {
    constexpr int BLK = 256;
    dim3 grid((N + BLK - 1) / BLK, L);
    k_mul_rns<<<grid, dim3(BLK), 0, s>>>(dst, a, b, N, d_primes);
}
void poly_fma_rns(uint64_t* dst, const uint64_t* a, const uint64_t* b,
                  int N, int L, const uint64_t* d_primes, cudaStream_t s) {
    constexpr int BLK = 256;
    dim3 grid((N + BLK - 1) / BLK, L);
    k_fma_rns<<<grid, dim3(BLK), 0, s>>>(dst, a, b, N, d_primes);
}
void poly_scalar_mul_rns(uint64_t* dst, const uint64_t* a,
                          const uint64_t* d_scalars,
                          int N, int L, const uint64_t* d_primes,
                          cudaStream_t s) {
    constexpr int BLK = 256;
    dim3 grid((N + BLK - 1) / BLK, L);
    k_scalar_mul_rns<<<grid, dim3(BLK), 0, s>>>(dst, a, d_scalars, N, d_primes);
}

} // namespace bfv_core
