// =============================================================================
// bfv_keygen.cu — BFV key generation: secret key, public key, Galois key
//
// Sampling uses rand() seeded deterministically for reproducible tests.
// All output polynomials are stored in NTT domain on the GPU.
// =============================================================================

#include "bfv_keygen.cuh"
#include "../bfv_core/ntt.cuh"
#include "../bfv_core/poly_ops.cuh"
#include "../bfv_core/galois.cuh"

#include <cstdlib>
#include <cstring>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>

namespace bfv {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Sample N ternary coefficients {-1, 0, 1} on CPU (uniform over the three).
static void sample_ternary_cpu(int* out, int N, unsigned seed) {
    srand(seed);
    for (int i = 0; i < N; i++) {
        out[i] = (rand() % 3) - 1;   // maps {0,1,2} → {-1,0,1}
    }
}

// Encode a ternary int array into RNS layout:
//   h_rns[l*N + i] = ternary[i] < 0 ? primes[l]-1 : (uint64_t)ternary[i]
static void encode_ternary_rns(uint64_t*       h_rns,
                                const int*      ternary,
                                int             N,
                                const uint64_t* primes,
                                int             num_limbs) {
    for (int l = 0; l < num_limbs; l++) {
        uint64_t p   = primes[l];
        uint64_t* dst = h_rns + (size_t)l * N;
        for (int i = 0; i < N; i++) {
            int v  = ternary[i];
            dst[i] = (v < 0) ? (p - 1) : (uint64_t)v;
        }
    }
}

// Allocate a device array of primes and copy from host.
static uint64_t* make_device_primes(const uint64_t* h_primes, int count) {
    uint64_t* d_p = nullptr;
    cudaMalloc(&d_p, (size_t)count * sizeof(uint64_t));
    cudaMemcpy(d_p, h_primes, (size_t)count * sizeof(uint64_t),
               cudaMemcpyHostToDevice);
    return d_p;
}

// NTT all limbs of a newly-populated RnsPoly (limb stride = N, tables 0..LK-1).
static void ntt_all_limbs(uint64_t* d_data, int LK, int N,
                           const std::vector<bfv_core::NttTable>& tables) {
    for (int l = 0; l < LK; l++) {
        bfv_core::ntt_forward_single(d_data + (size_t)l * N, tables[l]);
    }
    cudaDeviceSynchronize();
}

static void intt_all_limbs(uint64_t* d_data, int LK, int N,
                            const std::vector<bfv_core::NttTable>& tables) {
    for (int l = 0; l < LK; l++) {
        bfv_core::ntt_inverse_single(d_data + (size_t)l * N, tables[l]);
    }
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// bfv_secret_keygen
//
// Generates s ∈ R, with coefficients in {-1, 0, 1}, encoded in NTT domain
// across L+K RNS limbs (base primes + special primes).
// ---------------------------------------------------------------------------
void bfv_secret_keygen(const BfvContext& ctx, BfvSecretKey& sk) {
    const int LK = ctx.L + ctx.K;
    const int N  = ctx.N;

    sk.alloc(ctx);   // allocates (L+K) limbs

    // Build full prime list [q_0..q_{L-1}, p_0..p_{K-1}]
    std::vector<uint64_t> all_primes(LK);
    for (int l = 0; l < ctx.L; l++) all_primes[l]         = ctx.primes[l];
    for (int k = 0; k < ctx.K; k++) all_primes[ctx.L + k] = ctx.special_primes[k];

    // Sample ternary poly on CPU (seed=42 for reproducibility)
    std::vector<int> ternary(N);
    sample_ternary_cpu(ternary.data(), N, 42u);

    // Encode into RNS layout and upload
    std::vector<uint64_t> h_rns((size_t)LK * N);
    encode_ternary_rns(h_rns.data(), ternary.data(), N, all_primes.data(), LK);

    sk.s.copy_from_host(h_rns.data(), LK);
    sk.s.is_ntt = false;

    // Forward NTT per limb → NTT domain
    ntt_all_limbs(sk.s.d_data, LK, N, ctx.ntt_tables);
    sk.s.is_ntt = true;
}

// ---------------------------------------------------------------------------
// bfv_public_keygen
//
// Generates (b, a) where:
//   a ← uniform_rand mod Q  (L limbs, NTT form)
//   e ← {-1,0,1}^N          (L limbs, NTT form)
//   b = -(a * s + e) mod Q   (L limbs, NTT form)
// ---------------------------------------------------------------------------
void bfv_public_keygen(const BfvContext& ctx, const BfvSecretKey& sk,
                       BfvPublicKey& pk) {
    const int L = ctx.L;
    const int N = ctx.N;

    pk.alloc(ctx);   // allocates L limbs for b and a

    // ── a: uniform random mod q_j ──────────────────────────────────────────
    {
        std::vector<uint64_t> h_a((size_t)L * N);
        srand(43u);   // distinct seed from sk
        for (int l = 0; l < L; l++) {
            uint64_t p   = ctx.primes[l];
            uint64_t* row = h_a.data() + (size_t)l * N;
            for (int i = 0; i < N; i++)
                row[i] = (uint64_t)rand() % p;
        }
        pk.a.copy_from_host(h_a.data(), L);
        pk.a.is_ntt = false;
        ntt_all_limbs(pk.a.d_data, L, N, ctx.ntt_tables);
        pk.a.is_ntt = true;
    }

    // ── e: ternary error, NTT form ─────────────────────────────────────────
    RnsPoly e;
    e.alloc(ctx, L);
    {
        std::vector<int>      ternary_e(N);
        std::vector<uint64_t> h_e((size_t)L * N);
        sample_ternary_cpu(ternary_e.data(), N, 44u);
        encode_ternary_rns(h_e.data(), ternary_e.data(), N, ctx.primes, L);
        e.copy_from_host(h_e.data(), L);
        e.is_ntt = false;
        ntt_all_limbs(e.d_data, L, N, ctx.ntt_tables);
        e.is_ntt = true;
    }

    // ── b = -(a * s + e) mod Q ─────────────────────────────────────────────
    uint64_t* d_primes = make_device_primes(ctx.primes, L);

    RnsPoly tmp;
    tmp.alloc(ctx, L);

    // tmp = a * s  (first L limbs of s — sk.s has L+K, stride still N)
    bfv_core::poly_mul_rns(tmp.d_data, pk.a.d_data, sk.s.d_data,
                            N, L, d_primes);
    // tmp = a*s + e
    bfv_core::poly_add_rns(tmp.d_data, tmp.d_data, e.d_data,
                            N, L, d_primes);
    // b = -tmp
    bfv_core::poly_neg_rns(pk.b.d_data, tmp.d_data, N, L, d_primes);
    pk.b.is_ntt = true;

    cudaFree(d_primes);
    tmp.free();
    e.free();
}

// ---------------------------------------------------------------------------
// bfv_galois_keygen
//
// Generates one GaloisKeyEntry for galois_elt k (σ_k : f(X) ↦ f(X^k)).
//
// beta = L digits.  For each digit j:
//   a[j] ← uniform_rand mod QP   (L+K limbs, NTT)
//   e_j  ← ternary               (L+K limbs, NTT)
//   b[j]  = -(a[j] * s + e_j) + P * s_galois   mod QP   (NTT)
//
// where s_galois = σ_k(s) and P = special_primes[0].
// ---------------------------------------------------------------------------
void bfv_galois_keygen(const BfvContext& ctx, const BfvSecretKey& sk,
                       uint32_t galois_elt, GaloisKeyEntry& gke) {
    const int L  = ctx.L;
    const int K  = ctx.K;
    const int LK = L + K;
    const int N  = ctx.N;
    const int beta = 1;   // single key pair — bfv_rotate uses ModUp, not digit decomposition

    // Single key pair (beta=1): use full ModUp of c1 in bfv_rotate, so one key
    // suffices.  beta=L digit decomposition with zero-extension is NOT used
    // because the zero special-prime accumulator means ModDown cannot subtract
    // the large a·s·c1 term, producing noise ≈ q_l/2 >> Δ.
    gke.alloc(ctx, beta);

    // ── Build combined prime list QP = [q_0..q_{L-1}, p_0..p_{K-1}] ───────
    std::vector<uint64_t> all_primes_h(LK);
    for (int l = 0; l < L; l++) all_primes_h[l]     = ctx.primes[l];
    for (int k = 0; k < K; k++) all_primes_h[L + k] = ctx.special_primes[k];

    uint64_t* d_all_primes = make_device_primes(all_primes_h.data(), LK);

    // ── s_galois = σ_{galois_elt}(s) ─────────────────────────────────────
    // Work in a temp copy so sk is never mutated.
    RnsPoly s_gal;
    s_gal.alloc(ctx, LK);
    cudaMemcpy(s_gal.d_data, sk.s.d_data,
               (size_t)LK * N * sizeof(uint64_t),
               cudaMemcpyDeviceToDevice);

    // INTT → coeff domain
    intt_all_limbs(s_gal.d_data, LK, N, ctx.ntt_tables);
    s_gal.is_ntt = false;

    // Apply σ_k in-place (galois.cuh provides a temp internally)
    bfv_core::apply_galois_inplace(s_gal.d_data, galois_elt, N, LK,
                                   d_all_primes);
    cudaDeviceSynchronize();

    // NTT → s_galois in NTT domain
    ntt_all_limbs(s_gal.d_data, LK, N, ctx.ntt_tables);
    s_gal.is_ntt = true;

    // ── Precompute P * s_galois  (scalar = special_primes[0] mod each prime) ─
    const uint64_t P = ctx.special_primes[0];
    std::vector<uint64_t> P_scalars_h(LK);
    for (int l = 0; l < LK; l++)
        P_scalars_h[l] = P % all_primes_h[l];   // P mod q_l (= 0 at l=L)

    uint64_t* d_P_scalars = nullptr;
    cudaMalloc(&d_P_scalars, (size_t)LK * sizeof(uint64_t));
    cudaMemcpy(d_P_scalars, P_scalars_h.data(),
               (size_t)LK * sizeof(uint64_t), cudaMemcpyHostToDevice);

    RnsPoly P_s_gal;
    P_s_gal.alloc(ctx, LK);
    bfv_core::poly_scalar_mul_rns(P_s_gal.d_data, s_gal.d_data,
                                  d_P_scalars, N, LK, d_all_primes);
    P_s_gal.is_ntt = true;

    // ── Single key pair (j=0 only) ─────────────────────────────────────────
    // a[0]: uniform random over QP
    {
        std::vector<uint64_t> h_a_0((size_t)LK * N);
        srand(100u);
        for (int l = 0; l < LK; l++) {
            uint64_t p   = all_primes_h[l];
            uint64_t* row = h_a_0.data() + (size_t)l * N;
            for (int i = 0; i < N; i++)
                row[i] = (uint64_t)rand() % p;
        }
        gke.a[0].copy_from_host(h_a_0.data(), LK);
        gke.a[0].is_ntt = false;
        ntt_all_limbs(gke.a[0].d_data, LK, N, ctx.ntt_tables);
        gke.a[0].is_ntt = true;
    }

    // e_0: ternary error over QP
    RnsPoly e_0;
    e_0.alloc(ctx, LK);
    {
        std::vector<int>      ternary_e0(N);
        std::vector<uint64_t> h_e0((size_t)LK * N);
        sample_ternary_cpu(ternary_e0.data(), N, 200u);
        encode_ternary_rns(h_e0.data(), ternary_e0.data(), N,
                           all_primes_h.data(), LK);
        e_0.copy_from_host(h_e0.data(), LK);
        e_0.is_ntt = false;
        ntt_all_limbs(e_0.d_data, LK, N, ctx.ntt_tables);
        e_0.is_ntt = true;
    }

    // b[0] = -(a[0] * s + e_0) + P * s_galois   mod QP
    {
        RnsPoly tmp;
        tmp.alloc(ctx, LK);

        bfv_core::poly_mul_rns(tmp.d_data,
                               gke.a[0].d_data, sk.s.d_data,
                               N, LK, d_all_primes);
        bfv_core::poly_add_rns(tmp.d_data,
                               tmp.d_data, e_0.d_data,
                               N, LK, d_all_primes);
        bfv_core::poly_neg_rns(gke.b[0].d_data,
                               tmp.d_data, N, LK, d_all_primes);
        bfv_core::poly_add_rns(gke.b[0].d_data,
                               gke.b[0].d_data, P_s_gal.d_data,
                               N, LK, d_all_primes);
        gke.b[0].is_ntt = true;

        tmp.free();
    }
    e_0.free();

    // ── Cleanup ────────────────────────────────────────────────────────────
    cudaFree(d_all_primes);
    cudaFree(d_P_scalars);
    s_gal.free();
    P_s_gal.free();
}

} // namespace bfv
