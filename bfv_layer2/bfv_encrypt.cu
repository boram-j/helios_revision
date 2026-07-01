// =============================================================================
// bfv_encrypt.cu — BFV encryption, decryption, and ciphertext addition
//
// All polynomials are stored in NTT domain (L limbs, prime-major layout).
// Sampling uses srand/rand seeded deterministically for reproducible tests.
// =============================================================================

#include "bfv_encrypt.cuh"
#include "../bfv_core/ntt.cuh"
#include "../bfv_core/poly_ops.cuh"

#include <cstdlib>
#include <cstring>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>

namespace bfv {

// ---------------------------------------------------------------------------
// Internal helpers  (same patterns as bfv_keygen.cu)
// ---------------------------------------------------------------------------

static void sample_ternary_cpu(int* out, int N, unsigned seed) {
    srand(seed);
    for (int i = 0; i < N; i++)
        out[i] = (rand() % 3) - 1;   // {0,1,2} → {-1,0,1}
}

static void encode_ternary_rns(uint64_t*       h_rns,
                                const int*      ternary,
                                int             N,
                                const uint64_t* primes,
                                int             num_limbs) {
    for (int l = 0; l < num_limbs; l++) {
        uint64_t  p   = primes[l];
        uint64_t* dst = h_rns + (size_t)l * N;
        for (int i = 0; i < N; i++) {
            int v  = ternary[i];
            dst[i] = (v < 0) ? (p - 1) : (uint64_t)v;
        }
    }
}

static uint64_t* make_device_primes(const uint64_t* h_primes, int count) {
    uint64_t* d_p = nullptr;
    cudaMalloc(&d_p, (size_t)count * sizeof(uint64_t));
    cudaMemcpy(d_p, h_primes, (size_t)count * sizeof(uint64_t),
               cudaMemcpyHostToDevice);
    return d_p;
}

static void ntt_all_limbs(uint64_t* d_data, int num_limbs, int N,
                           const std::vector<bfv_core::NttTable>& tables) {
    for (int l = 0; l < num_limbs; l++)
        bfv_core::ntt_forward_single(d_data + (size_t)l * N, tables[l]);
    cudaDeviceSynchronize();
}

static void intt_all_limbs(uint64_t* d_data, int num_limbs, int N,
                            const std::vector<bfv_core::NttTable>& tables) {
    for (int l = 0; l < num_limbs; l++)
        bfv_core::ntt_inverse_single(d_data + (size_t)l * N, tables[l]);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// bfv_encrypt
//
// Samples u, e1, e2 ternary (seeds 99, 100, 101).
// Computes:
//   ct.c0 = pk.b * u + e1 + delta * m
//   ct.c1 = pk.a * u + e2
// All in NTT domain over L base primes.
// ---------------------------------------------------------------------------
void bfv_encrypt(const BfvContext& ctx, const BfvPublicKey& pk,
                 const BfvPlaintext& pt, BfvCiphertext& ct) {
    const int L = ctx.L;
    const int N = ctx.N;

    ct.alloc(ctx);
    ct.is_ntt = true;

    uint64_t* d_primes = make_device_primes(ctx.primes, L);

    // ── u: ternary, seed 99 ───────────────────────────────────────────────
    RnsPoly u;
    u.alloc(ctx, L);
    {
        std::vector<int>      ternary(N);
        std::vector<uint64_t> h_u((size_t)L * N);
        sample_ternary_cpu(ternary.data(), N, 99u);
        encode_ternary_rns(h_u.data(), ternary.data(), N, ctx.primes, L);
        u.copy_from_host(h_u.data(), L);
        u.is_ntt = false;
        ntt_all_limbs(u.d_data, L, N, ctx.ntt_tables);
        u.is_ntt = true;
    }

    // ── e1: ternary, seed 100 ────────────────────────────────────────────
    RnsPoly e1;
    e1.alloc(ctx, L);
    {
        std::vector<int>      ternary(N);
        std::vector<uint64_t> h_e((size_t)L * N);
        sample_ternary_cpu(ternary.data(), N, 100u);
        encode_ternary_rns(h_e.data(), ternary.data(), N, ctx.primes, L);
        e1.copy_from_host(h_e.data(), L);
        e1.is_ntt = false;
        ntt_all_limbs(e1.d_data, L, N, ctx.ntt_tables);
        e1.is_ntt = true;
    }

    // ── e2: ternary, seed 101 ────────────────────────────────────────────
    RnsPoly e2;
    e2.alloc(ctx, L);
    {
        std::vector<int>      ternary(N);
        std::vector<uint64_t> h_e((size_t)L * N);
        sample_ternary_cpu(ternary.data(), N, 101u);
        encode_ternary_rns(h_e.data(), ternary.data(), N, ctx.primes, L);
        e2.copy_from_host(h_e.data(), L);
        e2.is_ntt = false;
        ntt_all_limbs(e2.d_data, L, N, ctx.ntt_tables);
        e2.is_ntt = true;
    }

    // ── delta * m ─────────────────────────────────────────────────────────
    // pt.encoded is in NTT domain.  Copy, INTT to coeff domain, scale by
    // delta % q_l per limb, NTT back.
    RnsPoly dm;
    dm.alloc(ctx, L);
    cudaMemcpy(dm.d_data, pt.encoded.d_data,
               (size_t)L * N * sizeof(uint64_t),
               cudaMemcpyDeviceToDevice);
    dm.is_ntt = true;

    // INTT → coefficient domain
    intt_all_limbs(dm.d_data, L, N, ctx.ntt_tables);
    dm.is_ntt = false;

    // Per-limb delta scalars: ctx.delta is floor(q_0 / t) as uint64_t.
    // Each limb uses delta % q_l (delta < q_l in practice, so this equals delta).
    std::vector<uint64_t> delta_scalars_h(L);
    for (int l = 0; l < L; l++)
        delta_scalars_h[l] = ctx.delta % ctx.primes[l];

    uint64_t* d_delta = nullptr;
    cudaMalloc(&d_delta, (size_t)L * sizeof(uint64_t));
    cudaMemcpy(d_delta, delta_scalars_h.data(),
               (size_t)L * sizeof(uint64_t), cudaMemcpyHostToDevice);

    bfv_core::poly_scalar_mul_rns(dm.d_data, dm.d_data,
                                   d_delta, N, L, d_primes);
    cudaFree(d_delta);

    // NTT back → evaluation domain
    ntt_all_limbs(dm.d_data, L, N, ctx.ntt_tables);
    dm.is_ntt = true;

    // ── ct.c0 = pk.b * u + e1 + delta*m ─────────────────────────────────
    RnsPoly tmp;
    tmp.alloc(ctx, L);

    bfv_core::poly_mul_rns(tmp.d_data, pk.b.d_data, u.d_data,
                            N, L, d_primes);
    bfv_core::poly_add_rns(tmp.d_data, tmp.d_data, e1.d_data,
                            N, L, d_primes);
    bfv_core::poly_add_rns(ct.c0.d_data, tmp.d_data, dm.d_data,
                            N, L, d_primes);
    ct.c0.is_ntt = true;

    // ── ct.c1 = pk.a * u + e2 ────────────────────────────────────────────
    bfv_core::poly_mul_rns(tmp.d_data, pk.a.d_data, u.d_data,
                            N, L, d_primes);
    bfv_core::poly_add_rns(ct.c1.d_data, tmp.d_data, e2.d_data,
                            N, L, d_primes);
    ct.c1.is_ntt = true;

    // ── Cleanup ───────────────────────────────────────────────────────────
    cudaFree(d_primes);
    tmp.free();
    dm.free();
    e1.free();
    e2.free();
    u.free();
}

// ---------------------------------------------------------------------------
// bfv_decrypt
//
// phase = ct.c0 + ct.c1 * sk.s   (NTT domain, first L limbs of sk.s)
// INTT phase → coefficient domain.
// For each coefficient i, use limb 0 only:
//   center-lift mod q_0
//   round(coeff * t / q_0) via __int128 integer arithmetic
//   reduce mod t → pt_out.slots[i]
// ---------------------------------------------------------------------------
void bfv_decrypt(const BfvContext& ctx, const BfvSecretKey& sk,
                 const BfvCiphertext& ct, BfvPlaintext& pt_out) {
    const int L = ctx.L;
    const int N = ctx.N;

    uint64_t* d_primes = make_device_primes(ctx.primes, L);

    // ── phase = ct.c0 + ct.c1 * sk.s  (L limbs, NTT) ────────────────────
    // sk.s is L+K limbs; poly_mul_rns reads only the first L.
    RnsPoly tmp;
    tmp.alloc(ctx, L);
    bfv_core::poly_mul_rns(tmp.d_data, ct.c1.d_data, sk.s.d_data,
                            N, L, d_primes);

    RnsPoly phase;
    phase.alloc(ctx, L);
    bfv_core::poly_add_rns(phase.d_data, ct.c0.d_data, tmp.d_data,
                            N, L, d_primes);
    phase.is_ntt = true;

    tmp.free();
    cudaFree(d_primes);

    // ── INTT → coefficient domain ─────────────────────────────────────────
    intt_all_limbs(phase.d_data, L, N, ctx.ntt_tables);
    phase.is_ntt = false;

    // ── Copy limb 0 to host ───────────────────────────────────────────────
    // Only limb 0 is needed for decoding.
    std::vector<uint64_t> h_limb0(N);
    cudaMemcpy(h_limb0.data(), phase.d_data,
               (size_t)N * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    phase.free();

    // ── Decode coefficients ───────────────────────────────────────────────
    const uint64_t q0 = ctx.primes[0];
    const uint64_t t  = ctx.plain_mod;
    const int64_t  q0_half = (int64_t)(q0 / 2);

    pt_out.slots.resize(N);
    for (int i = 0; i < N; i++) {
        uint64_t raw = h_limb0[i];   // in [0, q0)

        // Center-lift: map to (-q0/2, q0/2]
        int64_t coeff = (raw > (uint64_t)q0_half)
                            ? (int64_t)(raw - q0)
                            : (int64_t)raw;

        // Round: val = floor((coeff * t + q0/2) / q0) using __int128
        // For valid decryption (|noise| < delta/2), numerator is non-negative.
        __int128 num = (__int128)coeff * (int64_t)t + q0_half;
        int64_t  val = (int64_t)(num / (int64_t)q0);

        // Reduce mod t to [0, t)
        int64_t t_s = (int64_t)t;
        val = ((val % t_s) + t_s) % t_s;

        pt_out.slots[i] = (uint64_t)val;
    }
}

// ---------------------------------------------------------------------------
// bfv_add
//
// out.c0 = a.c0 + b.c0  (mod Q, per-limb, NTT domain)
// out.c1 = a.c1 + b.c1
// ---------------------------------------------------------------------------
void bfv_add(const BfvContext& ctx,
             const BfvCiphertext& a, const BfvCiphertext& b,
             BfvCiphertext& out) {
    const int L = ctx.L;
    const int N = ctx.N;

    out.alloc(ctx);
    out.is_ntt = true;

    uint64_t* d_primes = make_device_primes(ctx.primes, L);

    bfv_core::poly_add_rns(out.c0.d_data, a.c0.d_data, b.c0.d_data,
                            N, L, d_primes);
    bfv_core::poly_add_rns(out.c1.d_data, a.c1.d_data, b.c1.d_data,
                            N, L, d_primes);
    out.c0.is_ntt = true;
    out.c1.is_ntt = true;

    cudaFree(d_primes);
}

} // namespace bfv
