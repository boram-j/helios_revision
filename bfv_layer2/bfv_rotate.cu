// =============================================================================
// bfv_rotate.cu — BFV Galois rotation: automorphism + key-switch
//
// Algorithm
// ---------
// Given ciphertext (c0, c1) encrypting m(X), rotate to produce
// (c0', c1') encrypting σ_k(m)(X) = m(X^k) where k = galois_elt.
//
// Step 1 — Galois automorphism on both components:
//   c0_gal = σ_k(c0),  c1_gal = σ_k(c1)
//   Done in coefficient domain (INTT → apply_galois_inplace → NTT).
//
// Step 2 — Key-switch σ_k(c1) using GaloisKeyEntry gke:
//   gke has beta=L digit pairs (b[j], a[j]), each L+K limbs, NTT form.
//   Simple RNS digit decomposition: digit j = limb j of c1_gal,
//   zero-extended to L+K limbs.
//   Accumulate:
//     ks_acc_b += gke.b[j] ⊙ digit_j   (ks_ip_single, L+K limbs)
//     ks_acc_a += gke.a[j] ⊙ digit_j
//
// Step 3 — ModDown:
//   rns_moddown: (L+K)*N → L*N  (K=1 required)
//   Results acc_b_down, acc_a_down are L-limb NTT-domain polynomials.
//
// Step 4 — Output:
//   ct_out.c0 = c0_gal + acc_b_down   (mod Q, NTT)
//   ct_out.c1 = acc_a_down            (mod Q, NTT)
//
// Decryption: ct_out.c0 + ct_out.c1 * s ≈ σ_k(c0 + c1*s) + small noise.
// =============================================================================

#include "bfv_rotate.cuh"
#include "../bfv_core/galois.cuh"
#include "../bfv_core/ntt.cuh"
#include "../bfv_core/rns.cuh"
#include "../bfv_core/ks_ip.cuh"
#include "../bfv_core/poly_ops.cuh"

#include <vector>
#include <cstring>
#include <cuda_runtime.h>

namespace bfv {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static uint64_t* make_device_primes(const uint64_t* h, int n) {
    uint64_t* d = nullptr;
    cudaMalloc(&d, (size_t)n * sizeof(uint64_t));
    cudaMemcpy(d, h, (size_t)n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    return d;
}

// Forward NTT on num_limbs (uses ctx.ntt_tables[0..num_limbs-1])
static void ntt_fwd(uint64_t* d_data, int num_limbs, int N,
                    const std::vector<bfv_core::NttTable>& tables) {
    for (int l = 0; l < num_limbs; l++)
        bfv_core::ntt_forward_single(d_data + (size_t)l * N, tables[l]);
    cudaDeviceSynchronize();
}

// Inverse NTT on num_limbs (uses ctx.ntt_tables[0..num_limbs-1])
static void ntt_inv(uint64_t* d_data, int num_limbs, int N,
                    const std::vector<bfv_core::NttTable>& tables) {
    for (int l = 0; l < num_limbs; l++)
        bfv_core::ntt_inverse_single(d_data + (size_t)l * N, tables[l]);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// bfv_rotate
// ---------------------------------------------------------------------------
void bfv_rotate(const BfvContext&    ctx,
                const GaloisKeyEntry& gke,
                uint32_t             galois_elt,
                const BfvCiphertext& ct_in,
                BfvCiphertext&       ct_out)
{
    const int L  = ctx.L;
    const int K  = ctx.K;
    const int LK = L + K;
    const int N  = ctx.N;

    // Build combined prime list [q_0..q_{L-1}, p_0..p_{K-1}]
    std::vector<uint64_t> all_primes_h(LK);
    for (int l = 0; l < L; l++) all_primes_h[l]     = ctx.primes[l];
    for (int k = 0; k < K; k++) all_primes_h[L + k] = ctx.special_primes[k];

    uint64_t* d_primes_base = make_device_primes(ctx.primes,        L);
    uint64_t* d_primes_all  = make_device_primes(all_primes_h.data(), LK);

    // ── Step 1: Galois automorphism on c0 and c1 ──────────────────────────────
    //
    // Copy each component, INTT to coefficient domain, apply σ_k in-place,
    // then NTT back.  c1_gal is kept until after the key-switch loop.

    RnsPoly c0_gal;
    c0_gal.alloc(ctx, L);
    cudaMemcpy(c0_gal.d_data, ct_in.c0.d_data,
               (size_t)L * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    c0_gal.is_ntt = true;

    ntt_inv(c0_gal.d_data, L, N, ctx.ntt_tables);
    c0_gal.is_ntt = false;
    bfv_core::apply_galois_inplace(c0_gal.d_data, galois_elt, N, L, d_primes_base);
    cudaDeviceSynchronize();
    ntt_fwd(c0_gal.d_data, L, N, ctx.ntt_tables);
    c0_gal.is_ntt = true;

    RnsPoly c1_gal;
    c1_gal.alloc(ctx, L);
    cudaMemcpy(c1_gal.d_data, ct_in.c1.d_data,
               (size_t)L * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    c1_gal.is_ntt = true;

    ntt_inv(c1_gal.d_data, L, N, ctx.ntt_tables);
    c1_gal.is_ntt = false;
    bfv_core::apply_galois_inplace(c1_gal.d_data, galois_elt, N, L, d_primes_base);
    cudaDeviceSynchronize();
    ntt_fwd(c1_gal.d_data, L, N, ctx.ntt_tables);
    c1_gal.is_ntt = true;

    // ── Step 2: Key-switch c1_gal ──────────────────────────────────────────────
    //
    // Accumulators (L+K limbs, zero-init, NTT domain).
    RnsPoly ks_acc_b, ks_acc_a;
    ks_acc_b.alloc(ctx, LK);
    ks_acc_a.alloc(ctx, LK);
    cudaMemset(ks_acc_b.d_data, 0, (size_t)LK * N * sizeof(uint64_t));
    cudaMemset(ks_acc_a.d_data, 0, (size_t)LK * N * sizeof(uint64_t));
    ks_acc_b.is_ntt = true;
    ks_acc_a.is_ntt = true;

    // Digit buffer: (L+K) limbs; only limb j non-zero for digit j.
    RnsPoly digit;
    digit.alloc(ctx, LK);
    digit.is_ntt = true;

    for (int j = 0; j < L; j++) {
        // Zero entire buffer, then copy limb j of c1_gal into limb j of digit.
        cudaMemset(digit.d_data, 0, (size_t)LK * N * sizeof(uint64_t));
        cudaMemcpy(digit.d_data + (size_t)j * N,
                   c1_gal.d_data + (size_t)j * N,
                   (size_t)N * sizeof(uint64_t),
                   cudaMemcpyDeviceToDevice);

        // ks_acc_b += gke.b[j] ⊙ digit_j  (mod each of the L+K primes)
        bfv_core::ks_ip_single(gke.b[j].d_data, digit.d_data,
                               ks_acc_b.d_data, N, LK, d_primes_all);
        // ks_acc_a += gke.a[j] ⊙ digit_j
        bfv_core::ks_ip_single(gke.a[j].d_data, digit.d_data,
                               ks_acc_a.d_data, N, LK, d_primes_all);
    }
    cudaDeviceSynchronize();

    digit.free();
    c1_gal.free();

    // ── Step 3: ModDown (L+K) → L limbs ───────────────────────────────────────
    //
    // Requires K == 1 (rns_moddown only implements the single-special-prime case).
    bfv_core::RnsModDownParams mdp =
        bfv_core::rns_moddown_params_create(ctx.primes, L, ctx.special_primes[0]);

    RnsPoly acc_b_down, acc_a_down;
    acc_b_down.alloc(ctx, L);
    acc_a_down.alloc(ctx, L);

    bfv_core::rns_moddown(acc_b_down.d_data, ks_acc_b.d_data, N, mdp);
    bfv_core::rns_moddown(acc_a_down.d_data, ks_acc_a.d_data, N, mdp);
    cudaDeviceSynchronize();

    bfv_core::rns_moddown_params_free(mdp);
    ks_acc_b.free();
    ks_acc_a.free();

    // ── Step 4: Assemble output ciphertext ─────────────────────────────────────

    ct_out.alloc(ctx);   // allocates c0 and c1 with L limbs each
    ct_out.is_ntt = true;

    // ct_out.c0 = c0_gal + acc_b_down   (mod Q, NTT domain)
    bfv_core::poly_add_rns(ct_out.c0.d_data,
                            c0_gal.d_data, acc_b_down.d_data,
                            N, L, d_primes_base);
    ct_out.c0.is_ntt = true;

    // ct_out.c1 = acc_a_down
    cudaMemcpy(ct_out.c1.d_data, acc_a_down.d_data,
               (size_t)L * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    ct_out.c1.is_ntt = true;

    // ── Cleanup ────────────────────────────────────────────────────────────────
    cudaFree(d_primes_base);
    cudaFree(d_primes_all);
    c0_gal.free();
    acc_b_down.free();
    acc_a_down.free();
}

} // namespace bfv
