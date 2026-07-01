// =============================================================================
// bfv_types.cu — GPU struct allocation / deallocation / NTT domain helpers
// =============================================================================

#include "bfv_types.cuh"
#include "../bfv_core/ntt.cuh"
#include <cassert>
#include <cstring>

namespace bfv {

// ── RnsPoly ─────────────────────────────────────────────────────────────────

void RnsPoly::alloc(const BfvContext& ctx) {
    alloc(ctx, ctx.L);
}

void RnsPoly::alloc(const BfvContext& ctx, int num_limbs) {
    L      = num_limbs;
    N      = ctx.N;
    is_ntt = false;
    cudaMalloc(&d_data, (size_t)L * N * sizeof(uint64_t));
}

void RnsPoly::free() {
    if (d_data) { cudaFree(d_data); d_data = nullptr; }
    L = N = 0;
}

void RnsPoly::to_ntt(const BfvContext& ctx) {
    assert(!is_ntt && "RnsPoly::to_ntt called on already-NTT poly");
    assert(L <= (int)ctx.ntt_tables.size());
    for (int l = 0; l < L; l++)
        bfv_core::ntt_forward_single(d_data + (size_t)l * N, ctx.ntt_tables[l]);
    cudaDeviceSynchronize();
    is_ntt = true;
}

void RnsPoly::to_coeff(const BfvContext& ctx) {
    assert(is_ntt && "RnsPoly::to_coeff called on non-NTT poly");
    assert(L <= (int)ctx.ntt_tables.size());
    for (int l = 0; l < L; l++)
        bfv_core::ntt_inverse_single(d_data + (size_t)l * N, ctx.ntt_tables[l]);
    cudaDeviceSynchronize();
    is_ntt = false;
}

void RnsPoly::copy_from_host(const uint64_t* h_src, int num_limbs) {
    cudaMemcpy(d_data, h_src,
               (size_t)num_limbs * N * sizeof(uint64_t),
               cudaMemcpyHostToDevice);
}

void RnsPoly::copy_to_host(uint64_t* h_dst) const {
    cudaMemcpy(h_dst, d_data,
               (size_t)L * N * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
}

// ── BfvCiphertext ────────────────────────────────────────────────────────────

void BfvCiphertext::alloc(const BfvContext& ctx) {
    c0.alloc(ctx);
    c1.alloc(ctx);
    is_ntt = false;
}

void BfvCiphertext::free() {
    c0.free();
    c1.free();
}

// ── BfvPlaintext ─────────────────────────────────────────────────────────────

void BfvPlaintext::alloc(const BfvContext& ctx) {
    slots.resize(ctx.N, 0);
    encoded.alloc(ctx);  // L limbs
}

void BfvPlaintext::free() {
    slots.clear();
    encoded.free();
}

// ── BfvSecretKey ─────────────────────────────────────────────────────────────

void BfvSecretKey::alloc(const BfvContext& ctx) {
    s.alloc(ctx, ctx.L + ctx.K);  // extended base for key-switching
}

void BfvSecretKey::free() {
    s.free();
}

// ── BfvPublicKey ─────────────────────────────────────────────────────────────

void BfvPublicKey::alloc(const BfvContext& ctx) {
    b.alloc(ctx);
    a.alloc(ctx);
}

void BfvPublicKey::free() {
    b.free();
    a.free();
}

// ── GaloisKeyEntry ───────────────────────────────────────────────────────────

void GaloisKeyEntry::alloc(const BfvContext& ctx, int beta_in) {
    beta = beta_in;
    b.resize(beta);
    a.resize(beta);
    for (int j = 0; j < beta; j++) {
        b[j].alloc(ctx, ctx.L + ctx.K);
        a[j].alloc(ctx, ctx.L + ctx.K);
    }
}

void GaloisKeyEntry::free() {
    for (auto& p : b) p.free();
    for (auto& p : a) p.free();
    b.clear();
    a.clear();
    beta = 0;
}

} // namespace bfv
