#pragma once
// =============================================================================
// bfv_types.cuh — GPU data structures for BFV Layer 2
//
// All GPU memory is raw uint64_t* (cudaMalloc/cudaFree).
// Explicit lifetime management — no RAII, no smart pointers.
// Compatible with HECache (Layer 3) direct-pointer semantics.
// =============================================================================

#include "bfv_context.cuh"
#include <vector>
#include <string>
#include <map>
#include <cstdint>
#include <cuda_runtime.h>

namespace bfv {

// ---------------------------------------------------------------------------
// RnsPoly — single RNS polynomial in prime-major flat layout on GPU
//
//   d_data[l * N + i]  =  coefficient i of limb l
//
// L limbs for base-Q polys (ciphertexts, plaintexts, public keys).
// L+K limbs for key-switching material (secret key, galois keys).
// ---------------------------------------------------------------------------
struct RnsPoly {
    uint64_t* d_data;   // GPU pointer; size = L * N * sizeof(uint64_t)
    int       L, N;
    bool      is_ntt;   // true  = NTT domain,  false = coefficient domain

    // Allocate with ctx.L limbs (ciphertext / plaintext poly)
    void alloc(const BfvContext& ctx);
    // Allocate with explicit limb count (use ctx.L+ctx.K for key material)
    void alloc(const BfvContext& ctx, int num_limbs);
    void free();

    // In-place domain conversion (all L limbs)
    void to_ntt  (const BfvContext& ctx);
    void to_coeff(const BfvContext& ctx);

    // Host <-> device transfers (num_limbs * N elements)
    void copy_from_host(const uint64_t* h_src, int num_limbs);
    void copy_to_host  (uint64_t* h_dst) const;
};

// ---------------------------------------------------------------------------
// BfvCiphertext — (c0, c1) in NTT domain
// ---------------------------------------------------------------------------
struct BfvCiphertext {
    RnsPoly     c0, c1;
    std::string name;    // optional handle for HECache (Layer 3)
    bool        is_ntt;

    void alloc(const BfvContext& ctx);
    void free();
};

// ---------------------------------------------------------------------------
// BfvPlaintext — N slots mod t (CPU) + encoded RnsPoly (GPU, NTT domain)
// ---------------------------------------------------------------------------
struct BfvPlaintext {
    std::vector<uint64_t> slots;    // N values mod t  (CPU)
    RnsPoly               encoded;  // L limbs, NTT domain

    void alloc(const BfvContext& ctx);
    void free();
};

// ---------------------------------------------------------------------------
// BfvSecretKey — s(X), binary/ternary, NTT form, L+K limbs
// ---------------------------------------------------------------------------
struct BfvSecretKey {
    RnsPoly s;   // L+K limbs, NTT domain

    void alloc(const BfvContext& ctx);
    void free();
};

// ---------------------------------------------------------------------------
// BfvPublicKey — (b, a) where b = -(a*s + e) mod q
// ---------------------------------------------------------------------------
struct BfvPublicKey {
    RnsPoly b, a;   // each L limbs, NTT domain

    void alloc(const BfvContext& ctx);
    void free();
};

// ---------------------------------------------------------------------------
// GaloisKeyEntry — one entry of the galois key set (for a single galois elt)
// beta = decomposition digit count (= L for RNS decomposition)
// b[j], a[j] are in the extended QP base (L+K limbs each)
// ---------------------------------------------------------------------------
struct GaloisKeyEntry {
    int                  beta;
    std::vector<RnsPoly> b, a;   // each of length beta; each RnsPoly has L+K limbs

    void alloc(const BfvContext& ctx, int beta_in);
    void free();
};

// ---------------------------------------------------------------------------
// BfvGaloisKey — full galois key: map from galois_elt -> GaloisKeyEntry
// ---------------------------------------------------------------------------
struct BfvGaloisKey {
    std::map<uint32_t, GaloisKeyEntry> keys;
};

} // namespace bfv
