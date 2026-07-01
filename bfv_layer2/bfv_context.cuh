#pragma once
// =============================================================================
// bfv_context.cuh — BFV Layer 2 context (prime selection, NTT tables, delta)
//
// No FHE operations. Zero dependency on Phantom or SEAL.
// =============================================================================

#include "../bfv_core/bfv_params.cuh"
#include "../bfv_core/ntt.cuh"
#include <vector>
#include <cstdint>

namespace bfv {

static constexpr int MAX_PRIMES = bfv_core::MAX_RNS_PRIMES;  // 8

// ---------------------------------------------------------------------------
// BfvContext — immutable after creation; shared by all BFV operations
// ---------------------------------------------------------------------------
struct BfvContext {
    int      N;           // polynomial degree (power of 2)
    int      logN;        // log2(N)
    int      L;           // number of RNS base primes (coeff_modulus)
    int      K;           // number of special primes (key-switching; K=1 typical)
    uint64_t plain_mod;   // t  (plaintext modulus)
    uint64_t delta;       // floor(q_0 / t) — L=1 approximation; stored for encrypt

    // NTT tables, one per prime.  ntt_tables[0..L-1] = base, [L..L+K-1] = special.
    // Host structs; each contains device pointers (d_roots, d_roots_inv).
    std::vector<bfv_core::NttTable> ntt_tables;   // length L+K

    uint64_t primes[MAX_PRIMES];       // base primes  q_0 .. q_{L-1}
    uint64_t special_primes[4];        // special primes P_0 .. P_{K-1}

    // Factory: pick first L primes from DEFAULT_PRIMES as base, next K as special.
    // Scales the precomputed 2*32768-th root down to 2N-th root for each prime.
    static BfvContext create(int N, int L, int K, uint64_t plain_mod);

    // Release device memory (NTT root tables).
    void destroy();

    void print() const;
};

} // namespace bfv
