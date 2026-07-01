// =============================================================================
// bfv_context.cu — BfvContext factory implementation
// =============================================================================

#include "bfv_context.cuh"
#include <cstdio>
#include <cassert>

namespace bfv {

BfvContext BfvContext::create(int N, int L, int K, uint64_t plain_mod) {
    assert(L >= 1 && K >= 0 && L + K <= MAX_PRIMES);

    BfvContext ctx;
    ctx.N         = N;
    ctx.L         = L;
    ctx.K         = K;
    ctx.plain_mod = plain_mod;

    // Compute logN
    ctx.logN = 0;
    for (int tmp = N; tmp > 1; tmp >>= 1) ctx.logN++;

    // Fill prime arrays from the Layer-1 precomputed table
    for (int i = 0; i < MAX_PRIMES; i++) ctx.primes[i]         = 0;
    for (int i = 0; i < 4;          i++) ctx.special_primes[i] = 0;
    for (int i = 0; i < L;          i++) ctx.primes[i]         = bfv_core::DEFAULT_PRIMES[i];
    for (int i = 0; i < K;          i++) ctx.special_primes[i] = bfv_core::DEFAULT_PRIMES[L + i];

    // delta = floor(q_0 / t)  — approximation; exact only for L=1
    ctx.delta = ctx.primes[0] / plain_mod;

    // Build NTT tables for each of L+K primes.
    //
    // DEFAULT_PRIM_ROOTS_2N[i] is the primitive 2*N_max-th root for N_max=32768.
    // For a target degree N, the primitive 2N-th root is:
    //   psi_N = psi_{N_max}^(N_max / N) mod p
    // because psi_{N_max}^(2*N_max) = 1  =>  (psi_N)^(2N) = psi_{N_max}^(2*N_max) = 1 ✓
    // and     psi_{N_max}^(N_max)   = -1 =>  (psi_N)^N    = psi_{N_max}^(N_max)   = -1 ✓
    const int N_max = bfv_core::MAX_POLY_DEG;
    assert(N <= N_max && (N_max % N == 0));

    ctx.ntt_tables.resize(L + K);
    for (int i = 0; i < L + K; i++) {
        uint64_t p       = (i < L) ? ctx.primes[i] : ctx.special_primes[i - L];
        uint64_t psi_max = bfv_core::DEFAULT_PRIM_ROOTS_2N[i];
        uint64_t psi_N   = bfv_core::powmod(psi_max, (uint64_t)(N_max / N), p);
        ctx.ntt_tables[i] = bfv_core::ntt_table_create(p, psi_N, N);
    }

    return ctx;
}

void BfvContext::destroy() {
    for (auto& tbl : ntt_tables) bfv_core::ntt_table_free(tbl);
    ntt_tables.clear();
}

void BfvContext::print() const {
    printf("BfvContext: N=%d  logN=%d  L=%d  K=%d\n", N, logN, L, K);
    printf("  plain_mod = %llu\n",  (unsigned long long)plain_mod);
    printf("  delta     = %llu\n",  (unsigned long long)delta);
    for (int i = 0; i < L; i++)
        printf("  q[%d] = 0x%llx\n", i, (unsigned long long)primes[i]);
    for (int i = 0; i < K; i++)
        printf("  P[%d] = 0x%llx\n", i, (unsigned long long)special_primes[i]);
}

} // namespace bfv
