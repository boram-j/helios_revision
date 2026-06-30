// =============================================================================
// verify_layer1.cu — BFV Layer 1 Correctness Verifier
//
// Standalone CUDA program with 5 tests:
//   1. NTT roundtrip:     random poly → NTT → INTT → check == original
//   2. Poly mul:          two small polys × schoolbook reference (mod prime)
//   3. ModUp/ModDown:     lift to extended base, reduce, check == original
//   4. KS-IP batch:       B=4 batch vs. sequential scalar reference
//   5. Galois roundtrip:  σ_k then σ_{k^{-1}}, check == original
//
// Tests use small N (16 or 64) and small well-tested primes for easy verification.
// Large-N tests (N=32768) are also run where practical for NTT.
//
// Usage:  ./bfv_layer1_verify [--large]
//   --large: also test N=32768 with 60-bit primes (default: skip)
// =============================================================================

#include "../bfv_params.cuh"
#include "../ntt.cuh"
#include "../poly_ops.cuh"
#include "../rns.cuh"
#include "../ks_ip.cuh"
#include "../galois.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <random>
#include <string>
#include <functional>

using namespace bfv_core;

// ---------------------------------------------------------------------------
// CUDA error check macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Test result tracking
// ---------------------------------------------------------------------------
static int g_pass = 0, g_fail = 0;

static void report(const char* name, bool ok, const char* detail = "") {
    if (ok) {
        printf("  [PASS] %s%s%s\n", name, detail[0] ? " — " : "", detail);
        g_pass++;
    } else {
        printf("  [FAIL] %s%s%s\n", name, detail[0] ? " — " : "", detail);
        g_fail++;
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

// Upload host vector to device
static uint64_t* h2d(const std::vector<uint64_t>& v) {
    uint64_t* d;
    CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    return d;
}

// Download device array to host vector
static std::vector<uint64_t> d2h(const uint64_t* d, size_t n) {
    std::vector<uint64_t> v(n);
    CUDA_CHECK(cudaMemcpy(v.data(), d, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return v;
}

// Fill host vector with random values in [0, p)
static void rand_poly(std::vector<uint64_t>& v, uint64_t p, std::mt19937_64& rng) {
    std::uniform_int_distribution<uint64_t> dist(0, p - 1);
    for (auto& x : v) x = dist(rng);
}

// Host-side schoolbook poly multiply mod p in Z[X]/(X^N + 1)
static std::vector<uint64_t> schoolbook_mul(const std::vector<uint64_t>& a,
                                             const std::vector<uint64_t>& b,
                                             uint64_t p)
{
    int N = (int)a.size();
    std::vector<uint64_t> c(N, 0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            uint64_t prod = (uint64_t)((__uint128_t)a[i] * b[j] % p);
            int idx = i + j;
            if (idx < N) {
                c[idx] = (c[idx] + prod) % p;
            } else {
                // X^N ≡ -1, so coefficient wraps with negation
                c[idx - N] = (c[idx - N] + p - prod) % p;
            }
        }
    }
    return c;
}

// Max coefficient difference between two polynomials
static uint64_t max_diff(const std::vector<uint64_t>& a,
                          const std::vector<uint64_t>& b)
{
    uint64_t mx = 0;
    for (size_t i = 0; i < a.size(); i++) {
        uint64_t d = (a[i] >= b[i]) ? a[i] - b[i] : b[i] - a[i];
        if (d > mx) mx = d;
    }
    return mx;
}

// Count mismatches
static int count_mismatch(const std::vector<uint64_t>& a,
                            const std::vector<uint64_t>& b)
{
    int cnt = 0;
    for (size_t i = 0; i < a.size(); i++)
        if (a[i] != b[i]) cnt++;
    return cnt;
}

// =============================================================================
// TEST 1: NTT Roundtrip
// =============================================================================
static void test_ntt_roundtrip()
{
    printf("\n[Test 1] NTT Roundtrip\n");

    // p=786433 = 3*2^18+1, primitive root=10.
    // ntt_table_create takes psi = primitive 2N-th root = 10^((p-1)/(2N)) mod p.
    // We test N=16, 64, 1024.  N=32768 only with --large.

    // p=97, primitive root=5: 97-1=96=2^5*3, so NTT-friendly for N up to 2^4=16.
    // psi_tiny = 5^(96/32) mod 97 = 5^3 mod 97 = 125 mod 97 = 28
    // Verify: 28^16 mod 97 = 1, 28^8 = -1 mod 97?
    // 28^2=784 mod97=784-8*97=784-776=8; 8^2=64; 64^2=4096 mod97=4096-42*97=4096-4074=22;
    // 22^2=484 mod97=484-4*97=484-388=96=-1 mod 97. YES! 28 is prim 32nd root.

    uint64_t p_small = 786433;
    uint64_t g_full  = 10;   // primitive root of 786433
    // psi for each N: g_full^((p-1)/(2N)) mod p
    auto make_psi = [&](int N_val) {
        return powmod(g_full, (p_small - 1) / (uint64_t)(2 * N_val), p_small);
    };

    struct Case { int N; uint64_t p; uint64_t psi; };
    std::vector<Case> cases = {
        {16,   97,      28ULL},             // psi=28 is prim 32nd root of unity mod 97
        {16,   786433,  make_psi(16)},
        {64,   786433,  make_psi(64)},
        {1024, 786433,  make_psi(1024)},
        {32768, 1152921504606584833ULL, 683635763366557341ULL},  // 60-bit
    };

    std::mt19937_64 rng(42);

    for (auto& cas : cases) {
        int N = cas.N;
        uint64_t p = cas.p;
        uint64_t psi = cas.psi;

        if (N > 4096) continue;  // skip large by default

        NttTable tbl = ntt_table_create(p, psi, N);

        // Random input
        std::vector<uint64_t> h_poly(N);
        rand_poly(h_poly, p, rng);
        std::vector<uint64_t> h_orig = h_poly;

        uint64_t* d_poly = h2d(h_poly);

        // Forward then inverse NTT
        ntt_forward_single(d_poly, tbl);
        CUDA_CHECK(cudaDeviceSynchronize());
        ntt_inverse_single(d_poly, tbl);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto h_result = d2h(d_poly, N);
        int mismatches = count_mismatch(h_result, h_orig);

        char detail[64];
        snprintf(detail, sizeof(detail), "N=%d p=%llu mismatches=%d",
                 N, (unsigned long long)p, mismatches);
        report("NTT roundtrip", mismatches == 0, detail);

        cudaFree(d_poly);
        ntt_table_free(tbl);
    }
}

// =============================================================================
// TEST 2: Polynomial Multiplication vs. Schoolbook
// =============================================================================
static void test_poly_mul()
{
    printf("\n[Test 2] Poly Mul vs Schoolbook\n");

    // Use N=8 and small prime 97 for hand-verifiable results,
    // then N=64 with prime 786433.
    struct Case { int N; uint64_t p; uint64_t g_full; };
    std::vector<Case> cases = {
        {8,  97,     0},     // 97-1=96=2^5*3, prim root=5, so 2^5|p-1, N=8 OK
        {64, 786433, 10},
    };
    // g_full for p=97: 5 (primitive root). g_2N = 5^(96/16) = 5^6 mod 97.
    cases[0].g_full = 5;

    std::mt19937_64 rng(123);

    for (auto& cas : cases) {
        int N = cas.N;
        uint64_t p = cas.p;
        uint64_t g = powmod(cas.g_full, (p - 1) / (2 * N), p);

        NttTable tbl = ntt_table_create(p, g, N);

        // Random inputs (keep small so schoolbook is easy to verify)
        std::vector<uint64_t> h_a(N), h_b(N);
        rand_poly(h_a, p, rng);
        rand_poly(h_b, p, rng);

        // GPU: NTT(a) * NTT(b) → INTT = a*b in Z[X]/(X^N+1)
        uint64_t* d_a = h2d(h_a);
        uint64_t* d_b = h2d(h_b);
        uint64_t* d_c;
        CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(uint64_t)));

        ntt_forward_single(d_a, tbl);
        ntt_forward_single(d_b, tbl);
        CUDA_CHECK(cudaDeviceSynchronize());

        poly_mul_single(d_c, d_a, d_b, N, p);
        CUDA_CHECK(cudaDeviceSynchronize());

        ntt_inverse_single(d_c, tbl);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto h_gpu = d2h(d_c, N);

        // CPU schoolbook reference
        auto h_ref = schoolbook_mul(h_a, h_b, p);

        int mismatches = count_mismatch(h_gpu, h_ref);

        char detail[80];
        snprintf(detail, sizeof(detail), "N=%d p=%llu mismatches=%d",
                 N, (unsigned long long)p, mismatches);
        report("Poly mul (NTT vs schoolbook)", mismatches == 0, detail);

        if (mismatches > 0 && N <= 8) {
            printf("    GPU result:  "); for (auto x : h_gpu) printf("%llu ", (unsigned long long)x); printf("\n");
            printf("    CPU result:  "); for (auto x : h_ref) printf("%llu ", (unsigned long long)x); printf("\n");
        }

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        ntt_table_free(tbl);
    }
}

// =============================================================================
// TEST 3: ModUp / ModDown Roundtrip
// =============================================================================
static void test_modup_moddown()
{
    printf("\n[Test 3] ModUp / ModDown\n");
    // Math note: ModDown computes b = (a - a_P) / P mod Q.
    // ModUp followed by ModDown is NOT the identity — it divides by P.
    // We run two separate sub-tests:
    //   3a: ModUp extension — verify FastBConv gives correct a mod P
    //   3b: ModDown correctness — verify (P * b) / P = b exactly

    int N = 16;
    uint64_t q0 = 97, q1 = 101, P = 103;
    uint64_t q_primes[2] = {q0, q1};

    RnsModUpParams   up_p   = rns_modup_params_create(q_primes, 2, &P, 1);
    RnsModDownParams down_p = rns_moddown_params_create(q_primes, 2, P);

    std::mt19937_64 rng(999);
    char detail[128];

    // ── 3a: ModUp extension accuracy ───────────────────────────────────────
    // FastBConv (approximate base extension) computes:
    //   ext[j] = sum_i x_i * (Q/q_i mod P) mod P
    //   where x_i = a_i * (Q/q_i)^{-1} mod q_i.
    // This equals a mod P + e*(Q mod P) for integer e ∈ {0,..,L-1} (approximation).
    // We compute the exact same formula on the host as the reference and verify the
    // GPU gives an identical result (tests kernel correctness, not mathematical exactness).
    {
        std::uniform_int_distribution<uint64_t> dist(0, 40);
        std::vector<uint64_t> h_a0(N), h_a1(N), expected_P(N);

        // Host-side FastBConv precomputed constants (mirrors rns_modup_params_create)
        uint64_t hat_inv_q0 = invmod(q1 % q0, q0);  // (Q/q0)^{-1} mod q0 = q1^{-1} mod q0
        uint64_t hat_inv_q1 = invmod(q0 % q1, q1);  // (Q/q1)^{-1} mod q1 = q0^{-1} mod q1
        uint64_t hat_q0_P   = q1 % P;               // (Q/q0) mod P
        uint64_t hat_q1_P   = q0 % P;               // (Q/q1) mod P

        for (int i = 0; i < N; i++) {
            uint64_t c  = dist(rng);
            h_a0[i]     = c % q0;
            h_a1[i]     = c % q1;
            // Reference FastBConv output (matches k_modup_step1 + k_modup_step2)
            uint64_t x0 = mulmod(h_a0[i], hat_inv_q0, q0);
            uint64_t x1 = mulmod(h_a1[i], hat_inv_q1, q1);
            expected_P[i] = (uint64_t)(((__uint128_t)x0 * hat_q0_P +
                                        (__uint128_t)x1 * hat_q1_P) % P);
        }

        std::vector<uint64_t> h_in(2 * N);
        for (int i = 0; i < N; i++) { h_in[i] = h_a0[i]; h_in[N + i] = h_a1[i]; }

        uint64_t* d_in = h2d(h_in);
        uint64_t* d_qp;
        CUDA_CHECK(cudaMalloc(&d_qp, 3 * N * sizeof(uint64_t)));

        rns_modup(d_qp, d_in, N, up_p);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto h_qp = d2h(d_qp, 3 * N);
        int bad = 0;
        for (int i = 0; i < N; i++)
            if (h_qp[2 * N + i] != expected_P[i]) bad++;

        snprintf(detail, sizeof(detail),
                 "FastBConv: N=%d q=[%llu,%llu] P=%llu gpu_vs_host_mismatches=%d",
                 N, (unsigned long long)q0, (unsigned long long)q1,
                 (unsigned long long)P, bad);
        report("ModUp extension (FastBConv gpu==host ref)", bad == 0, detail);

        cudaFree(d_in);
        cudaFree(d_qp);
    }

    // ── 3b: ModDown correctness: (P * b) / P = b ───────────────────────────
    // Input a = P * b where b has small coefficients.
    // a_P = (P * b) mod P = 0, so correction = 0.
    // b_i = (a_i - 0) * P^{-1} mod q_i = (P*b mod q_i) * P^{-1} mod q_i = b mod q_i.
    {
        std::uniform_int_distribution<uint64_t> dist_b(0, 30);
        std::vector<uint64_t> h_b(N);
        for (auto& x : h_b) x = dist_b(rng);   // b coefficients in [0,30]

        // a = P * b: a_qi = (P * b[i]) mod qi, a_P = 0
        std::vector<uint64_t> h_in(3 * N, 0);  // [a mod q0 | a mod q1 | a mod P=0]
        for (int i = 0; i < N; i++) {
            h_in[i]             = (uint64_t)((__uint128_t)P * h_b[i] % q0);
            h_in[N + i]         = (uint64_t)((__uint128_t)P * h_b[i] % q1);
            h_in[2 * N + i]     = 0;   // P * b ≡ 0 mod P
        }

        uint64_t* d_qp  = h2d(h_in);
        uint64_t* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, 2 * N * sizeof(uint64_t)));

        rns_moddown(d_out, d_qp, N, down_p);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto h_out = d2h(d_out, 2 * N);
        int bad = 0;
        for (int i = 0; i < N; i++) {
            uint64_t exp0 = h_b[i] % q0;
            uint64_t exp1 = h_b[i] % q1;
            if (h_out[i] != exp0 || h_out[N + i] != exp1) bad++;
        }

        snprintf(detail, sizeof(detail),
                 "ModDown((P*b) mod QP): N=%d b_range=[0,30] mismatches=%d", N, bad);
        report("ModDown correctness (exact division)", bad == 0, detail);

        if (bad > 0) {
            printf("    b:       "); for(int i=0;i<N;i++) printf("%llu ",h_b[i]); printf("\n");
            printf("    got q0:  "); for(int i=0;i<N;i++) printf("%llu ",h_out[i]); printf("\n");
            printf("    want q0: "); for(int i=0;i<N;i++) printf("%llu ",(uint64_t)(h_b[i]%q0)); printf("\n");
        }

        cudaFree(d_qp);
        cudaFree(d_out);
    }

    rns_modup_params_free(up_p);
    rns_moddown_params_free(down_p);
}

// =============================================================================
// TEST 4: KS-IP Batch vs. Sequential Scalar Reference
// =============================================================================
static void test_ks_ip()
{
    printf("\n[Test 4] KS-IP Batch vs Sequential Reference\n");

    int N         = 64;
    int num_limbs = 2;
    int B         = 4;

    uint64_t q_primes[2] = {97, 101};

    // Build a device primes array
    uint64_t* d_primes = h2d({97ULL, 101ULL});

    std::mt19937_64 rng(555);

    // Random key digit polynomial
    std::vector<uint64_t> h_kd(num_limbs * N);
    for (int l = 0; l < num_limbs; l++) {
        uint64_t p = q_primes[l];
        for (int i = 0; i < N; i++)
            h_kd[l * N + i] = std::uniform_int_distribution<uint64_t>(0, p-1)(rng);
    }
    uint64_t* d_kd = h2d(h_kd);

    // B ciphertext polynomials
    std::vector<std::vector<uint64_t>> h_cts(B, std::vector<uint64_t>(num_limbs * N));
    std::vector<uint64_t*> d_ct_ptrs(B);
    for (int b = 0; b < B; b++) {
        for (int l = 0; l < num_limbs; l++) {
            uint64_t p = q_primes[l];
            for (int i = 0; i < N; i++)
                h_cts[b][l * N + i] = std::uniform_int_distribution<uint64_t>(0, p-1)(rng);
        }
        d_ct_ptrs[b] = h2d(h_cts[b]);
    }

    // Accumulators initialized to zero
    std::vector<uint64_t*> d_acc_ptrs(B);
    std::vector<std::vector<uint64_t>> h_accs_init(B, std::vector<uint64_t>(num_limbs * N, 0));
    for (int b = 0; b < B; b++)
        d_acc_ptrs[b] = h2d(h_accs_init[b]);

    // Upload pointer arrays
    const uint64_t** d_ct_arr = make_d_ptr_array(
        std::vector<const uint64_t*>(d_ct_ptrs.begin(), d_ct_ptrs.end()));
    uint64_t** d_acc_arr = make_d_ptr_array_mutable(d_acc_ptrs);

    // === GPU batch KS-IP ===
    ks_ip_batch(d_kd, d_ct_arr, d_acc_arr, B, N, num_limbs, d_primes);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download GPU results
    std::vector<std::vector<uint64_t>> h_gpu(B);
    for (int b = 0; b < B; b++)
        h_gpu[b] = d2h(d_acc_ptrs[b], num_limbs * N);

    // === CPU sequential reference ===
    // acc_ref[b][l*N+n] = (key_digit[l*N+n] * ct[b][l*N+n]) mod q[l]
    std::vector<std::vector<uint64_t>> h_ref(B, std::vector<uint64_t>(num_limbs * N, 0));
    for (int b = 0; b < B; b++) {
        for (int l = 0; l < num_limbs; l++) {
            uint64_t p = q_primes[l];
            for (int n = 0; n < N; n++) {
                size_t off = (size_t)l * N + n;
                h_ref[b][off] = (uint64_t)((__uint128_t)h_kd[off] * h_cts[b][off] % p);
            }
        }
    }

    // Compare
    int total_bad = 0;
    for (int b = 0; b < B; b++) {
        total_bad += count_mismatch(h_gpu[b], h_ref[b]);
    }

    char detail[64];
    snprintf(detail, sizeof(detail),
             "B=%d N=%d L=%d total_mismatches=%d", B, N, num_limbs, total_bad);
    report("KS-IP batch vs sequential", total_bad == 0, detail);

    // Cleanup
    cudaFree(d_kd);
    cudaFree(d_primes);
    for (int b = 0; b < B; b++) { cudaFree(d_ct_ptrs[b]); cudaFree(d_acc_ptrs[b]); }
    cudaFree((void*)d_ct_arr);
    cudaFree(d_acc_arr);
}

// =============================================================================
// TEST 5: Galois Roundtrip (σ_k then σ_{k^{-1}} == identity)
// =============================================================================
static void test_galois()
{
    printf("\n[Test 5] Galois Automorphism Roundtrip\n");

    int N = 16;
    int L = 2;
    uint64_t q_primes[2] = {97, 101};

    // Test with galois_elt = 3 (must be odd and < 2N = 32)
    // 3 * 3 mod 32 = 9, 3 * 11 = 33 mod 32 = 1, so 3^{-1} = 11 mod 32
    uint32_t k = 3;
    uint32_t k_inv = galois_inverse(k, N);  // should be 11

    std::mt19937_64 rng(777);
    std::vector<uint64_t> h_poly(L * N);
    for (int l = 0; l < L; l++) {
        uint64_t p = q_primes[l];
        for (int i = 0; i < N; i++)
            h_poly[l * N + i] = std::uniform_int_distribution<uint64_t>(0, p-1)(rng);
    }
    auto h_orig = h_poly;

    uint64_t* d_primes = h2d({97ULL, 101ULL});
    uint64_t* d_poly   = h2d(h_poly);
    uint64_t* d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, L * N * sizeof(uint64_t)));

    // Apply σ_k then σ_{k^{-1}}
    apply_galois(d_poly, k,     d_tmp,  N, L, d_primes);
    CUDA_CHECK(cudaDeviceSynchronize());
    apply_galois(d_tmp,  k_inv, d_poly, N, L, d_primes);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto h_result = d2h(d_poly, L * N);
    int mismatches = count_mismatch(h_result, h_orig);

    char detail[80];
    snprintf(detail, sizeof(detail),
             "N=%d L=%d k=%u k_inv=%u mismatches=%d",
             N, L, k, k_inv, mismatches);
    report("Galois roundtrip (σ_k ∘ σ_{k^{-1}})", mismatches == 0, detail);

    // Also test with column-swap element k = 2N-1
    uint32_t k2 = 2 * N - 1;  // = 31, inverse is also 31 since 31*31 = 961 mod 32 = 1
    uint32_t k2_inv = galois_inverse(k2, N);

    // Reinitialize d_poly to original
    CUDA_CHECK(cudaMemcpy(d_poly, h_orig.data(), L * N * sizeof(uint64_t), cudaMemcpyHostToDevice));
    apply_galois(d_poly, k2,     d_tmp,  N, L, d_primes);
    CUDA_CHECK(cudaDeviceSynchronize());
    apply_galois(d_tmp,  k2_inv, d_poly, N, L, d_primes);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto h_result2 = d2h(d_poly, L * N);
    int mismatches2 = count_mismatch(h_result2, h_orig);

    snprintf(detail, sizeof(detail),
             "N=%d L=%d k=%u (col-swap, self-inv) mismatches=%d",
             N, L, k2, mismatches2);
    report("Galois roundtrip (column swap)", mismatches2 == 0, detail);

    // Test that σ_k is NOT the identity (sanity check)
    CUDA_CHECK(cudaMemcpy(d_poly, h_orig.data(), L * N * sizeof(uint64_t), cudaMemcpyHostToDevice));
    apply_galois(d_poly, k, d_tmp, N, L, d_primes);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto h_permuted = d2h(d_tmp, L * N);
    int same = count_mismatch(h_permuted, h_orig);
    // 'same == 0' would be wrong (σ_3 should change the poly) — we expect same > 0
    snprintf(detail, sizeof(detail),
             "σ_k changed %d coefficients (expected > 0 for non-identity k=%u)", same, k);
    report("Galois not identity (sanity)", same > 0, detail);

    cudaFree(d_poly); cudaFree(d_tmp); cudaFree(d_primes);
}

// =============================================================================
// Optional large-N NTT test (N=32768, 60-bit prime)
// =============================================================================
static void test_ntt_large()
{
    printf("\n[Test 1b] NTT Roundtrip N=32768 (60-bit prime)\n");

    int N = 32768;
    uint64_t p = 1152921504606584833ULL;   // DEFAULT_PRIMES[0]
    uint64_t g = 683635763366557341ULL;    // prim 65536th root mod p

    NttTable tbl = ntt_table_create(p, g, N);

    std::mt19937_64 rng(42);
    std::vector<uint64_t> h_poly(N);
    {
        std::uniform_int_distribution<uint64_t> dist(0, p - 1);
        for (auto& x : h_poly) x = dist(rng);
    }
    auto h_orig = h_poly;

    uint64_t* d_poly = h2d(h_poly);

    ntt_forward_single(d_poly, tbl);
    CUDA_CHECK(cudaDeviceSynchronize());
    ntt_inverse_single(d_poly, tbl);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto h_result = d2h(d_poly, N);
    int mismatches = count_mismatch(h_result, h_orig);

    char detail[80];
    snprintf(detail, sizeof(detail),
             "N=%d p=0x%llx mismatches=%d",
             N, (unsigned long long)p, mismatches);
    report("NTT roundtrip N=32768", mismatches == 0, detail);

    cudaFree(d_poly);
    ntt_table_free(tbl);
}

// =============================================================================
// main
// =============================================================================
int main(int argc, char** argv)
{
    bool run_large = false;
    for (int i = 1; i < argc; i++)
        if (std::string(argv[i]) == "--large") run_large = true;

    // Print GPU info
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("=== BFV Layer 1 Correctness Verifier ===\n");
    printf("GPU: %s (sm_%d%d, %.0f MB global mem)\n\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1e6);

    test_ntt_roundtrip();
    test_poly_mul();
    test_modup_moddown();
    test_ks_ip();
    test_galois();

    if (run_large) {
        test_ntt_large();
    }

    printf("\n=== Summary: %d PASS, %d FAIL ===\n", g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}
