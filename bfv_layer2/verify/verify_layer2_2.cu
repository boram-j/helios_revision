// =============================================================================
// verify_layer2_2.cu — Layer 2-2 correctness tests
//
// Tests:
//  1. Secret keygen     — ternary coefficients: values in {0, 1, q_0-1}
//  2. Encrypt/decrypt   — N=64, L=2 roundtrip
//  3. CT addition       — encrypt a + encrypt b → decrypt == (a+b) mod t
//  4. Encrypt/decrypt   — N=1024, L=2 roundtrip
//  5. Galois rotation   — step=1, galois_elt=3: verify σ_3 permutation
//  6. Galois rotation   — step=32, galois_elt=65: verify σ_65 permutation
//
// For tests 5-6 the expected output is computed on the host via the galois
// permutation formula  out[k*j mod 2N] = ±in[j]  and compared against the
// decrypted rotation result.
// =============================================================================

#include "../bfv_context.cuh"
#include "../bfv_types.cuh"
#include "../bfv_encode.cuh"
#include "../bfv_keygen.cuh"
#include "../bfv_encrypt.cuh"
#include "../bfv_rotate.cuh"
#include "../../bfv_core/ntt.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

// ── Test harness ──────────────────────────────────────────────────────────────

static int s_pass = 0, s_fail = 0;

static void report(const char* name, bool ok) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (ok) s_pass++; else s_fail++;
}

// ── Galois permutation (host reference) ───────────────────────────────────────
//
// Computes σ_k(f) in the ring Z_t[X]/(X^N + 1):
//   f(X) = Σ in[j] X^j
//   σ_k(f)(X) = f(X^k)
//   Coefficient at index idx in σ_k(f):
//     out[k*j mod 2N] += ±in[j]
//     sign -1 if (k*j mod 2N) >= N  (X^N ≡ -1)
//
// galois_elt k must satisfy gcd(k, 2N) = 1; each output index is hit exactly once.
static std::vector<uint64_t> galois_permute_host(
        const std::vector<uint64_t>& in,
        uint32_t k, int N, uint64_t t)
{
    std::vector<uint64_t> out(N, 0);
    const uint64_t two_N = (uint64_t)2 * N;
    for (int j = 0; j < N; j++) {
        uint64_t kj  = ((uint64_t)k * j) % two_N;
        bool     neg = (kj >= (uint64_t)N);
        int      idx = neg ? (int)(kj - N) : (int)kj;
        uint64_t v   = in[j];
        if (neg) v = (t - v) % t;          // −in[j] mod t  (handles v==0 correctly)
        out[idx] = (out[idx] + v) % t;     // unique idx per j, addition for safety
    }
    return out;
}

// ── Test 1: Secret keygen — ternary property ─────────────────────────────────

static void test1_secret_keygen() {
    bfv::BfvContext ctx = bfv::BfvContext::create(64, 2, 1, 65537);
    bfv::BfvSecretKey sk;
    bfv::bfv_secret_keygen(ctx, sk);

    const int N = ctx.N;

    // INTT limb 0 of sk.s into a scratch buffer, then check for ternary values.
    uint64_t* d_tmp = nullptr;
    cudaMalloc(&d_tmp, (size_t)N * sizeof(uint64_t));
    cudaMemcpy(d_tmp, sk.s.d_data, (size_t)N * sizeof(uint64_t),
               cudaMemcpyDeviceToDevice);
    bfv_core::ntt_inverse_single(d_tmp, ctx.ntt_tables[0]);
    cudaDeviceSynchronize();

    std::vector<uint64_t> h0(N);
    cudaMemcpy(h0.data(), d_tmp, (size_t)N * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_tmp);

    // Valid ternary values mod q_0:  0, 1, and q_0-1 (representing -1).
    const uint64_t p0 = ctx.primes[0];
    bool ok = true;
    for (int i = 0; ok && i < N; i++) {
        uint64_t v = h0[i];
        if (v != 0 && v != 1 && v != p0 - 1) ok = false;
    }

    sk.free();
    ctx.destroy();
    report("Test 1: secret keygen ternary property", ok);
}

// ── Test 2: Encrypt/decrypt roundtrip N=64, L=2 ──────────────────────────────

static void test2_enc_dec_64() {
    srand(12345);
    bfv::BfvContext ctx = bfv::BfvContext::create(64, 2, 1, 65537);

    bfv::BfvSecretKey sk;  bfv::bfv_secret_keygen(ctx, sk);
    bfv::BfvPublicKey pk;  bfv::bfv_public_keygen(ctx, sk, pk);

    const int N = ctx.N;
    std::vector<uint64_t> vals(N);
    for (int i = 0; i < N; i++) vals[i] = (uint64_t)rand() % 65537;

    bfv::BfvPlaintext pt;  pt.alloc(ctx);
    bfv::bfv_encode(ctx, vals, pt);

    bfv::BfvCiphertext ct;
    bfv::bfv_encrypt(ctx, pk, pt, ct);

    bfv::BfvPlaintext pt_out;  pt_out.alloc(ctx);
    bfv::bfv_decrypt(ctx, sk, ct, pt_out);

    bool ok = ((int)pt_out.slots.size() == N);
    for (int i = 0; ok && i < N; i++)
        if (pt_out.slots[i] != vals[i]) ok = false;

    ct.free();  pt.free();  pt_out.free();
    pk.free();  sk.free();  ctx.destroy();
    report("Test 2: encrypt/decrypt roundtrip N=64 L=2", ok);
}

// ── Test 3: CT addition — decrypt(enc(a) + enc(b)) == (a+b) mod t ────────────

static void test3_ct_add() {
    srand(12345);
    bfv::BfvContext ctx = bfv::BfvContext::create(64, 2, 1, 65537);

    bfv::BfvSecretKey sk;  bfv::bfv_secret_keygen(ctx, sk);
    bfv::BfvPublicKey pk;  bfv::bfv_public_keygen(ctx, sk, pk);

    const int      N = ctx.N;
    const uint64_t t = ctx.plain_mod;

    std::vector<uint64_t> a(N), b(N);
    for (int i = 0; i < N; i++) a[i] = (uint64_t)rand() % t;
    for (int i = 0; i < N; i++) b[i] = (uint64_t)rand() % t;

    bfv::BfvPlaintext pta, ptb;
    pta.alloc(ctx);  ptb.alloc(ctx);
    bfv::bfv_encode(ctx, a, pta);
    bfv::bfv_encode(ctx, b, ptb);

    bfv::BfvCiphertext cta, ctb, ctsum;
    bfv::bfv_encrypt(ctx, pk, pta, cta);
    bfv::bfv_encrypt(ctx, pk, ptb, ctb);
    bfv::bfv_add(ctx, cta, ctb, ctsum);

    bfv::BfvPlaintext pt_out;  pt_out.alloc(ctx);
    bfv::bfv_decrypt(ctx, sk, ctsum, pt_out);

    bool ok = true;
    for (int i = 0; ok && i < N; i++) {
        uint64_t expected = (a[i] + b[i]) % t;
        if (pt_out.slots[i] != expected) ok = false;
    }

    ctsum.free();  ctb.free();  cta.free();
    ptb.free();    pta.free();  pt_out.free();
    pk.free();     sk.free();   ctx.destroy();
    report("Test 3: ciphertext addition (a+b) mod t", ok);
}

// ── Test 4: Encrypt/decrypt roundtrip N=1024 ─────────────────────────────────

static void test4_enc_dec_1024() {
    srand(12345);
    bfv::BfvContext ctx = bfv::BfvContext::create(1024, 2, 1, 65537);

    bfv::BfvSecretKey sk;  bfv::bfv_secret_keygen(ctx, sk);
    bfv::BfvPublicKey pk;  bfv::bfv_public_keygen(ctx, sk, pk);

    const int N = ctx.N;
    std::vector<uint64_t> vals(N);
    for (int i = 0; i < N; i++) vals[i] = (uint64_t)rand() % 65537;

    bfv::BfvPlaintext pt;  pt.alloc(ctx);
    bfv::bfv_encode(ctx, vals, pt);

    bfv::BfvCiphertext ct;
    bfv::bfv_encrypt(ctx, pk, pt, ct);

    bfv::BfvPlaintext pt_out;  pt_out.alloc(ctx);
    bfv::bfv_decrypt(ctx, sk, ct, pt_out);

    bool ok = ((int)pt_out.slots.size() == N);
    for (int i = 0; ok && i < N; i++)
        if (pt_out.slots[i] != vals[i]) ok = false;

    ct.free();  pt.free();  pt_out.free();
    pk.free();  sk.free();  ctx.destroy();
    report("Test 4: encrypt/decrypt roundtrip N=1024 L=2", ok);
}

// ── Test 5: Galois rotation, step=1 ──────────────────────────────────────────
//
// galois_elt = (2*1 + 1) % (2*64) = 3.
// Expected output: galois_permute_host([0,...,63], 3, 64, t).
// For each j: out[3j mod 128] = ±in[j], negative if 3j mod 128 >= 64.

static void test5_rotate_step1() {
    bfv::BfvContext ctx = bfv::BfvContext::create(64, 2, 1, 65537);

    const int      N = ctx.N;
    const uint64_t t = ctx.plain_mod;

    bfv::BfvSecretKey sk;  bfv::bfv_secret_keygen(ctx, sk);
    bfv::BfvPublicKey pk;  bfv::bfv_public_keygen(ctx, sk, pk);

    const uint32_t step       = 1;
    const uint32_t galois_elt = (2 * step + 1) % (uint32_t)(2 * N);  // 3

    bfv::GaloisKeyEntry gke;
    bfv::bfv_galois_keygen(ctx, sk, galois_elt, gke);

    // Input: [0, 1, 2, ..., N-1]
    std::vector<uint64_t> vals(N);
    for (int i = 0; i < N; i++) vals[i] = (uint64_t)i;

    // Reference: host-computed galois permutation
    std::vector<uint64_t> expected = galois_permute_host(vals, galois_elt, N, t);

    bfv::BfvPlaintext pt;  pt.alloc(ctx);
    bfv::bfv_encode(ctx, vals, pt);

    bfv::BfvCiphertext ct, ct_rot;
    bfv::bfv_encrypt(ctx, pk, pt, ct);
    bfv::bfv_rotate(ctx, gke, galois_elt, ct, ct_rot);

    bfv::BfvPlaintext pt_out;  pt_out.alloc(ctx);
    bfv::bfv_decrypt(ctx, sk, ct_rot, pt_out);

    bool ok = ((int)pt_out.slots.size() == N);
    for (int i = 0; ok && i < N; i++)
        if (pt_out.slots[i] != expected[i]) ok = false;

    // Debug: always print first 4 slots so failures can be diagnosed on the
    // target machine even before any test-harness output.
    printf("    [dbg test5] first 4 actual vs expected (galois_elt=%u):\n",
           (unsigned)galois_elt);
    for (int i = 0; i < 4 && i < N; i++) {
        uint64_t got  = pt_out.slots[i];   // slots has N entries after alloc+decrypt
        uint64_t want = expected[i];
        printf("      slot[%d]: got=%llu  want=%llu  %s\n",
               i,
               (unsigned long long)got,
               (unsigned long long)want,
               (got == want) ? "OK" : "MISMATCH");
    }

    ct_rot.free();  ct.free();  pt.free();  pt_out.free();
    gke.free();  pk.free();  sk.free();  ctx.destroy();
    report("Test 5: rotation step=1 (galois_elt=3)", ok);
}

// ── Test 6: Galois rotation, step=N/2=32 ─────────────────────────────────────
//
// galois_elt = (2*32 + 1) % (2*64) = 65.
// Expected output: galois_permute_host([0,...,63], 65, 64, t).
// For each j: out[65j mod 128] = ±in[j], negative if 65j mod 128 >= 64.

static void test6_rotate_half() {
    bfv::BfvContext ctx = bfv::BfvContext::create(64, 2, 1, 65537);

    const int      N = ctx.N;
    const uint64_t t = ctx.plain_mod;

    bfv::BfvSecretKey sk;  bfv::bfv_secret_keygen(ctx, sk);
    bfv::BfvPublicKey pk;  bfv::bfv_public_keygen(ctx, sk, pk);

    const uint32_t step       = (uint32_t)(N / 2);                        // 32
    const uint32_t galois_elt = (2 * step + 1) % (uint32_t)(2 * N);       // 65

    bfv::GaloisKeyEntry gke;
    bfv::bfv_galois_keygen(ctx, sk, galois_elt, gke);

    // Input: [0, 1, 2, ..., N-1]
    std::vector<uint64_t> vals(N);
    for (int i = 0; i < N; i++) vals[i] = (uint64_t)i;

    // Reference: host-computed galois permutation
    std::vector<uint64_t> expected = galois_permute_host(vals, galois_elt, N, t);

    bfv::BfvPlaintext pt;  pt.alloc(ctx);
    bfv::bfv_encode(ctx, vals, pt);

    bfv::BfvCiphertext ct, ct_rot;
    bfv::bfv_encrypt(ctx, pk, pt, ct);
    bfv::bfv_rotate(ctx, gke, galois_elt, ct, ct_rot);

    bfv::BfvPlaintext pt_out;  pt_out.alloc(ctx);
    bfv::bfv_decrypt(ctx, sk, ct_rot, pt_out);

    bool ok = ((int)pt_out.slots.size() == N);
    for (int i = 0; ok && i < N; i++)
        if (pt_out.slots[i] != expected[i]) ok = false;

    ct_rot.free();  ct.free();  pt.free();  pt_out.free();
    gke.free();  pk.free();  sk.free();  ctx.destroy();
    report("Test 6: rotation step=N/2 (galois_elt=65)", ok);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    printf("========================================\n");
    printf("  BFV Layer 2-2 verification\n");
    printf("========================================\n");

    test1_secret_keygen();
    test2_enc_dec_64();
    test3_ct_add();
    test4_enc_dec_1024();
    test5_rotate_step1();
    test6_rotate_half();

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n", s_pass, s_fail);
    printf("========================================\n");

    return (s_fail == 0) ? 0 : 1;
}
