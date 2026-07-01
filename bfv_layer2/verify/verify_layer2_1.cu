// =============================================================================
// verify_layer2_1.cu — Layer 2-1 correctness tests
//
// Tests:
//  1. BfvContext creation     — params, primes, delta nonzero
//  2. RnsPoly alloc/copy      — write known values, read back
//  3. RnsPoly NTT roundtrip   — const-1 poly: NTT then INTT == identity
//  4. Encode/decode N=64      — 64 random values mod t roundtrip
//  5. Encode/decode N=1024    — 1024 random values mod t roundtrip
// =============================================================================

#include "../bfv_context.cuh"
#include "../bfv_types.cuh"
#include "../bfv_encode.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ── Test harness ─────────────────────────────────────────────────────────────

static int s_pass = 0, s_fail = 0;

static void report(const char* name, bool ok) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (ok) s_pass++; else s_fail++;
}

// ── Test 1: BfvContext creation ───────────────────────────────────────────────

static void test_context() {
    printf("\n=== Test 1: BfvContext creation ===\n");

    bfv::BfvContext ctx = bfv::BfvContext::create(64, 2, 1, 65537);
    ctx.print();

    bool ok = true;
    ok &= (ctx.N         == 64);
    ok &= (ctx.logN      == 6);
    ok &= (ctx.L         == 2);
    ok &= (ctx.K         == 1);
    ok &= (ctx.plain_mod == 65537);
    ok &= (ctx.primes[0] != 0);
    ok &= (ctx.primes[1] != 0);
    ok &= (ctx.special_primes[0] != 0);
    ok &= (ctx.delta     >  0);
    ok &= ((int)ctx.ntt_tables.size() == 3);  // L+K = 3

    ctx.destroy();
    report("BfvContext: N=64, L=2, K=1, t=65537", ok);
}

// ── Test 2: RnsPoly alloc / copy ─────────────────────────────────────────────

static void test_rnspoly_copy() {
    printf("\n=== Test 2: RnsPoly alloc / copy ===\n");

    bfv::BfvContext ctx = bfv::BfvContext::create(64, 2, 1, 65537);

    bfv::RnsPoly poly;
    poly.alloc(ctx);   // 2 limbs × 64 coeffs

    // Write sequential values
    const int total = ctx.L * ctx.N;   // 128
    std::vector<uint64_t> h_src(total), h_dst(total, 0);
    for (int i = 0; i < total; i++) h_src[i] = (uint64_t)(i + 1);

    poly.copy_from_host(h_src.data(), ctx.L);
    poly.copy_to_host(h_dst.data());

    bool ok = true;
    for (int i = 0; i < total; i++)
        if (h_dst[i] != h_src[i]) { ok = false; break; }

    poly.free();
    ctx.destroy();
    report("RnsPoly alloc/copy roundtrip", ok);
}

// ── Test 3: RnsPoly NTT roundtrip ────────────────────────────────────────────

static void test_ntt_roundtrip() {
    printf("\n=== Test 3: RnsPoly NTT roundtrip (const-1 poly) ===\n");

    bfv::BfvContext ctx = bfv::BfvContext::create(64, 2, 1, 65537);

    bfv::RnsPoly poly;
    poly.alloc(ctx);

    // Set each limb to the constant polynomial 1:
    //   coeff 0 = 1, coeffs 1..N-1 = 0
    std::vector<uint64_t> h_in(ctx.L * ctx.N, 0);
    for (int l = 0; l < ctx.L; l++)
        h_in[(size_t)l * ctx.N] = 1;   // coeff 0 of limb l

    poly.copy_from_host(h_in.data(), ctx.L);
    poly.is_ntt = false;

    poly.to_ntt(ctx);
    poly.to_coeff(ctx);

    std::vector<uint64_t> h_out(ctx.L * ctx.N, 0);
    poly.copy_to_host(h_out.data());

    bool ok = true;
    for (int l = 0; l < ctx.L && ok; l++) {
        const uint64_t* lp = h_out.data() + (size_t)l * ctx.N;
        if (lp[0] != 1) ok = false;
        for (int i = 1; i < ctx.N && ok; i++)
            if (lp[i] != 0) ok = false;
    }

    poly.free();
    ctx.destroy();
    report("NTT roundtrip: INTT(NTT(1)) == 1", ok);
}

// ── Test 4: Encode / decode N=64 ─────────────────────────────────────────────

static void test_encode_decode_64() {
    printf("\n=== Test 4: Encode/decode roundtrip N=64, L=2, t=65537 ===\n");

    bfv::BfvContext ctx = bfv::BfvContext::create(64, 2, 1, 65537);

    std::vector<uint64_t> values(64);
    srand(42);
    for (int i = 0; i < 64; i++) values[i] = (uint64_t)rand() % 65537;

    bfv::BfvPlaintext pt;
    pt.alloc(ctx);

    bfv::bfv_encode(ctx, values, pt);

    std::vector<uint64_t> decoded;
    bfv::bfv_decode(ctx, pt, decoded);

    bool ok = ((int)decoded.size() == 64);
    for (int i = 0; ok && i < 64; i++)
        ok = (decoded[i] == values[i]);

    if (!ok) {
        printf("  First mismatch:\n");
        for (int i = 0; i < 64; i++)
            if (decoded[i] != values[i]) {
                printf("    slot %d: expected %llu, got %llu\n",
                       i, (unsigned long long)values[i],
                          (unsigned long long)decoded[i]);
                break;
            }
    }

    pt.free();
    ctx.destroy();
    report("Encode/decode N=64", ok);
}

// ── Test 5: Encode / decode N=1024 ───────────────────────────────────────────

static void test_encode_decode_1024() {
    printf("\n=== Test 5: Encode/decode roundtrip N=1024, L=2, t=65537 ===\n");

    bfv::BfvContext ctx = bfv::BfvContext::create(1024, 2, 1, 65537);

    std::vector<uint64_t> values(1024);
    srand(123);
    for (int i = 0; i < 1024; i++) values[i] = (uint64_t)rand() % 65537;

    bfv::BfvPlaintext pt;
    pt.alloc(ctx);

    bfv::bfv_encode(ctx, values, pt);

    std::vector<uint64_t> decoded;
    bfv::bfv_decode(ctx, pt, decoded);

    bool ok = ((int)decoded.size() == 1024);
    for (int i = 0; ok && i < 1024; i++)
        ok = (decoded[i] == values[i]);

    pt.free();
    ctx.destroy();
    report("Encode/decode N=1024", ok);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    printf("========================================\n");
    printf("  BFV Layer 2-1 verification\n");
    printf("========================================\n");

    test_context();
    test_rnspoly_copy();
    test_ntt_roundtrip();
    test_encode_decode_64();
    test_encode_decode_1024();

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n", s_pass, s_fail);
    printf("========================================\n");

    return (s_fail == 0) ? 0 : 1;
}
