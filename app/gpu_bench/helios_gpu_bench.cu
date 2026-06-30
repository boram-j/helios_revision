/**
 * helios_gpu_bench.cu — GPU benchmark for nshedb_gpu (Phantom BFV).
 *
 * Benchmarks:
 *   1. isLessThan (ct-ct)    — the core comparison circuit
 *   2. COUNT_GPU              — log-depth slot summation
 *   3. Band-join pattern     — BETWEEN_GPU + COUNT_GPU
 *
 * Parameters match the CPU bench (helios_bucket_bench.cpp):
 *   poly_modulus_degree = 32768
 *   plain_modulus       = 65537
 *   coeff_modulus       = CoeffModulus::BFVDefault(32768)
 *
 * Output format compatible with helios_calib.txt:
 *   slot_count  isLessThan_ms  count_ms  band_join_ms
 *
 * Usage:
 *   ./bin/helios_gpu_bench [warmup]
 *     warmup  — optional flag to run a warmup round before timing
 */

#include "nshedb_gpu/nshedb_gpu.h"
#include "hecache/hecache.h"

#include <phantom.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <stdexcept>
#include <cmath>

using namespace phantom;
using namespace phantom::arith;
using namespace nshedb_gpu::core;
using namespace nshedb_gpu::predicates;
using nshedb_gpu::utils::BfvRotationKeyStream;

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------
struct GpuTimer {
    cudaEvent_t start_, stop_;

    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() { cudaEventRecord(start_); }

    float stop_ms() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};

// ---------------------------------------------------------------------------
// Benchmark parameters
// ---------------------------------------------------------------------------
static constexpr size_t POLY_DEG    = 32768;
static constexpr size_t PLAIN_MOD   = 65537;
static constexpr int    N_ITERS     = 3;     // averaging iterations
static constexpr int    WARMUP_ITERS = 1;

// ---------------------------------------------------------------------------
// Build BFV context matching the CPU bench
// ---------------------------------------------------------------------------
static PhantomContext make_context()
{
    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(POLY_DEG);

    // Use BFVDefault for 128-bit security at degree 32768
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(POLY_DEG));
    parms.set_plain_modulus(Modulus(PLAIN_MOD));

    // Default mul_tech for BFV is HPS (set in EncryptionParameters ctor)
    return PhantomContext(parms);
}

// ---------------------------------------------------------------------------
// Encode a random plaintext vector of slot_count uint64_t values in [0, p-1]
// ---------------------------------------------------------------------------
static PhantomPlaintext random_plain(const PhantomContext     &ctx,
                                     const PhantomBatchEncoder &enc,
                                     size_t                     slot_count,
                                     uint64_t                   p,
                                     std::mt19937_64           &rng)
{
    std::uniform_int_distribution<uint64_t> dist(0, p - 1);
    std::vector<uint64_t> vec(slot_count);
    for (auto &v : vec) v = dist(rng);
    return enc.encode(ctx, vec);
}

// ---------------------------------------------------------------------------
// Benchmark isLessThan (ct-ct)
// ---------------------------------------------------------------------------
static float bench_isLessThan(const PhantomContext     &ctx,
                               const PhantomBatchEncoder &enc,
                               const PhantomSecretKey    &sk,
                               const PhantomRelinKey     &rlk,
                               GpuComparator             &comp,
                               int                        n_iters,
                               std::mt19937_64           &rng)
{
    size_t slots = enc.slot_count();
    GpuTimer timer;
    float total = 0.f;

    for (int it = 0; it < n_iters; it++) {
        PhantomPlaintext pa = random_plain(ctx, enc, slots, PLAIN_MOD, rng);
        PhantomPlaintext pb = random_plain(ctx, enc, slots, PLAIN_MOD, rng);

        PhantomCiphertext ca = sk.encrypt_symmetric(ctx, pa);
        PhantomCiphertext cb = sk.encrypt_symmetric(ctx, pb);

        // Warm GPU up slightly (first iter of any sequence tends to be slower)
        cudaDeviceSynchronize();

        timer.start();
        auto res = comp.isLessThan(rlk, ca, cb);
        total += timer.stop_ms();

        (void)res; // prevent optimisation away
    }
    return total / static_cast<float>(n_iters);
}

// ---------------------------------------------------------------------------
// Benchmark COUNT_GPU
// ---------------------------------------------------------------------------
static float bench_count(const PhantomContext     &ctx,
                          const PhantomBatchEncoder &enc,
                          const PhantomSecretKey    &sk,
                          BfvRotationKeyStream      &rks,
                          int                        n_iters,
                          std::mt19937_64           &rng)
{
    size_t slots = enc.slot_count();
    GpuTimer timer;
    float total = 0.f;

    for (int it = 0; it < n_iters; it++) {
        PhantomPlaintext pa = random_plain(ctx, enc, slots, 2, rng); // binary selector
        PhantomCiphertext ca = sk.encrypt_symmetric(ctx, pa);

        cudaDeviceSynchronize();

        timer.start();
        PhantomCiphertext res = COUNT_GPU(ctx, ca, slots, rks);
        total += timer.stop_ms();

        (void)res;
    }
    return total / static_cast<float>(n_iters);
}

// ---------------------------------------------------------------------------
// Benchmark BETWEEN_GPU + COUNT_GPU  (simple band-join predicate)
// ---------------------------------------------------------------------------
static float bench_band_join(const PhantomContext     &ctx,
                               const PhantomBatchEncoder &enc,
                               const PhantomSecretKey    &sk,
                               const PhantomRelinKey     &rlk,
                               BfvRotationKeyStream      &rks,
                               GpuComparator             &comp,
                               int                        n_iters,
                               std::mt19937_64           &rng)
{
    // Simulate a band-join: |a - b| <= delta
    // We encrypt the shifted difference (a + offset - b) and apply BETWEEN.
    static constexpr uint64_t OFFSET = 64;
    static constexpr uint64_t DELTA  = 16;
    // [lo, hi] = [OFFSET-DELTA, OFFSET+DELTA] = [48, 80]
    static constexpr uint64_t LO = OFFSET - DELTA;
    static constexpr uint64_t HI = OFFSET + DELTA;

    size_t slots = enc.slot_count();
    GpuTimer timer;
    float total = 0.f;

    // Precompute constant plaintexts for the bounds
    std::vector<uint64_t> lo_vec(slots, LO);
    std::vector<uint64_t> hi_vec(slots, HI);
    PhantomPlaintext plain_lo = enc.encode(ctx, lo_vec);
    PhantomPlaintext plain_hi = enc.encode(ctx, hi_vec);

    for (int it = 0; it < n_iters; it++) {
        // Encrypt shifted difference  diff = a + OFFSET - b  in [1, 127]
        std::uniform_int_distribution<uint64_t> dist_a(1, 64);
        std::uniform_int_distribution<uint64_t> dist_b(1, 64);
        std::vector<uint64_t> diff_vec(slots);
        for (auto &v : diff_vec) {
            uint64_t a = dist_a(rng), b = dist_b(rng);
            v = OFFSET + a - b + 64; // keep positive
            v %= PLAIN_MOD;
        }
        PhantomPlaintext  plain_diff = enc.encode(ctx, diff_vec);
        PhantomCiphertext ct_diff    = sk.encrypt_symmetric(ctx, plain_diff);

        cudaDeviceSynchronize();
        timer.start();

        // BETWEEN: lo <= diff <= hi
        PhantomCiphertext mask = BETWEEN_GPU(comp, ctx, rlk, ct_diff,
                                              plain_lo, plain_hi);
        // COUNT: sum up matching slots
        PhantomCiphertext count = COUNT_GPU(ctx, mask, slots, rks);

        total += timer.stop_ms();
        (void)count;
    }
    return total / static_cast<float>(n_iters);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    bool do_warmup = false;
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "warmup") do_warmup = true;
    }

    // -----------------------------------------------------------------------
    // HECache initialisation
    // Manages intermediate ciphertext and rotation-key VRAM with LRU eviction.
    // Budget values below are conservative starting points — tune for your GPU.
    //   gpu_budget_bytes : how much VRAM HECache tracks / may evict from
    //   cpu_l2_bytes     : CPU pinned spill buffer for SPILL-mode CTs
    // -----------------------------------------------------------------------
    {
        hecache::HECacheConfig cfg;
        cfg.gpu_budget_bytes = 8ULL  * 1024 * 1024 * 1024;  //  8 GB GPU
        cfg.cpu_l2_bytes     = 32ULL * 1024 * 1024 * 1024;  // 32 GB CPU
        // key_directory left empty: we use the monolithic PhantomGaloisKey
        // directly via BfvRotationKeyStream (no per-element disk files needed).
        hecache::HECache::initialize(cfg);
    }

    std::cout << "=================================================\n"
              << "  helios_gpu_bench  (Phantom BFV, GPU)\n"
              << "=================================================\n"
              << "  poly_degree  : " << POLY_DEG   << "\n"
              << "  plain_modulus: " << PLAIN_MOD  << "\n"
              << "  iterations   : " << N_ITERS    << "\n\n";

    // -----------------------------------------------------------------------
    // Context & keys
    // -----------------------------------------------------------------------
    std::cout << "Building context... " << std::flush;
    PhantomContext ctx = make_context();
    std::cout << "done.\n";

    std::cout << "Generating keys...  " << std::flush;
    PhantomSecretKey sk(ctx);
    PhantomRelinKey  rlk = sk.gen_relinkey(ctx);
    PhantomGaloisKey gk  = sk.create_galois_keys(ctx);
    std::cout << "done.\n";

    PhantomBatchEncoder enc(ctx);
    size_t slot_count = enc.slot_count();
    std::cout << "  slot_count   : " << slot_count << "\n\n";

    // Wrap galois key in a BfvRotationKeyStream for COUNT_GPU / SUM_GPU.
    // BfvRotationKeyStream holds non-owning references; gk and ctx must
    // remain in scope for the lifetime of rks.
    BfvRotationKeyStream rks(ctx, gk);

    // -----------------------------------------------------------------------
    // Comparator
    // -----------------------------------------------------------------------
    std::cout << "Building GpuComparator (computing poly coefficients)... " << std::flush;
    GpuComparator comp(ctx, enc);
    std::cout << "done.  k=" << comp.getK() << "  m=" << comp.getM() << "\n\n";

    // -----------------------------------------------------------------------
    // RNG
    // -----------------------------------------------------------------------
    std::mt19937_64 rng(42);

    // -----------------------------------------------------------------------
    // Optional warmup
    // -----------------------------------------------------------------------
    if (do_warmup) {
        std::cout << "Warmup pass...\n";
        for (int w = 0; w < WARMUP_ITERS; w++) {
            PhantomPlaintext pa = random_plain(ctx, enc, slot_count, PLAIN_MOD, rng);
            PhantomPlaintext pb = random_plain(ctx, enc, slot_count, PLAIN_MOD, rng);
            PhantomCiphertext ca = sk.encrypt_symmetric(ctx, pa);
            PhantomCiphertext cb = sk.encrypt_symmetric(ctx, pb);
            auto res = comp.isLessThan(rlk, ca, cb);
            (void)res;
        }
        std::cout << "Warmup done.\n\n";
    }

    // -----------------------------------------------------------------------
    // Benchmarks
    // -----------------------------------------------------------------------
    float t_lt = bench_isLessThan(ctx, enc, sk, rlk, comp, N_ITERS, rng);
    float t_cnt = bench_count    (ctx, enc, sk, rks,        N_ITERS, rng);
    float t_bj  = bench_band_join(ctx, enc, sk, rlk, rks, comp, N_ITERS, rng);

    // -----------------------------------------------------------------------
    // Print results
    // -----------------------------------------------------------------------
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "=================================================\n";
    std::cout << "RESULTS (avg over " << N_ITERS << " iterations)\n";
    std::cout << "=================================================\n";
    std::cout << "  slot_count           : " << slot_count << "\n";
    std::cout << "  isLessThan (ct-ct)   : " << std::setw(8) << t_lt  << " ms\n";
    std::cout << "  COUNT_GPU            : " << std::setw(8) << t_cnt << " ms\n";
    std::cout << "  band_join (BETWEEN+COUNT): " << std::setw(8) << t_bj  << " ms\n";
    std::cout << "\n";

    // Machine-readable line for helios_calib.txt compatibility:
    //   slot_count  isLessThan_ms  count_ms  band_join_ms
    std::cout << "CALIB: "
              << slot_count << " "
              << std::setprecision(3)
              << t_lt  << " "
              << t_cnt << " "
              << t_bj  << "\n";

    // -----------------------------------------------------------------------
    // HECache statistics — hit rates, PCIe movement, eviction counts
    // -----------------------------------------------------------------------
    hecache::HECache::instance().print_stats("helios_bench");

    return 0;
}
