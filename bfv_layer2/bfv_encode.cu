// =============================================================================
// bfv_encode.cu — BFV batch encoder / decoder implementation
// =============================================================================

#include "bfv_encode.cuh"
#include "../bfv_core/ntt.cuh"
#include <cassert>
#include <cuda_runtime.h>

namespace bfv {

void bfv_encode(const BfvContext&            ctx,
                const std::vector<uint64_t>& values,
                BfvPlaintext&                pt)
{
    assert((int)values.size() == ctx.N);

    // Store CPU-side slots
    pt.slots = values;

    // Build L*N host buffer: limb l coeff i = values[i] % q_l
    // values[i] < t < q_l (all primes ~2^60, t <= 2^30), so % is a no-op in
    // practice, but included for correctness in the general case.
    std::vector<uint64_t> h_plain((size_t)ctx.L * ctx.N);
    for (int l = 0; l < ctx.L; l++) {
        const uint64_t q_l = ctx.primes[l];
        uint64_t* dst = h_plain.data() + (size_t)l * ctx.N;
        for (int i = 0; i < ctx.N; i++)
            dst[i] = values[i] % q_l;
    }

    // Upload to GPU and take NTT
    pt.encoded.copy_from_host(h_plain.data(), ctx.L);
    pt.encoded.is_ntt = false;
    pt.encoded.to_ntt(ctx);   // sets is_ntt = true
}

void bfv_decode(const BfvContext&       ctx,
                const BfvPlaintext&     pt,
                std::vector<uint64_t>&  values_out)
{
    // We only need limb 0.  Copy it to a temporary device buffer and INTT it.
    // The resulting coefficients are m_i mod q_0; since m_i < t < q_0 this is exact.
    uint64_t* d_tmp = nullptr;
    cudaMalloc(&d_tmp, (size_t)ctx.N * sizeof(uint64_t));

    // Copy limb 0 from the encoded poly (source = first N elements of d_data)
    cudaMemcpy(d_tmp, pt.encoded.d_data,
               (size_t)ctx.N * sizeof(uint64_t),
               cudaMemcpyDeviceToDevice);

    if (pt.encoded.is_ntt) {
        bfv_core::ntt_inverse_single(d_tmp, ctx.ntt_tables[0]);
        cudaDeviceSynchronize();
    }

    // Download
    std::vector<uint64_t> h_coeffs(ctx.N);
    cudaMemcpy(h_coeffs.data(), d_tmp,
               (size_t)ctx.N * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    cudaFree(d_tmp);

    // Reduce mod t
    values_out.resize(ctx.N);
    for (int i = 0; i < ctx.N; i++)
        values_out[i] = h_coeffs[i] % ctx.plain_mod;
}

} // namespace bfv
