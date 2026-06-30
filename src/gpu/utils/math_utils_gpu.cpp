// math_utils_gpu.cpp — dividePoly implementation for nshedb_gpu.
// Pure C++ (no CUDA), compiled as a regular translation unit.

#include "nshedb_gpu/utils/math_utils_gpu.h"

namespace nshedb_gpu {
namespace utils {

PolyDivResult dividePoly(const std::vector<int64_t>& dividend,
                         const std::vector<int64_t>& divisor,
                         int64_t p)
{
    int64_t dividend_degree = getDegree(dividend);
    int64_t divisor_degree  = getDegree(divisor);

    if (divisor_degree < 0) return {};

    std::vector<int64_t> q(
        std::max(int64_t(0), dividend_degree - divisor_degree + 1), 0);
    std::vector<int64_t> r = dividend;

    for (int64_t i = dividend_degree - divisor_degree; i >= 0; i--) {
        if (divisor_degree + i >= static_cast<int64_t>(r.size())) continue;

        int64_t quotient_term = r[divisor_degree + i] / divisor[divisor_degree];
        q[i] = quotient_term;

        for (int64_t j = divisor_degree + i; j >= i; j--) {
            int64_t val = (quotient_term * divisor[j - i]) % p;
            r[j] -= val;
            if (r[j] < 0) r[j] += p;
        }
    }

    int64_t r_deg = getDegree(r);
    r.resize(r_deg + 1);

    return {q, r};
}

} // namespace utils
} // namespace nshedb_gpu
