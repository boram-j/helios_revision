#include "nshedb/utils/math_utils.h"

namespace nshedb {
namespace utils {

PolyDivResult dividePoly(const std::vector<int64_t>& dividend, 
                         const std::vector<int64_t>& divisor, 
                         int64_t p) 
{
    int64_t dividend_degree = getDegree(dividend);
    int64_t divisor_degree = getDegree(divisor);

    if (divisor_degree < 0) return {}; 

    std::vector<int64_t> q(std::max(0L, dividend_degree - divisor_degree + 1), 0);
    std::vector<int64_t> r = dividend;

    for (int64_t i = dividend_degree - divisor_degree; i >= 0; i--) {
        if (divisor_degree + i >= r.size()) continue;

        int64_t quotient_term = r[divisor_degree + i] / divisor[divisor_degree];
        q[i] = quotient_term;

        for (int64_t j = divisor_degree + i; j >= i; j--) {
            int64_t val = (quotient_term * divisor[j - i]) % p;
            r[j] -= val;
            if (r[j] < 0) r[j] += p;
        }
    }
    
    // Resize remainder to fit degree
    int64_t r_deg = getDegree(r);
    r.resize(r_deg + 1);
    
    return {q, r};
}

} // namespace utils
} // namespace nshedb