#pragma once
// math_utils_gpu.h — pure C++ integer math helpers, no SEAL/Phantom dependency.
// This is a copy of nshedb/utils/math_utils.h adapted for the nshedb_gpu namespace.

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace nshedb_gpu {
namespace utils {

struct PolyDivResult {
    std::vector<int64_t> q;
    std::vector<int64_t> r;
};

// Modular exponentiation: (base^exp) % modulus
inline int64_t powMod(int64_t base, int64_t exp, int64_t modulus) {
    int64_t res = 1;
    base %= modulus;
    while (exp > 0) {
        if (exp % 2 == 1) res = (res * base) % modulus;
        base = (base * base) % modulus;
        exp /= 2;
    }
    if (res < 0) res += modulus;
    return res;
}

// Returns floor(log2(n)) such that 2^result >= n (ceiling of log2)
inline int64_t nextPowerOf2(int64_t n) {
    int64_t a = static_cast<int64_t>(std::log2(static_cast<double>(n)));
    if ((int64_t)(1LL << a) == n) return a;
    else return a + 1;
}

// Polynomial long division over Z_modulus.
// Returns quotient q and remainder r such that: dividend = q * divisor + r (mod modulus)
PolyDivResult dividePoly(const std::vector<int64_t>& dividend,
                         const std::vector<int64_t>& divisor,
                         int64_t modulus);

// Returns the index of the leading (highest-degree non-zero) coefficient.
inline int64_t getDegree(const std::vector<int64_t>& coeffs) {
    for (int64_t i = static_cast<int64_t>(coeffs.size()) - 1; i > 0; i--) {
        if (coeffs[i] != 0) return i;
    }
    return 0;
}

// Generates all combinations of elements drawn one-from-each sub-vector.
// E.g. {{1,2},{3,4}} → {{1,3},{1,4},{2,3},{2,4}}
template<typename T>
std::vector<std::vector<T>> generateCombinations(const std::vector<std::vector<T>>& vectors) {
    if (vectors.empty()) return {};
    std::vector<int> indices(vectors.size(), 0);
    std::vector<std::vector<T>> res;

    while (true) {
        std::vector<T> tmp;
        for (size_t i = 0; i < vectors.size(); ++i) {
            if (vectors[i].empty()) return {};
            tmp.push_back(vectors[i][indices[i]]);
        }
        res.push_back(tmp);

        int next = static_cast<int>(vectors.size()) - 1;
        while (next >= 0 && (indices[next] + 1 >= static_cast<int>(vectors[next].size()))) {
            indices[next] = 0;
            next--;
        }
        if (next < 0) break;
        indices[next]++;
    }
    return res;
}

} // namespace utils
} // namespace nshedb_gpu
