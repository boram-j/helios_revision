#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace nshedb {
namespace utils {

struct PolyDivResult {
    std::vector<int64_t> q;
    std::vector<int64_t> r;
};

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

inline int64_t nextPowerOf2(int64_t n) {
    int64_t a = log2(n);
    if (pow(2, a) == n) return a;
    else return a + 1;
}

// Declaration only
PolyDivResult dividePoly(const std::vector<int64_t>& dividend, 
                         const std::vector<int64_t>& divisor, 
                         int64_t modulus);

inline int64_t getDegree(const std::vector<int64_t>& coeffs) {
    for (int64_t i = coeffs.size() - 1; i > 0; i--) {
        if (coeffs[i] != 0) return i;
    }
    return 0;
}

// Template implementation
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

        int next = vectors.size() - 1;
        while (next >= 0 && (indices[next] + 1 >= vectors[next].size())) {
            indices[next] = 0;
            next--;
        }
        if (next < 0) break;
        indices[next]++;
    }
    return res;
}

} // namespace utils
} // namespace nshedb