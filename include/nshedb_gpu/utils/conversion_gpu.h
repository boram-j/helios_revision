#pragma once
// conversion_gpu.h — uint64 coefficient helpers for Phantom BFV plaintexts.
// Phantom batch-encodes std::vector<uint64_t>, so negative SEAL-style
// coefficients (stored as int64_t) must be lifted into Z_p first.

#include <cstdint>

namespace nshedb_gpu {
namespace utils {

// Convert an int64_t polynomial coefficient (possibly negative, already
// reduced mod p) into its canonical Z_p representative as uint64_t.
//
// Convention: coefficients in (-p/2, p/2] are used internally.
//   - coef >= 0 → returns coef unchanged as uint64_t
//   - coef <  0 → returns p + coef  (wraps into [p/2+1, p-1])
//
// Example: p=65537, coef=-3 → 65534
inline uint64_t coef_to_uint64(int64_t coef, int64_t p) {
    if (coef < 0) return static_cast<uint64_t>(p + coef);
    return static_cast<uint64_t>(coef);
}

// Reduce a raw polynomial coefficient (possibly >= p or < 0) fully into
// the canonical signed representative in (-p/2, p/2], then lift to uint64_t.
inline uint64_t raw_coef_to_uint64(int64_t coef, int64_t p) {
    coef %= p;
    if (coef > p / 2) coef -= p;
    else if (coef < -(p / 2)) coef += p;
    return coef_to_uint64(coef, p);
}

} // namespace utils
} // namespace nshedb_gpu
