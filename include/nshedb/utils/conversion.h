#pragma once
#include "seal/seal.h"
#include <string>
#include <cstdint>

namespace nshedb {
namespace utils {

inline std::string intToHex(int64_t num, int64_t p) {
    uint64_t unum = (num < 0) ? (p + static_cast<uint64_t>(num)) : static_cast<uint64_t>(num);
    return seal::util::uint_to_hex_string(&unum, 1);
}

inline std::string uint64_to_hex_string(std::uint64_t value) {
    return seal::util::uint_to_hex_string(&value, std::size_t(1));
}

} // namespace utils
} // namespace nshedb