#pragma once
#include "seal/seal.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

namespace seal_examples {

inline void print_example_banner(std::string title) {
    if (!title.empty()) {
        std::size_t title_length = title.length();
        std::size_t banner_length = title_length + 2 * 10;
        std::string banner_top = "+" + std::string(banner_length - 2, '-') + "+";
        std::string banner_middle = "|" + std::string(9, ' ') + title + std::string(9, ' ') + "|";
        std::cout << std::endl << banner_top << std::endl << banner_middle << std::endl << banner_top << std::endl;
    }
}

inline void print_parameters(const seal::SEALContext &context) {
    auto &context_data = *context.key_context_data();
    std::string scheme_name;
    switch (context_data.parms().scheme()) {
        case seal::scheme_type::bfv: scheme_name = "BFV"; break;
        case seal::scheme_type::ckks: scheme_name = "CKKS"; break;
        case seal::scheme_type::bgv: scheme_name = "BGV"; break;
        default: scheme_name = "unsupported scheme";
    }
    std::cout << "/\n| Encryption parameters :\n";
    std::cout << "|   scheme: " << scheme_name << "\n";
    std::cout << "|   poly_modulus_degree: " << context_data.parms().poly_modulus_degree() << "\n";
    std::cout << "|   coeff_modulus size: " << context_data.total_coeff_modulus_bit_count() << " (";
    auto coeff_modulus = context_data.parms().coeff_modulus();
    for (std::size_t i = 0; i < coeff_modulus.size() - 1; i++) std::cout << coeff_modulus[i].bit_count() << " + ";
    std::cout << coeff_modulus.back().bit_count() << ") bits\n";
    if (context_data.parms().scheme() == seal::scheme_type::bfv) {
        std::cout << "|   plain_modulus: " << context_data.parms().plain_modulus().value() << "\n";
    }
    std::cout << "\\" << std::endl;
}

template <typename T>
inline void print_vector(std::vector<T> vec, std::size_t print_size = 4, int prec = 3) {
    std::cout << std::fixed << std::setprecision(prec) << "\n    [";
    std::size_t slot_count = vec.size();
    if (slot_count <= 2 * print_size) {
        for (std::size_t i = 0; i < slot_count; i++) std::cout << " " << vec[i] << ((i != slot_count - 1) ? "," : " ]\n");
    } else {
        for (std::size_t i = 0; i < print_size; i++) std::cout << " " << vec[i] << ",";
        std::cout << " ...,";
        for (std::size_t i = slot_count - print_size; i < slot_count; i++) std::cout << " " << vec[i] << ((i != slot_count - 1) ? "," : " ]\n");
    }
    std::cout << std::endl;
}

template <typename T>
inline void print_dec(std::string st, const seal::Ciphertext ctxt_res, int numEle, seal::BatchEncoder& batch_encoder, seal::Decryptor& decryptor) {
    seal::Plaintext ptxt_res;
    std::vector<T> res;
    decryptor.decrypt(ctxt_res, ptxt_res);
    batch_encoder.decode(ptxt_res, res);
    std::cout << st << ":" << std::setw(3) << "[";
    for (int i = 0; i < numEle; i++) std::cout << std::setw(3) << std::right << res[i] << ",";
    std::cout << "] " << std::endl;
    std::cout << "    + noise budget: " << decryptor.invariant_noise_budget(ctxt_res) << " bits" << std::endl;
}

} // namespace seal_examples