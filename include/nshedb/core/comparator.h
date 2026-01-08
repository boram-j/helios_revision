#pragma once
#include "seal/seal.h"
#include <vector>
#include <memory>

namespace nshedb {
namespace core {

class Comparator {
public:
    explicit Comparator(const seal::SEALContext &context);

    // Comparison Logic
    std::vector<seal::Ciphertext> isLessThan(seal::Evaluator& evaluator, const seal::RelinKeys &relin_keys, seal::Ciphertext& a, seal::Ciphertext& b);
    std::vector<seal::Ciphertext> isLessThan(seal::Evaluator& evaluator, const seal::RelinKeys &relin_keys, seal::Ciphertext& a, const seal::Plaintext& b);
    std::vector<seal::Ciphertext> isLessThan(seal::Evaluator& evaluator, const seal::RelinKeys &relin_keys, const seal::Plaintext& a, seal::Ciphertext& b);

    seal::Ciphertext isEqual(seal::Evaluator& evaluator, const seal::RelinKeys &relin_keys, seal::Ciphertext& a, seal::Ciphertext& b);
    seal::Ciphertext isEqual(seal::Evaluator& evaluator, const seal::RelinKeys &relin_keys, seal::Ciphertext& a, seal::Plaintext& b);

    // Getters
    int64_t getK() const { return k_; }
    int64_t getM() const { return m_; }

private:
    void initCoeff();
    void initPolyParam();
    
    // Helper methods for polynomial evaluation
    void evalPolyHelper(seal::Evaluator& evaluator, const seal::RelinKeys &relin_keys,
                        seal::Ciphertext& x, std::vector<int64_t> poly,
                        std::vector<seal::Ciphertext>& babyStep,
                        std::vector<seal::Ciphertext>& giantStep);
                        
    void evalPolyHelperPowerOf2(seal::Evaluator& evaluator, const seal::RelinKeys &relin_keys,
                                seal::Ciphertext& x, std::vector<int64_t>& poly,
                                std::vector<seal::Ciphertext>& babyStep,
                                std::vector<seal::Ciphertext>& giantStep);
                                
    void evalPolyPS(seal::Evaluator& evaluator, const seal::RelinKeys &relin_keys,
                    seal::Ciphertext& x, std::vector<int64_t>& poly,
                    std::vector<seal::Ciphertext>& babyStep,
                    std::vector<seal::Ciphertext>& giantStep,
                    int64_t t, int64_t delta);
                    
    void evalPolySimple(seal::Evaluator& evaluator, const seal::RelinKeys &relin_keys,
                        seal::Ciphertext& x, std::vector<int64_t>& poly,
                        std::vector<seal::Ciphertext>& babyStep);

    seal::Ciphertext getPower(seal::Evaluator& evaluator, const seal::RelinKeys &relin_keys,
                              std::vector<seal::Ciphertext>& x, int64_t e);
    
    std::vector<seal::Ciphertext> evalPoly(seal::Evaluator& evaluator, const seal::RelinKeys &relin_keys, seal::Ciphertext& x);

    std::vector<int64_t> poly_;
    int64_t p_;
    int64_t top_coef_;
    int64_t topInv_;
    int64_t extra_coef_;
    int64_t m_, k_, d_comp_;
    int64_t baby_idx_, giant_idx_, kk_;
    bool divisible_;
};

} // namespace core
} // namespace nshedb