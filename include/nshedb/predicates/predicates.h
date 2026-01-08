#pragma once
#include "nshedb/core/comparator.h"
#include "nshedb/utils/math_utils.h" 
#include "seal/seal.h"
#include <vector>

namespace nshedb {
namespace predicates {

using nshedb::core::Comparator;

// --- Non-Templated Prototypes (Logic in .cpp) ---
seal::Ciphertext COUNT(seal::Evaluator &evaluator, const seal::Ciphertext &a, int slot_count, seal::GaloisKeys& galois_keys);
seal::Ciphertext SUM(seal::Evaluator &evaluator, seal::Ciphertext &a, int slot_count, seal::GaloisKeys& galois_keys);

// --- Templated Implementations (Must be in Header) ---

template <typename T1, typename T2>
seal::Ciphertext LT(Comparator &comp, seal::Evaluator &eval, seal::RelinKeys &rk, T1 &a, T2 &b) {
    return comp.isLessThan(eval, rk, a, b)[0];
}

template <typename T1, typename T2>
seal::Ciphertext GT(Comparator &comp, seal::Evaluator &eval, seal::RelinKeys &rk, T1 &a, T2 &b) {
    return comp.isLessThan(eval, rk, b, a)[0]; 
}

template <typename T1, typename T2>
seal::Ciphertext LTE(Comparator &comp, seal::Evaluator &eval, seal::RelinKeys &rk, T1 &a, T2 &b) {
    auto cmp = comp.isLessThan(eval, rk, a, b);
    seal::Ciphertext res;
    eval.add(cmp[0], cmp[1], res);
    return res;
}

template <typename T1, typename T2>
seal::Ciphertext GTE(Comparator &comp, seal::Evaluator &eval, seal::RelinKeys &rk, T1 &a, T2 &b) {
    auto cmp = comp.isLessThan(eval, rk, b, a);
    seal::Ciphertext res;
    eval.add(cmp[0], cmp[1], res);
    return res;
}

template <typename T1, typename T2>
seal::Ciphertext BETWEEN(Comparator &comp, seal::Evaluator &eval, seal::RelinKeys &rk, 
                         seal::Ciphertext &a, T1 &cond1, T2 &cond2) 
{
    seal::Ciphertext gte = GTE(comp, eval, rk, a, cond1);
    seal::Ciphertext lte = LTE(comp, eval, rk, a, cond2);
    seal::Ciphertext res;
    eval.multiply(gte, lte, res);
    eval.relinearize_inplace(res, rk);
    return res;
}

template <typename T>
seal::Ciphertext IN(Comparator &comp, seal::Evaluator &eval, seal::RelinKeys &rk, 
                    seal::Ciphertext &a, std::vector<T> &set) 
{
    if(set.empty()) return seal::Ciphertext();
    std::vector<seal::Ciphertext> res(set.size());
    for(size_t i = 0; i < set.size(); i++) {
        res[i] = comp.isEqual(eval, rk, a, set[i]);
    }
    for(size_t i = 1; i < set.size(); i++) {
        eval.add_inplace(res[0], res[i]);
    }
    return res[0];
}

// Group By Templates
template <typename T>
std::vector<seal::Ciphertext> GROUPBY(Comparator &comp, seal::Evaluator &eval, seal::RelinKeys &rk,
                           seal::Ciphertext &ctxt_group, std::vector<T> &groups)
{
    std::vector<seal::Ciphertext> ctxt_res_grp(groups.size());
    for(size_t i = 0; i < groups.size(); i++){
        ctxt_res_grp[i] = comp.isEqual(eval, rk, ctxt_group, groups[i]);
    }
    return ctxt_res_grp;
}

template <typename T>
std::vector<seal::Ciphertext> GROUPBY(Comparator &comp, seal::Evaluator &eval, seal::RelinKeys &rk,
                            std::vector<seal::Ciphertext> &ctxt_group, 
                            std::vector<std::vector<T>> &groups)
{
    using nshedb::utils::generateCombinations;
    std::vector<std::vector<seal::Ciphertext>> ctxt_tmp_filter(ctxt_group.size());

    for (size_t i = 0; i < groups.size(); i++) {
        ctxt_tmp_filter[i] = GROUPBY(comp, eval, rk, ctxt_group[i], groups[i]);
    }

    auto ctxt_comb_grp = generateCombinations<seal::Ciphertext>(ctxt_tmp_filter);
    std::vector<seal::Ciphertext> ctxt_res_grp(ctxt_comb_grp.size());
    for (size_t i = 0; i < ctxt_comb_grp.size(); i++) {
        eval.multiply_many(ctxt_comb_grp[i], rk, ctxt_res_grp[i]);
    }
    return ctxt_res_grp;
}

} // namespace predicates
} // namespace nshedb