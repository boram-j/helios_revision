#include "nshedb/predicates/predicates.h"

namespace nshedb {
namespace predicates {

using namespace seal;

Ciphertext COUNT(Evaluator &evaluator, const Ciphertext &a, int slot_count, GaloisKeys &galois_keys) {
    Ciphertext res = a;
    for(int i = 1; i < slot_count/2; i <<= 1) {
        Ciphertext tmp;
        evaluator.rotate_rows(res, i, galois_keys, tmp);
        evaluator.add_inplace(res, tmp);
    }
    Ciphertext rot;
    evaluator.rotate_columns(res, galois_keys, rot);
    evaluator.add_inplace(res, rot);
    return res;
}

Ciphertext SUM(Evaluator &evaluator, Ciphertext &a, int slot_count, GaloisKeys& galois_keys) {
    Ciphertext res = a;
    for(int i = 1; i < slot_count/2; i <<= 1) {
        Ciphertext tmp;
        evaluator.rotate_rows(res, i, galois_keys, tmp);
        evaluator.add_inplace(res, tmp);        
    }
    Ciphertext rot;
    evaluator.rotate_columns(res, galois_keys, rot);
    evaluator.add_inplace(res, rot);
    return res;
}

} // namespace predicates
} // namespace nshedb