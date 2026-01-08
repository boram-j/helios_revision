#include "nshedb/core/comparator.h"
#include "nshedb/utils/math_utils.h"
#include "nshedb/utils/conversion.h"

using namespace seal;
using namespace std;
using namespace nshedb::utils;

namespace nshedb {
namespace core {

Comparator::Comparator(const SEALContext &context) {
    p_ = context.key_context_data()->parms().plain_modulus().value();
    initCoeff();
    initPolyParam();
}

vector<Ciphertext> Comparator::isLessThan(Evaluator &evaluator, const RelinKeys &relin_keys, Ciphertext& a, Ciphertext& b) {
    Ciphertext cipher_Z;
    evaluator.sub(a, b, cipher_Z);
    return evalPoly(evaluator, relin_keys, cipher_Z);
}

vector<Ciphertext> Comparator::isLessThan(Evaluator &evaluator, const RelinKeys &relin_keys, Ciphertext& a, const Plaintext& b) {
    Ciphertext cipher_Z;
    evaluator.sub_plain(a, b, cipher_Z);
    return evalPoly(evaluator, relin_keys, cipher_Z);
}

vector<Ciphertext> Comparator::isLessThan(Evaluator &evaluator, const RelinKeys &relin_keys, const Plaintext& a, Ciphertext& b) {
    Ciphertext cipher_Z;
    evaluator.sub_plain(b, a, cipher_Z);
    evaluator.negate_inplace(cipher_Z);
    return evalPoly(evaluator, relin_keys, cipher_Z);
}

Ciphertext Comparator::isEqual(Evaluator &evaluator, const RelinKeys &relin_keys, Ciphertext& a, Ciphertext& b) {
    Ciphertext res;
    evaluator.sub(a, b, res);
    for (int i = 0; i < log2(p_-1); i++) { 
        evaluator.square_inplace(res);
        evaluator.relinearize_inplace(res, relin_keys);
    }
    Plaintext one("1");
    evaluator.negate_inplace(res);
    evaluator.add_plain_inplace(res, one);
    return res;
}

Ciphertext Comparator::isEqual(Evaluator &evaluator, const RelinKeys &relin_keys, Ciphertext& a, Plaintext& b) {
    Ciphertext res;
    evaluator.sub_plain(a, b, res);
    for (int i = 0; i < log2(p_-1); i++) { 
        evaluator.square_inplace(res);
        evaluator.relinearize_inplace(res, relin_keys);
    }
    Plaintext one("1");
    evaluator.negate_inplace(res);
    evaluator.add_plain_inplace(res, one);
    return res;
} 

void Comparator::initCoeff() {
    poly_.resize((p_-1)/2, 0);
    int64_t coeff;
    for (size_t idx = 1; idx < p_-1; idx+=2){
        coeff = 1;
        for (size_t a = 2; a <= ((p_-1)>>1); a++) {
            int64_t power = powMod(a, p_-1-idx, p_);
            coeff += power;
            if (coeff >= p_) coeff %= p_;
        }
        poly_[(idx-1)>>1] = coeff;
    }
}

void Comparator::initPolyParam() {
    d_comp_ = getDegree(poly_);
    kk_ = static_cast<int64_t>(sqrt(d_comp_/2.0));
    k_ = 1 << nextPowerOf2(kk_);
    
    if ((k_ == 16 && d_comp_ > 167) || (k_ > 16 && k_ >(1.44*kk_))) k_ /= 2;
    m_ = (d_comp_ + k_ - 1)/k_;
    
    top_coef_ = poly_[d_comp_]; 
    topInv_ = p_-8; 
    extra_coef_ = 0;
    divisible_ = (m_ * k_) == d_comp_;
    
    if(m_ != 1 << nextPowerOf2(m_)) {
        if (!divisible_) {
            top_coef_ = 1;
            topInv_ = 1;
            extra_coef_ = (top_coef_ - poly_[m_*k_]) % p_;
            poly_[m_*k_] = 1;
        }
        if (top_coef_ != 1) {
            for (size_t i=0; i < poly_.size(); i++) poly_[i] *= topInv_;
            for (size_t i=0; i <= m_*k_; i++) poly_[i] %= p_;
            poly_.resize(getDegree(poly_)+1);
        }
    }
    int64_t top_deg_ = (p_-1) >> 1;
    baby_idx_ = top_deg_ % k_;
    giant_idx_ = top_deg_ / k_;
    if (baby_idx_ == 0) {
        baby_idx_ = k_;
        giant_idx_--;
    } 
}

Ciphertext Comparator::getPower(Evaluator &evaluator, const RelinKeys &relin_keys, vector<Ciphertext>& x, int64_t e) {
    if(x[e-1].size() == 0) {   
        int64_t k = 1 << (nextPowerOf2(e)-1);
        x[e-1] = getPower(evaluator, relin_keys, x, e-k);
        evaluator.multiply_inplace(x[e-1], getPower(evaluator, relin_keys, x, k));
        evaluator.relinearize_inplace(x[e-1], relin_keys);
    }
    return x[e-1];
}

vector<Ciphertext> Comparator::evalPoly(Evaluator &evaluator, const RelinKeys &relin_keys, Ciphertext& x) {
    Ciphertext lt, eq, x2, topTerm;
    evaluator.multiply(x, x, x2);
    evaluator.relinearize_inplace(x2, relin_keys);
    vector<Ciphertext> babyStep(k_);
    vector<Ciphertext> giantStep(m_);
    babyStep[0] = x2;
    Ciphertext x2k = getPower(evaluator, relin_keys, babyStep, k_);
    giantStep[0] = x2k;
    lt = x;

    if (m_ == (1 << nextPowerOf2(m_)))
        evalPolyHelperPowerOf2(evaluator, relin_keys, lt, poly_, babyStep, giantStep);
    else {
        evalPolyHelper(evaluator, relin_keys, lt, poly_, babyStep, giantStep);
        if (top_coef_ != 1) {
            Plaintext plain_top_coef(intToHex(top_coef_, p_));
            evaluator.multiply_plain_inplace(lt, plain_top_coef);
        }
        if (extra_coef_ != 0) {
            topTerm = getPower(evaluator, relin_keys, giantStep, m_); 
            Plaintext plain_extra_coef(intToHex(extra_coef_, p_));
            evaluator.multiply_plain_inplace(topTerm, plain_extra_coef);
            evaluator.sub_inplace(lt, topTerm);
        }
    }

    evaluator.multiply_inplace(lt, x);
    evaluator.relinearize_inplace(lt, relin_keys);

    evaluator.multiply(getPower(evaluator, relin_keys, babyStep, baby_idx_),
                       getPower(evaluator, relin_keys, giantStep, giant_idx_), 
                       topTerm);
    evaluator.relinearize_inplace(topTerm, relin_keys);

    eq = topTerm; 
    Plaintext one("1");
    evaluator.negate_inplace(eq);
    evaluator.add_plain_inplace(eq, one);

    Plaintext alpha(intToHex((p_+1)>>1, p_));
    evaluator.multiply_plain_inplace(topTerm, alpha);
    evaluator.add_inplace(lt, topTerm);

    return vector<Ciphertext>{{lt, eq}};
} 

void Comparator::evalPolyHelper(Evaluator &evaluator, const RelinKeys &relin_keys, Ciphertext& x, vector<int64_t> poly, vector<Ciphertext>& babyStep, vector<Ciphertext>& giantStep) {   
    int64_t deg = getDegree(poly);
    if (deg <= babyStep.size()) {
        evalPolySimple(evaluator, relin_keys, x, poly, babyStep);
        return;
    }
    int64_t delta = deg % k_;
    int64_t n = (deg+k_-1)/k_;
    int64_t t = 1 << nextPowerOf2(n);
    
    if (n == t) {
        evalPolyHelperPowerOf2(evaluator, relin_keys, x, poly, babyStep, giantStep);
        return;
    }

    if (n == t-1 && delta==0) {
        evalPolyPS(evaluator, relin_keys, x, poly, babyStep, giantStep, t/2, delta);
        return;
    }

    t = t/2;
    int64_t u = deg - k_*(t-1);
    vector<int64_t> uu(u+1, 0);
    uu.back() = 1;

    PolyDivResult divqr = dividePoly(poly, uu, p_);
    vector<int64_t> r = divqr.r;
    vector<int64_t> q = divqr.q;

    q[0]--;
    if(u >= r.size()) r.resize(u+1, 0);
    r[u] = 1;

    evalPolyPS(evaluator, relin_keys, x, q, babyStep, giantStep, t/2, 0);

    Ciphertext tmp = getPower(evaluator, relin_keys, giantStep, u/k_);
    if (delta != 0) {
        evaluator.multiply_inplace(tmp, getPower(evaluator, relin_keys, babyStep, delta));
        evaluator.relinearize_inplace(tmp, relin_keys);
    }
    evaluator.multiply_inplace(x, tmp);
    evaluator.relinearize_inplace(x, relin_keys);

    evalPolyHelper(evaluator, relin_keys, tmp, r, babyStep, giantStep);
    evaluator.add_inplace(x, tmp);
}

void Comparator::evalPolyHelperPowerOf2(Evaluator &evaluator, const RelinKeys &relin_keys, Ciphertext& x, vector<int64_t>& poly, vector<Ciphertext>& babyStep, vector<Ciphertext>& giantStep) {
    int64_t deg = getDegree(poly);
    if (deg <= babyStep.size()) {
        evalPolySimple(evaluator, relin_keys, x, poly, babyStep);
        return;
    }

    int64_t n = (deg+k_-1)/k_;
    n = 1 << nextPowerOf2(n);

    vector<int64_t> uu(((n-1)*k_)+1, 0);
    uu.back() = 1;

    PolyDivResult divqr = dividePoly(poly, uu, p_);
    vector<int64_t> r = divqr.r;
    vector<int64_t> q = divqr.q;

    if ((n-1)*k_ >= r.size()) r.resize((n-1) * k_ + 1, 0);
    r[(n-1) * k_] = 1;
    q[0]--;
    evalPolyPS(evaluator, relin_keys, x, r, babyStep, giantStep, n/2, 0);

    Ciphertext tmp;
    evalPolySimple(evaluator, relin_keys, tmp, q, babyStep);

    for (int64_t i=1; i<n; i*=2) {
        evaluator.multiply_inplace(tmp, getPower(evaluator, relin_keys, giantStep, i));
        evaluator.relinearize_inplace(tmp, relin_keys);
    }
    evaluator.add_inplace(x, tmp);
}

void Comparator::evalPolyPS(Evaluator &evaluator, const RelinKeys &relin_keys, Ciphertext& x, vector<int64_t>& poly, vector<Ciphertext>& babyStep, vector<Ciphertext>& giantStep, int64_t t, int64_t delta) {
    int64_t deg = getDegree(poly);
    if (deg <= babyStep.size()) {
        evalPolySimple(evaluator, relin_keys, x, poly, babyStep);
        return;
    }
    vector<int64_t> uu((k_*t)+1, 0);
    uu.back() = 1;
    
    PolyDivResult divqr = dividePoly(poly, uu, p_);
    vector<int64_t> r = divqr.r; 
    vector<int64_t> q = divqr.q; 
    
    deg = getDegree(q);
    int64_t coef = r[deg];
    r[deg]--;

    PolyDivResult divcs = dividePoly(r, q, p_);
    vector<int64_t> s = divcs.r;
    vector<int64_t> c = divcs.q;

    if(deg >= s.size()) s.resize(deg + 1, 0);
    s[deg] = 1;
    for (long i=0; i<c.size(); i++) c[i] %= p_;
    c.resize(getDegree(c)+1); 
    for (long i=0; i<=s.size(); i++) s[i] %= p_;
    s.resize(getDegree(s)+1);

    evalPolyPS(evaluator, relin_keys, x, q, babyStep, giantStep, t/2, delta);
    Ciphertext tmp;
    evalPolySimple(evaluator, relin_keys, tmp, c, babyStep);
    evaluator.add_inplace(tmp, getPower(evaluator, relin_keys, giantStep, t));
    evaluator.multiply_inplace(x, tmp);
    evaluator.relinearize_inplace(x, relin_keys);

    evalPolyPS(evaluator, relin_keys, tmp, s, babyStep, giantStep, t/2, delta);
    evaluator.add_inplace(x, tmp);
}

void Comparator::evalPolySimple(Evaluator &evaluator, const RelinKeys &relin_keys, Ciphertext& x, vector<int64_t>& poly, vector<Ciphertext>& babyStep) {
    int64_t coef;
    for (int64_t i=1; i < poly.size(); i++) {
        coef = poly[i] % p_;
        if (coef > p_/2) coef -= p_;
        Ciphertext tmp = getPower(evaluator, relin_keys, babyStep, i);
        if (coef == 0) {
            evaluator.multiply_plain_inplace(tmp, Plaintext(coef));
        }
        else {
            Plaintext plain_coef(intToHex(coef, p_));
            evaluator.multiply_plain_inplace(tmp, plain_coef);
        }
        
        if (i == 1) x = tmp;
        else evaluator.add_inplace(x, tmp);
    }
    coef = poly[0] % p_;
    if (coef > p_/2) coef -= p_;
    Plaintext plain_coef(intToHex(coef, p_));
    evaluator.add_plain_inplace(x, plain_coef);
}

} // namespace core
} // namespace nshedb