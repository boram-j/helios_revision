#include "nshedb/nshedb.h"
#include "nshedb/utils/seal_examples.h"
#include "omp.h"
#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

using namespace seal;
using namespace std;
using namespace nshedb;
using namespace nshedb::core;
using namespace nshedb::predicates;
using namespace nshedb::utils;
using namespace seal_examples;

#define PROMO 1

// ==============================================================================
// Verification Helpers
// ==============================================================================

template <typename T>
inline std::vector<T> dec(const seal::Ciphertext &ctxt_res, int numEle, seal::BatchEncoder& batch_encoder, seal::Decryptor& decryptor) {
    seal::Plaintext ptxt_res;
    std::vector<T> res;
    decryptor.decrypt(ctxt_res, ptxt_res);
    batch_encoder.decode(ptxt_res, res);
    return res;
}

template <typename T>
inline std::vector<T> print_dec_(std::string st, const seal::Ciphertext &ctxt_res, int numEle, seal::BatchEncoder& batch_encoder, seal::Decryptor& decryptor) {
    auto res = dec<T>(ctxt_res, numEle, batch_encoder, decryptor);
    std::cout << st << ":" << std::setw(3) << "[";
    for (int i = 0; i < numEle; i++)
    {
        std::cout << std::setw(3) << std::right << res[i] << ",";
    }
    std::cout << "] " << std::endl;
    std::cout << "    + noise budget: " << decryptor.invariant_noise_budget(ctxt_res) << " bits"
         << std::endl;
    return res;
}

inline void print_vec(std::string st, std::vector<int64_t> vec, int numEle)
{
    std::cout << st << ":" << std::setw(3) << "[";
    for (std::size_t i = 0; i < numEle; i++)
    {
        std::cout << std::setw(3) << std::right << vec[i] << ",";
    }
    std::cout << "] " << std::endl;
}

// ==============================================================================
// Verification Logic (TPC-H 14 Specific)
// ==============================================================================

// SELECT
//     100.00 * sum(case
//         when p_type like 'PROMO%'
//             then l_extendedprice * (1 - l_discount)
//         else 0
//     end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
// FROM
//     lineitem,
//     part
// WHERE
//     l_partkey = p_partkey
//     AND l_shipdate >= date '1995-09-01'
//     AND l_shipdate < date '1995-09-01' + interval '1' month;

void verify(vector<int64_t> l_shipdate, vector<int64_t> l_charge, vector<int64_t> l_partkey, 
            vector<int64_t> p_partkey, vector<int64_t> p_type, 
            Ciphertext ctxt_promo, Ciphertext ctxt_total,
            Decryptor &decryptor, BatchEncoder &batch_encoder) {
    int64_t total=0, promo=0;

    print_vec("l_charge", l_charge, l_charge.size());
    print_vec("l_shipdate", l_shipdate, l_shipdate.size());
    cout << "toDays(Date{1995,9,1}): " << toDays(Date{1995,9,1}) << endl;
    cout << "toDays(Date{1995,10,1}): " << toDays(Date{1995,10,1}) << endl;
    
    for(size_t i = 0; i < l_partkey.size(); i++) {
        bool cond = (l_shipdate[i] >= toDays(Date{1995,9,1})) &&
                    (l_shipdate[i] < toDays(Date{1995,10,1}));
        if(cond) {
            promo += l_charge[i] * p_type[l_partkey[i]-1];
            total += l_charge[i];
        }
    }

    int64_t res_promo = print_dec_<int64_t>("SUM_PROMO", ctxt_promo, l_charge.size(), 
                                               batch_encoder, decryptor)[0];
    int64_t res_total = print_dec_<int64_t>("SUM_TOTAL", ctxt_total, l_charge.size(), 
                                               batch_encoder, decryptor)[0];
    if(promo != res_promo) {
        printf("PROMO SUM Expected: %ld Result: %ld\n", promo, res_promo);
        exit(1);
    }
    if(total != res_total) {
        printf("TOTAL SUM Expected: %ld Result: %ld\n", total, res_total);
        exit(1);
    }
}


// ==============================================================================
// Main App
// ==============================================================================

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        printf("./tpch14 < # runs > < debug mode > < # numEle > < # numEle for Part> <numThreads>\n"
            "[Debug Mode]\n0 - no debug\n"
            "1 - verify once at the end\n"
            "2 - verify each operation\n");
        exit(1);
    }

    int runs = stoi(argv[1]);
    int debug = stoi(argv[2]);
    int numEle = stoi(argv[3]);
    int numEle_p = stoi(argv[4]);
    int numThreads = stoi(argv[5]);
    omp_set_num_threads(numThreads);

    Timer time_parm("[PARAM SETUP] ");
    Timer time_keygen("[KEY GEN] ");
    Timer time_tpch("[TPCH-14] ");
    Timer time_filter("[FILTER] ");
    Timer time_agg("[AGG] ");

    time_parm.start();
    EncryptionParameters parms(scheme_type::bfv);
    size_t poly_modulus_degree = 16384*2; //32768
    size_t plaintext_modulus = 65537;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
    parms.set_plain_modulus(plaintext_modulus);
    SEALContext context(parms);
    time_parm.stop();


    time_keygen.start();
    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key;
    RelinKeys relin_keys;
    GaloisKeys galois_keys;
    keygen.create_public_key(public_key);
    keygen.create_relin_keys(relin_keys);
    keygen.create_galois_keys(galois_keys);
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Comparator comparator(context);
    BatchEncoder batch_encoder(context);
    Decryptor decryptor(context, secret_key);
    time_keygen.stop();

    print_parameters(context);
    cout << endl;

    int slot_count = batch_encoder.slot_count();
    cout << "Total slots: " << slot_count << endl;
    cout << "Total elements: " << numEle << endl;

   for(int k = 0; k < runs; k++)
    {

        /*
         Prepare Data...
        */
        vector<int64_t> l_shipdate, l_extendedprice, l_discount, 
                        l_partkey, l_charge;
        vector<int64_t> p_type, p_partkey;

        srand (time(NULL));
        for (int i = 0; i < numEle; i++) {
            l_shipdate.push_back(rand()%2000 - 3000);
            l_extendedprice.push_back(rand()%201);
            l_discount.push_back(rand()%11);
            l_partkey.push_back((rand()%numEle_p)+1);
            l_charge.push_back(l_extendedprice[i]*(100-l_discount[i])/100);
        }

        for (int i = 0; i < numEle_p; i++) {
            p_partkey.push_back(i+1);
            p_type.push_back((rand()%2));
        }

        Plaintext ptxt_l_shipdate, ptxt_l_extendedprice, 
                  ptxt_l_discount, ptxt_l_partkey, ptxt_l_charge;
        Plaintext ptxt_p_partkey, ptxt_p_type;
        Ciphertext ctxt_l_shipdate, ctxt_l_extendedprice, 
                  ctxt_l_discount, ctxt_l_partkey, ctxt_l_charge;
        Ciphertext ctxt_p_partkey, ctxt_p_type;
                  
        batch_encoder.encode(l_shipdate, ptxt_l_shipdate);
        batch_encoder.encode(l_extendedprice, ptxt_l_extendedprice);
        batch_encoder.encode(l_discount, ptxt_l_discount);
        batch_encoder.encode(l_partkey, ptxt_l_partkey);
        batch_encoder.encode(l_charge, ptxt_l_charge);
        batch_encoder.encode(p_type, ptxt_p_type);
        batch_encoder.encode(p_partkey, ptxt_p_partkey);

        encryptor.encrypt(ptxt_l_shipdate, ctxt_l_shipdate);
        encryptor.encrypt(ptxt_l_extendedprice, ctxt_l_extendedprice);
        encryptor.encrypt(ptxt_l_discount, ctxt_l_discount);
        encryptor.encrypt(ptxt_l_partkey, ctxt_l_partkey);
        encryptor.encrypt(ptxt_l_charge, ctxt_l_charge);
        encryptor.encrypt(ptxt_p_partkey, ctxt_p_partkey);
        encryptor.encrypt(ptxt_p_type, ctxt_p_type);
        
        Plaintext tgt_date1(intToHex(toDays(Date{1995,9,1}),plaintext_modulus));
        Plaintext tgt_date2(intToHex(toDays(Date{1995,10,1}),plaintext_modulus));
        Ciphertext gte_tgt_date1, lt_tgt_date2, ctxt_filter;
        vector<int64_t> mask(numEle_p, 0);
        vector<Ciphertext> ctxt_key_match(numEle_p);
        Ciphertext ctxt_partkey;
        Ciphertext ctxt_total, ctxt_promo, ctxt_joined;
        /*
         Query Starts...
        */
        time_tpch.start();
        time_filter.start();
#pragma omp parallel
#pragma omp single
{
        
    #pragma omp task
    {
        // l_shipdate >= date '1995-09-01'
        gte_tgt_date1 = GTE(comparator, evaluator, relin_keys, ctxt_l_shipdate, tgt_date1);
    }
    #pragma omp task
    {
        // l_shipdate < date '1995-09-01' + interval '1' month;
        lt_tgt_date2 = LT(comparator, evaluator, relin_keys, ctxt_l_shipdate, tgt_date2);
    }

    #pragma omp task
    {
        evaluator.multiply(ctxt_p_partkey, ctxt_p_type, ctxt_partkey);
        evaluator.relinearize_inplace(ctxt_partkey, relin_keys);
    }

    #pragma omp taskwait
        evaluator.multiply(gte_tgt_date1, lt_tgt_date2, ctxt_filter);
        evaluator.relinearize_inplace(ctxt_filter, relin_keys);
        

        // l_partkey = p_partkey
        #pragma omp taskloop nogroup
        for(int i = 0; i < numEle_p; i++) {
            Plaintext ptxt_mask;
            Ciphertext ctxt_masked_partkey;
            mask[i] = 1;
            batch_encoder.encode(mask, ptxt_mask);
            evaluator.multiply_plain(ctxt_partkey, ptxt_mask, ctxt_masked_partkey);
            // BATCH -> SINGLE
            ctxt_masked_partkey = SUM(evaluator, ctxt_masked_partkey, slot_count, galois_keys);
            ctxt_key_match[i] = comparator.isEqual(evaluator, relin_keys,
                                                   ctxt_masked_partkey, ctxt_l_partkey);
            mask[i] = 0;
        }
        #pragma omp taskwait
        for(size_t i = 1; i < ctxt_key_match.size(); i++) 
            evaluator.add_inplace(ctxt_key_match[0], ctxt_key_match[i]);

        ctxt_joined = ctxt_key_match[0];
        evaluator.multiply(ctxt_filter, ctxt_joined, ctxt_joined);
        evaluator.relinearize_inplace(ctxt_joined, relin_keys);
        time_filter.pause();

        time_agg.start();
    #pragma omp task
    {
        evaluator.multiply_inplace(ctxt_joined, ctxt_l_charge);
        evaluator.relinearize_inplace(ctxt_joined, relin_keys);
        ctxt_promo = SUM(evaluator, ctxt_joined, slot_count, galois_keys);
    }
    #pragma omp task
    {
        evaluator.multiply_inplace(ctxt_filter, ctxt_l_charge);
        evaluator.relinearize_inplace(ctxt_filter, relin_keys);
        ctxt_total = SUM(evaluator, ctxt_filter, slot_count, galois_keys);
    }
        time_agg.pause();
        time_tpch.pause();
}
        if(debug != 0) {
            verify(l_shipdate, l_charge, l_partkey, p_partkey, p_type, 
                    ctxt_promo, ctxt_total, decryptor, batch_encoder);
        }
    }
    
    time_tpch.stop(runs);
    time_agg.stop(runs);
    time_filter.stop(runs);
    
    if(debug != 0)
        cout << "Verified. Done.\n";
    else
        cout << "Verify skipped. Done" << endl;
    return 0;
}