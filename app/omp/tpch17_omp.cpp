#include "nshedb/nshedb.h"
#include "nshedb/utils/seal_examples.h"
#include "omp.h"
#include <vector>
#include <iostream>
#include <iomanip>

using namespace seal;
using namespace std;
using namespace nshedb;
using namespace nshedb::core;
using namespace nshedb::predicates;
using namespace nshedb::utils;
using namespace seal_examples;

#define BRAND23 23
#define MED_BOX 12
#define NUM_BRAND 25
#define NUM_CONTAINER 40

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
// Verification Logic (TPC-H 17 Specific)
// ==============================================================================

// SELECT SUM(L_EXTENDEDPRICE)/7.0 AS AVG_YEARLY FROM LINEITEM, PART
// WHERE P_PARTKEY = L_PARTKEY AND P_BRAND = 'Brand#23' AND P_CONTAINER = 'MED BOX'
// AND L_QUANTITY < (SELECT 0.2*AVG(L_QUANTITY) FROM LINEITEM WHERE L_PARTKEY = P_PARTKEY)
void verify(vector<int64_t> l_partkey, vector<int64_t> l_extendedprice, vector<int64_t> l_quantity, 
            int64_t l_avg_quantity, vector<int64_t> p_brand, vector<int64_t> p_container,
            Ciphertext ctxt_sum_yearly, Ciphertext ctxt_cnt,
            Decryptor &decryptor, BatchEncoder &batch_encoder) {
    int64_t sum=0, cnt=0;

    for(size_t i = 0; i < l_partkey.size(); i++) {
        int idx = l_partkey[i]-1;
        bool cond = p_brand[idx] == BRAND23 && 
                    p_container[idx] == MED_BOX &&
                    (l_quantity[i]*10) < (2*l_avg_quantity);
        if(cond) {
            sum += l_extendedprice[i];
            cnt += 1;
        }
    }
    int64_t res_sum = print_dec_<int64_t>("SUM_YEARLY", ctxt_sum_yearly, l_partkey.size(), 
                                               batch_encoder, decryptor)[0];
    int64_t res_cnt = print_dec_<int64_t>("SUM_CNT", ctxt_cnt, l_partkey.size(), 
                                               batch_encoder, decryptor)[0];
    if(cnt != res_cnt) {
        printf("PROMO SUM Expected: %ld Result: %ld\n", cnt, res_cnt);
        exit(1);
    }
    if(sum != res_sum) {
        printf("TOTAL SUM Expected: %ld Result: %ld\n", sum, res_sum);
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
        printf("./tpch17 < # runs > < debug mode > < # numEle > < # numEle for Part> <numThreads>\n"
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
    Timer time_tpch("[TPCH-17] ");
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
        vector<int64_t> l_extendedprice, l_quantity, l_partkey;
        vector<int64_t> p_brand, p_partkey, p_container, p_type;
        int64_t l_avg_quantity = 0;

        srand (time(NULL));
        for (int i = 0; i < numEle; i++) {
            l_quantity.push_back(rand()%50);
            l_extendedprice.push_back(rand()%201);
            l_partkey.push_back((rand()%numEle_p)+1);
            l_avg_quantity += l_quantity[i];
        }
        l_avg_quantity /= numEle;

        for (int i = 0; i < numEle_p; i++) {
            p_partkey.push_back(i+1);
            p_brand.push_back((rand()%NUM_BRAND)+1);
            p_container.push_back((rand()%NUM_CONTAINER)+1);
            p_type.push_back((rand()%2));
        }

        Plaintext ptxt_l_quantity, ptxt_l_extendedprice, ptxt_l_partkey, 
                  ptxt_l_avg_quantity(intToHex(l_avg_quantity, plaintext_modulus));
        Plaintext ptxt_p_partkey, ptxt_p_brand, ptxt_p_container;
        Ciphertext ctxt_l_quantity, ctxt_l_extendedprice, ctxt_l_partkey, ctxt_l_avg_qty;
        Ciphertext ctxt_p_partkey, ctxt_p_brand, ctxt_p_container;
                  
        batch_encoder.encode(l_quantity, ptxt_l_quantity);
        batch_encoder.encode(l_extendedprice, ptxt_l_extendedprice);
        batch_encoder.encode(l_partkey, ptxt_l_partkey);
        batch_encoder.encode(p_brand, ptxt_p_brand);
        batch_encoder.encode(p_partkey, ptxt_p_partkey);
        batch_encoder.encode(p_container, ptxt_p_container);

        encryptor.encrypt(ptxt_l_quantity, ctxt_l_quantity);
        encryptor.encrypt(ptxt_l_extendedprice, ctxt_l_extendedprice);
        encryptor.encrypt(ptxt_l_partkey, ctxt_l_partkey);
        encryptor.encrypt(ptxt_p_container, ctxt_p_container);
        encryptor.encrypt(ptxt_p_partkey, ctxt_p_partkey);
        encryptor.encrypt(ptxt_p_brand, ctxt_p_brand);
        encryptor.encrypt(ptxt_l_avg_quantity, ctxt_l_avg_qty);
        
        Plaintext tgt_brand(intToHex(BRAND23,plaintext_modulus));
        Plaintext tgt_container(intToHex(MED_BOX, plaintext_modulus));
        Plaintext ptxt_two("2");
        Plaintext ptxt_ten(intToHex(10, plaintext_modulus));
        Ciphertext ctxt_lt_avg, ctxt_scaled_qty,
                   ctxt_eq_brand_23, ctxt_eq_med_box, ctxt_filter;
        vector<int64_t> mask(numEle_p, 0);
        vector<Ciphertext> ctxt_key_match(numEle_p);
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
        evaluator.multiply_plain_inplace(ctxt_l_avg_qty, ptxt_two);
        evaluator.multiply_plain(ctxt_l_quantity, ptxt_ten, ctxt_scaled_qty);
        ctxt_lt_avg = LT(comparator, evaluator, relin_keys, 
                         ctxt_scaled_qty, ctxt_l_avg_qty);
    }
    #pragma omp task
    {
        ctxt_eq_brand_23 = comparator.isEqual(evaluator, relin_keys, 
                                              ctxt_p_brand, tgt_brand);
    }
    #pragma omp task
    {
        ctxt_eq_med_box = comparator.isEqual(evaluator, relin_keys,
                                        ctxt_p_container, tgt_container);
    }

    #pragma omp taskwait
        evaluator.multiply(ctxt_eq_brand_23, ctxt_eq_med_box, ctxt_filter);
        evaluator.relinearize_inplace(ctxt_filter, relin_keys);

        // p_partkey = l_partkey
        
        #pragma omp taskloop nogroup
        for(int i = 0; i < numEle_p; i++) {
            Plaintext ptxt_mask;
            Ciphertext ctxt_masked_partkey, ctxt_masked_filter;
            mask[i] = 1;
            batch_encoder.encode(mask, ptxt_mask);
            evaluator.multiply_plain(ctxt_p_partkey, ptxt_mask, ctxt_masked_partkey);
            // BATCH -> SINGLE
            ctxt_masked_partkey = SUM(evaluator, ctxt_masked_partkey, slot_count, galois_keys);
            ctxt_key_match[i] = comparator.isEqual(evaluator, relin_keys,
                                                   ctxt_masked_partkey, ctxt_l_partkey);
            evaluator.multiply_inplace(ctxt_key_match[i], ctxt_lt_avg);
            evaluator.relinearize_inplace(ctxt_key_match[i], relin_keys);
            
            // BATCH -> SINGLE
            evaluator.multiply_plain(ctxt_filter, ptxt_mask, ctxt_masked_filter);
            ctxt_masked_filter = SUM(evaluator, ctxt_masked_filter, slot_count, galois_keys);
            
            evaluator.multiply_inplace(ctxt_key_match[i], ctxt_masked_filter);
            evaluator.relinearize_inplace(ctxt_key_match[i], relin_keys);
            mask[i] = 0;
        }
    #pragma omp taskwait
}
        Ciphertext ctxt_sum_yearly, ctxt_cnt;
        for(size_t i = 1; i < ctxt_key_match.size(); i++) 
            evaluator.add_inplace(ctxt_key_match[0], ctxt_key_match[i]);
        evaluator.multiply(ctxt_key_match[0], ctxt_l_extendedprice, ctxt_sum_yearly);
        evaluator.relinearize_inplace(ctxt_sum_yearly, relin_keys);
        time_filter.pause();
        time_agg.start();
        ctxt_cnt = COUNT(evaluator, ctxt_key_match[0], slot_count, galois_keys);
        ctxt_sum_yearly = SUM(evaluator, ctxt_sum_yearly, slot_count, galois_keys);
        time_agg.pause();
        time_tpch.pause();

        if(debug != 0) {
            verify(l_partkey, l_extendedprice, l_quantity, l_avg_quantity,
                   p_brand, p_container, 
                   ctxt_sum_yearly, ctxt_cnt, decryptor, batch_encoder);
        }
        cout << "    + noise budget left: " << decryptor.invariant_noise_budget(ctxt_sum_yearly) << " bits"
         << endl;
        cout << "    + noise budget left: " << decryptor.invariant_noise_budget(ctxt_cnt) << " bits"
         << endl;
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