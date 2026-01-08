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
using namespace seal_examples; // For print_parameters

#define MAIL 1
#define SHIP 7
#define URGENT 1
#define HIGH 2
#define NUM_SHIP_GROUP 7
#define REQUIRED_NUM_GROUP 2
#define NUM_PRIORITY_GROUP 5

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

// ==============================================================================
// Verification Logic (TPC-H 12 Specific)
// ==============================================================================

void verify(vector<int64_t> o_orderpriority, vector<int64_t> o_orderkey, 
            vector<int64_t> l_orderkey, vector<int64_t> l_shipmode,
            vector<int64_t> l_commitdate, vector<int64_t> l_receiptdate, 
            vector<int64_t> l_shipdate, 
            vector<Ciphertext> ctxt_high_line_count, 
            vector<Ciphertext> ctxt_low_line_count,
            BatchEncoder& batch_encoder, Decryptor &decryptor) {

    map<int, int64_t> order, sol_high_line_count, sol_low_line_count;
    for(size_t i = 0; i < o_orderkey.size(); i++)
        order[o_orderkey[i]] = o_orderpriority[i];
    for(int i = 0; i < REQUIRED_NUM_GROUP; i++) {
        sol_high_line_count[i] = 0;
        sol_low_line_count[i] = 0;
    }

    for(size_t i = 0; i < l_orderkey.size(); i++) {
        if((l_shipmode[i] == MAIL || l_shipmode[i] == SHIP) && 
            (l_commitdate[i] < l_receiptdate[i]) &&
            (l_shipdate[i] < l_commitdate[i]) &&
            (l_receiptdate[i] >= toDays(Date{1994,1,1})) &&
            l_receiptdate[i] < (toDays(Date{1994,1,1})+365)) {
                if (order[l_orderkey[i]] == HIGH || order[l_orderkey[i]] == URGENT) {
                    if (l_shipmode[i] == SHIP) sol_high_line_count[1]++;
                    else  sol_high_line_count[0]++;
                }
                else {
                    if (l_shipmode[i] == SHIP) sol_low_line_count[1]++;
                    else  sol_low_line_count[0]++;
                }
            }
    }

    for(size_t i = 0; i < ctxt_high_line_count.size(); i++) {
        vector<int64_t> decrypted;
        decrypted = print_dec_<int64_t>("RESULT", 
                                    ctxt_high_line_count[i], l_shipdate.size(), 
                                    batch_encoder, decryptor);
        if(sol_high_line_count[i] != decrypted[i]) {
            printf("high_line_count res[%ld] Expected: %ld Result: %ld\n", 
                    i, sol_high_line_count[i], decrypted[i]);
            exit(1);
        }
    }

    for(size_t i = 0; i < ctxt_low_line_count.size(); i++) {
        vector<int64_t> decrypted;
        decrypted = print_dec_<int64_t>("RESULT", 
                                    ctxt_low_line_count[i], l_shipdate.size(), 
                                    batch_encoder, decryptor);
        if(sol_low_line_count[i] != decrypted[i]) {
            printf("low_line_count res[%ld] Expected: %ld Result: %ld\n", 
                    i, sol_low_line_count[i], decrypted[i]);
            exit(1);
        }
    }
}

void verify_priority(vector<int64_t>o_orderpriority, Ciphertext ctxt_p12, 
                    BatchEncoder &batch_encoder, Decryptor &decryptor) {
    vector<int64_t> decrypted;
    decrypted = print_dec_<int64_t>(" o_orderpriority = 1 or 2", 
                                    ctxt_p12, o_orderpriority.size(), 
                                    batch_encoder, decryptor);
    for (size_t i = 0; i < o_orderpriority.size(); i++) {
        bool cond = (o_orderpriority[i] == HIGH | o_orderpriority[i] == URGENT);
        if (cond!= decrypted[i]) {
            printf("verify_priority [%ld] Expected: %d Result: %ld\n", i, cond, decrypted[i]);
            exit(1);
        }
    }
    cout << "Verify... DONE " << endl;
}


void verify_filtered_orderkey(vector<int64_t> &o_orderkey, vector<int64_t> &o_orderpriority,
                        Ciphertext &ctxt_filtered_o_orderkey,
                        BatchEncoder &batch_encoder, Decryptor &decryptor)
{   
    vector<int64_t> decrypted;
    decrypted = print_dec_<int64_t>("orderpriority == 1or2", 
                                    ctxt_filtered_o_orderkey, o_orderkey.size(), 
                                    batch_encoder, decryptor);
    for(size_t i = 0; i < o_orderkey.size(); i++){
        bool cond = (o_orderpriority[i] == HIGH || o_orderpriority[i] == URGENT) *
                    o_orderkey[i];
        if (cond != decrypted[i]) {
            printf("verify_filtered_orderkey [%ld] Expected: %d Result: %ld\n", i, cond, decrypted[i]);
            exit(1);
        }
    }
}


void verify_join_keymatch(vector<int64_t> &o_orderpriority, vector<int64_t> &o_orderkey, 
                    vector<int64_t> &l_orderkey,
                    Ciphertext &ctxt_joined, 
                    BatchEncoder &batch_encoder, Decryptor &decryptor)
{
    vector<int64_t> decrypted = print_dec_<int64_t>("verify_join_keymatch", 
                                                    ctxt_joined, 
                                                    l_orderkey.size(), batch_encoder, decryptor);
    vector <int64_t> sol(l_orderkey.size(), 0);
    for(size_t i = 0; i < o_orderkey.size(); i++) {
        for(size_t j = 0; j < l_orderkey.size(); j++) {
            bool cond1 = o_orderkey[i] == l_orderkey[j];
            bool cond2 = o_orderpriority[i] == URGENT || o_orderpriority[i] == HIGH;
            if(cond1&cond2) sol[j]++;
        }
    }
    for(size_t i = 0; i < l_orderkey.size(); i++) {
        if(decrypted[i]!=sol[i]) {
            printf("verify_join_keymatch [%ld] Expected: %ld Result: %ld\n", i, sol[i], decrypted[i]);
            exit(1);
        }
    }
}


void verify_in(vector<int64_t>l_shipmode, Ciphertext ctxt_shipmode_in,
                BatchEncoder &batch_encoder, Decryptor &decryptor) {
    vector<int64_t> decrypted;
    decrypted = print_dec_<int64_t>("l_shipmode in ('MAIL'(0), 'SHIP'(6))", 
                                    ctxt_shipmode_in, l_shipmode.size(), 
                                    batch_encoder, decryptor);
    for (size_t i = 0; i < l_shipmode.size(); i++) {
        bool cond = (l_shipmode[i] == MAIL| l_shipmode[i] == SHIP);
        if (cond!= decrypted[i]) {
            printf("verify_in [%ld] Expected: %d Result: %ld\n", i, cond, decrypted[i]);
            exit(1);
        }
    }
    cout << "Verify... DONE " << endl;
}

void verify_receipt_gt_commit(vector<int64_t>l_receiptdate, vector<int64_t> l_commitdate, 
                             Ciphertext ctxt_lt_receiptdate,
                            BatchEncoder &batch_encoder, Decryptor &decryptor) {
    vector<int64_t> decrypted;
    decrypted = print_dec_<int64_t>("l_commitdate < l_receiptdate", 
                                    ctxt_lt_receiptdate, l_commitdate.size(),
                                    batch_encoder, decryptor);
    for (size_t i = 0; i < l_receiptdate.size(); i++) {
        bool cond = (l_commitdate[i] < l_receiptdate[i]);
        if (cond!= decrypted[i]) {
            printf("verify_receipt_gt_commit [%ld] Expected: %d Result: %ld\n", i, cond, decrypted[i]);
            exit(1);
        }
    }
    cout << "Verify... DONE " << endl;
}


void verify_commi_gt_ship(vector<int64_t>l_commitdate, vector<int64_t> l_shipdate, 
                          Ciphertext ctxt_lt_commitdate, 
                          BatchEncoder &batch_encoder, Decryptor &decryptor) {
    vector<int64_t> decrypted;
    decrypted = print_dec_<int64_t>("l_shipdate < l_commitdate", 
                                    ctxt_lt_commitdate, l_shipdate.size(), 
                                    batch_encoder, decryptor);
    for (size_t i = 0; i < l_commitdate.size(); i++) {
        bool cond = (l_shipdate[i] < l_commitdate[i]);
        if (cond!= decrypted[i]) {
            printf("verify_commi_gt_ship [%ld] Expected: %d Result: %ld\n", i, cond, decrypted[i]);
            exit(1);
        }
    }
    cout << "Verify... DONE " << endl;
}

void verify_receipt_gte_date(vector<int64_t>l_receiptdate, Ciphertext ctxt_gte_date, 
                             BatchEncoder &batch_encoder, Decryptor &decryptor) {
    vector<int64_t> decrypted;
    decrypted = print_dec_<int64_t>("l_receiptdate >= date '1994-01-01'", 
                                    ctxt_gte_date, l_receiptdate.size(),
                                    batch_encoder, decryptor);
    for (size_t i = 0; i < l_receiptdate.size(); i++) {
        bool cond = (l_receiptdate[i] >= toDays(Date{1994,01,01}));
        if (cond!= decrypted[i]) {
            printf("verify_receipt_gte_date [%ld] Expected: %d Result: %ld\n", i, cond, decrypted[i]);
            exit(1);
        }
    }
    cout << "Verify... DONE " << endl;
}

void verify_receipt_lt_date(vector<int64_t>l_receiptdate, Ciphertext ctxt_gte_date, 
                            BatchEncoder &batch_encoder, Decryptor &decryptor) {
    vector<int64_t> decrypted;
    decrypted = print_dec_<int64_t>("l_receiptdate < date '1994-01-01'+365", 
                                    ctxt_gte_date, l_receiptdate.size(),
                                    batch_encoder, decryptor);
    for (size_t i = 0; i < l_receiptdate.size(); i++) {
        bool cond = (l_receiptdate[i] < (toDays(Date{1994, 01, 01})+365));
        if (cond!= decrypted[i]) {
            printf("verify_receipt_lt_date [%ld] Expected: %d Result: %ld\n", i, cond, decrypted[i]);
            exit(1);
        }
    }
    cout << "Verify... DONE " << endl;
}

// ==============================================================================
// Main App
// ==============================================================================

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        printf("./tpch12 < # runs > < debug mode > < # numEle for Lineitem> <# numEle for Orders> <numThreads>\n"
            "[Debug Mode]\n0 - no debug\n"
            "1 - verify once at the end\n"
            "2 - verify each operation\n");
        exit(1);
    }

    int runs = stoi(argv[1]);
    int debug = stoi(argv[2]);
    int numEle_l = stoi(argv[3]);
    int numEle_o = stoi(argv[4]);
    int numThreads = stoi(argv[5]);
    omp_set_num_threads(numThreads);

    Timer time_parm("[PARAM SETUP] ");
    Timer time_keygen("[KEY GEN] ");
    Timer time_tpch("[TPCH-12] ");
    Timer time_groupby("[GROUP BY] ");
    Timer time_select("[SELECT] ");
    Timer time_where("[WHERE] ");
    Timer time_join("[JOIN] ");

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
    cout << "Total elements: " << numEle_l << endl;

    for(int k = 0; k < runs; k++)
    {
        vector<int64_t> o_orderkey, o_orderpriority;
        vector<int64_t> l_orderkey, l_commitdate, l_receiptdate, l_shipdate, l_shipmode;

        srand (time(NULL));
        for (int i = 0; i < numEle_l; i++) {
            l_orderkey.push_back((rand()%numEle_o)+1);
            l_commitdate.push_back(rand()%1000-3000);
            l_receiptdate.push_back(rand()%1000-3000);
            l_shipdate.push_back(rand()%1000-3000);
            l_shipmode.push_back((rand()% NUM_SHIP_GROUP)+1);
        }

        for (int i = 0; i < numEle_o; i++) {
            o_orderpriority.push_back((rand()%NUM_PRIORITY_GROUP)+1);
            o_orderkey.push_back(i+1);
        }

        Plaintext ptxt_o_orderpriority, ptxt_o_orderkey;
        Plaintext ptxt_l_orderkey, ptxt_l_commitdate, 
                  ptxt_l_receiptdate, ptxt_l_shipdate, ptxt_l_shipmode;
        Ciphertext ctxt_o_orderpriority, ctxt_o_orderkey;
        Ciphertext ctxt_l_orderkey, ctxt_l_commitdate, 
                   ctxt_l_receiptdate, ctxt_l_shipdate, ctxt_l_shipmode;
        
        batch_encoder.encode(o_orderpriority, ptxt_o_orderpriority);
        batch_encoder.encode(o_orderkey, ptxt_o_orderkey);
        batch_encoder.encode(l_orderkey, ptxt_l_orderkey);
        batch_encoder.encode(l_commitdate, ptxt_l_commitdate);
        batch_encoder.encode(l_receiptdate, ptxt_l_receiptdate);
        batch_encoder.encode(l_shipdate, ptxt_l_shipdate);
        batch_encoder.encode(l_shipmode, ptxt_l_shipmode);
        encryptor.encrypt(ptxt_o_orderpriority, ctxt_o_orderpriority);
        encryptor.encrypt(ptxt_o_orderkey, ctxt_o_orderkey);
        encryptor.encrypt(ptxt_l_orderkey, ctxt_l_orderkey);
        encryptor.encrypt(ptxt_l_commitdate, ctxt_l_commitdate);
        encryptor.encrypt(ptxt_l_receiptdate, ctxt_l_receiptdate);
        encryptor.encrypt(ptxt_l_shipdate, ctxt_l_shipdate);
        encryptor.encrypt(ptxt_l_shipmode, ctxt_l_shipmode);

        Ciphertext ctxt_priority_res, ctxt_shipmode_in, ctxt_lt_receiptdate,
                    ctxt_lt_commitdate, ctxt_gte_date, 
                    ctxt_lt_date, ctxt_priority_res_neg;
        Ciphertext ctxt_priority_urgent, ctxt_priority_high, ctxt_filtered_o_orderkey;
        Ciphertext ctxt_joined, ctxt_joined_neg;
        Ciphertext tmp_filter1, tmp_filter2, tmp_filter3, tmp_filter3_neg;
        Ciphertext ctxt_filter_neg, ctxt_filter;
        vector<Ciphertext> ctxt_key_match(numEle_o), ctxt_key_match_neg(numEle_o);
        vector<Ciphertext> ctxt_grp_res;
        vector<Ciphertext> ctxt_high_line_count(REQUIRED_NUM_GROUP),
                           ctxt_low_line_count(REQUIRED_NUM_GROUP);
        time_tpch.start();

#pragma omp parallel
#pragma omp single
{
        
    #pragma omp task
    {
        time_groupby.start();
        vector<Plaintext> ptxt_shipmode = {Plaintext(intToHex(MAIL, plaintext_modulus)),
                                           Plaintext(intToHex(SHIP, plaintext_modulus))};
        ctxt_grp_res = GROUPBY(comparator, evaluator, relin_keys, ctxt_l_shipmode, ptxt_shipmode);
        time_groupby.pause();
    }

        time_where.start();
    #pragma omp task
    {
        //l_shipmode in ('MAIL'(0), 'SHIP'(6))
        vector<Plaintext> tgt_in;
        tgt_in = {Plaintext(to_string(MAIL)), Plaintext(to_string(SHIP))};
        ctxt_shipmode_in = IN(comparator, evaluator, relin_keys, ctxt_l_shipmode, tgt_in);
    }
    #pragma omp task
    {
        //l_commitdate < l_receiptdate
        ctxt_lt_receiptdate = LT(comparator, evaluator, relin_keys, ctxt_l_commitdate, ctxt_l_receiptdate);
    }   

    #pragma omp task
    {
        //l_shipdate < l_commitdate
        ctxt_lt_commitdate = LT(comparator, evaluator, relin_keys, ctxt_l_shipdate, ctxt_l_commitdate);
    }

    #pragma omp task
    {
        //l_receiptdate >= date '1994-01-01'
        Plaintext tgt_gt_date(intToHex(toDays(Date{1994, 01, 01}), plaintext_modulus));
        ctxt_gte_date = GTE(comparator, evaluator, relin_keys, ctxt_l_receiptdate, tgt_gt_date);
    }

    #pragma omp task
    {
        //l_receiptdate < date '1994-01-01' + interval '1' year
        Plaintext tgt_lt_date(intToHex(toDays(Date{1994, 01, 01})+365, plaintext_modulus));
        ctxt_lt_date = LT(comparator, evaluator, relin_keys, ctxt_l_receiptdate, tgt_lt_date);
    }

        time_select.start();
    // o_orderpriority = 1 OR o_orderpriority = 2
    #pragma omp task
    {
        Plaintext tgt_urgent = Plaintext(to_string(URGENT));
        ctxt_priority_urgent = comparator.isEqual(evaluator, relin_keys,
                                                  ctxt_o_orderpriority, tgt_urgent);
                                             
    }
    #pragma omp task
    {
        Plaintext tgt_high = Plaintext(to_string(HIGH));
        ctxt_priority_high = comparator.isEqual(evaluator, relin_keys,
                                                ctxt_o_orderpriority, tgt_high);
    }
    #pragma omp taskwait
        evaluator.add(ctxt_priority_urgent, ctxt_priority_high, ctxt_priority_res);
        evaluator.negate(ctxt_priority_res, ctxt_priority_res_neg);
        Plaintext one("1");
        evaluator.add_plain_inplace(ctxt_priority_res_neg, one);
        time_select.pause();
        time_join.start();

    #pragma omp taskloop nogroup
        for (int i = 0; i < numEle_o; i++) {
            Plaintext ptxt_mask;
            Ciphertext ctxt_orderkey; // hold one single orderkey
            Ciphertext ctxt_priority; // hold one single filtered priority
            vector<int64_t> mask(numEle_o, 0);
            mask[i] = 1;
            batch_encoder.encode(mask, ptxt_mask);
            
            evaluator.multiply_plain(ctxt_o_orderkey, ptxt_mask, ctxt_orderkey);
            evaluator.relinearize_inplace(ctxt_orderkey, relin_keys);
            // BATCH -> SINGLE
            ctxt_orderkey = SUM(evaluator, ctxt_orderkey, slot_count, galois_keys);

            Ciphertext tmp = comparator.isEqual(evaluator, relin_keys,
                                               ctxt_l_orderkey, ctxt_orderkey);

            evaluator.multiply_plain(ctxt_priority_res, ptxt_mask, ctxt_priority);
            evaluator.relinearize_inplace(ctxt_priority, relin_keys);
            // BATCH -> SINGLE
            ctxt_priority = SUM(evaluator, ctxt_priority, slot_count, galois_keys);

            evaluator.multiply_inplace(ctxt_priority, tmp);
            evaluator.relinearize_inplace(ctxt_priority, relin_keys);
            ctxt_key_match[i] = ctxt_priority;
        }

    #pragma omp taskloop nogroup
        for (int i = 0; i < numEle_o; i++) {
            Plaintext ptxt_mask;
            Ciphertext ctxt_orderkey; // hold one single orderkey
            Ciphertext ctxt_priority; // hold one single filtered priority
            vector<int64_t> mask(numEle_o, 0);
            mask[i] = 1;
            batch_encoder.encode(mask, ptxt_mask);
            
            evaluator.multiply_plain(ctxt_o_orderkey, ptxt_mask, ctxt_orderkey);
            evaluator.relinearize_inplace(ctxt_orderkey, relin_keys);
            // BATCH -> SINGLE
            ctxt_orderkey = SUM(evaluator, ctxt_orderkey, slot_count, galois_keys);

            Ciphertext tmp = comparator.isEqual(evaluator, relin_keys,
                                               ctxt_l_orderkey, ctxt_orderkey);

            
            evaluator.multiply_plain(ctxt_priority_res_neg, ptxt_mask, ctxt_priority);
            evaluator.relinearize_inplace(ctxt_priority, relin_keys);
            // BATCH -> SINGLE
            ctxt_priority = SUM(evaluator, ctxt_priority, slot_count, galois_keys);

            evaluator.multiply_inplace(ctxt_priority, tmp);
            evaluator.relinearize_inplace(ctxt_priority, relin_keys);
            ctxt_key_match_neg[i] = ctxt_priority;
        }

    #pragma omp taskwait
        
        for(size_t i = 1; i < ctxt_key_match.size(); i++) {
            evaluator.add_inplace(ctxt_key_match[0], ctxt_key_match[i]);
        }
        ctxt_joined = ctxt_key_match[0];

        for(size_t i = 1; i < ctxt_key_match_neg.size(); i++) {
            evaluator.add_inplace(ctxt_key_match_neg[0], ctxt_key_match_neg[i]);
        }
        ctxt_joined_neg = ctxt_key_match_neg[0];
        time_join.pause();

        //Combine all predicates
    #pragma omp task 
    {
        evaluator.multiply(ctxt_shipmode_in, ctxt_lt_receiptdate, tmp_filter1);
        evaluator.relinearize_inplace(tmp_filter1, relin_keys);
    }

    #pragma omp task 
    {
        evaluator.multiply(ctxt_lt_commitdate, ctxt_gte_date, tmp_filter2);
        evaluator.relinearize_inplace(tmp_filter2, relin_keys);
    }

    #pragma omp task 
    {
        evaluator.multiply(ctxt_joined, ctxt_lt_date, tmp_filter3);
        evaluator.relinearize_inplace(tmp_filter3, relin_keys);
    }

    #pragma omp task 
    {
        evaluator.multiply(ctxt_joined_neg, ctxt_lt_date, tmp_filter3_neg);
        evaluator.relinearize_inplace(tmp_filter3_neg, relin_keys);
    }

    #pragma omp taskwait

    #pragma omp task
    {
        evaluator.multiply(tmp_filter3, tmp_filter2, ctxt_filter);
        evaluator.relinearize_inplace(ctxt_filter, relin_keys);
    }

    #pragma omp task
    {
        evaluator.multiply(tmp_filter3_neg, tmp_filter2, ctxt_filter_neg);
        evaluator.relinearize_inplace(ctxt_filter_neg, relin_keys);
    }

    #pragma omp taskwait

    #pragma omp task
    {   
        evaluator.multiply(tmp_filter1, ctxt_filter_neg, ctxt_filter_neg);
        evaluator.relinearize_inplace(ctxt_filter_neg, relin_keys);
    }

    #pragma omp task
    {   
        evaluator.multiply(tmp_filter1, ctxt_filter, ctxt_filter);
        evaluator.relinearize_inplace(ctxt_filter, relin_keys);
    }
    #pragma omp taskwait
        time_where.pause();
        time_groupby.start();
    #pragma omp taskloop nogroup
        for(size_t i = 0; i < ctxt_grp_res.size(); i++) {
            Ciphertext tmp;
            evaluator.multiply(ctxt_grp_res[i], ctxt_filter, tmp);
            evaluator.relinearize_inplace(tmp, relin_keys);
            ctxt_high_line_count[i] = tmp;
        }
    #pragma omp taskloop nogroup
        for(size_t i = 0; i < ctxt_grp_res.size(); i++) {
            Ciphertext tmp;
            evaluator.multiply(ctxt_grp_res[i], ctxt_filter_neg, tmp);
            evaluator.relinearize_inplace(tmp, relin_keys);
            ctxt_low_line_count[i] = tmp;
        }    
    #pragma omp taskwait
        time_groupby.pause();
        
        time_select.start();
    #pragma omp taskloop nogroup
        for(size_t i = 0; i < ctxt_high_line_count.size(); i++) {
            ctxt_high_line_count[i] = SUM(evaluator, ctxt_high_line_count[i], 
                                          slot_count, galois_keys);
        }
    #pragma omp taskloop nogroup
        for(size_t i = 0; i < ctxt_low_line_count.size(); i++) {
            ctxt_low_line_count[i] = SUM(evaluator, ctxt_low_line_count[i], 
                                          slot_count, galois_keys);
        }
    #pragma omp taskwait
    time_select.pause();
    time_tpch.pause();
}
        if(debug == 2) {
            time_tpch.pause();
            verify_priority(o_orderpriority, ctxt_priority_res, batch_encoder, decryptor);
            verify_in(l_shipmode, ctxt_shipmode_in, batch_encoder, decryptor);
            verify_receipt_gt_commit(l_receiptdate, l_commitdate, ctxt_lt_receiptdate, 
                                    batch_encoder, decryptor);
            verify_commi_gt_ship(l_commitdate, l_shipdate, ctxt_lt_commitdate, 
                                    batch_encoder, decryptor);               
            verify_receipt_gte_date(l_receiptdate, ctxt_gte_date, 
                                    batch_encoder, decryptor);             
            verify_receipt_lt_date(l_receiptdate, ctxt_lt_date, 
                                    batch_encoder, decryptor);
            verify_join_keymatch(o_orderpriority, o_orderkey, l_orderkey, ctxt_joined,
                            batch_encoder, decryptor);
            time_tpch.start();
        }

        if(debug != 0) {
            verify(o_orderpriority, o_orderkey, l_orderkey, l_shipmode,
                    l_commitdate, l_receiptdate, l_shipdate, 
                    ctxt_high_line_count, ctxt_low_line_count, batch_encoder, decryptor);
        }
        
    }
    time_select.stop(runs);
    time_where.stop(runs);
    time_join.stop(runs);
    time_groupby.stop(runs);
    time_tpch.stop(runs);
    if(debug != 0)
        cout << "Verified. Done.\n";
    else
        cout << "Verify skipped. Done" << endl;
    return 0;
}