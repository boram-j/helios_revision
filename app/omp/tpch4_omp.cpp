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
// Verification Logic (TPC-H 4 Specific)
// ==============================================================================

/*
    SELECT O_ORDERPRIORITY, COUNT(*) AS ORDER_COUNT FROM ORDERS
    WHERE O_ORDERDATE >= '1993-07-01' 
          AND O_ORDERDATE < dateadd(mm,3, cast('1993-07-01' as date))
          AND EXISTS (SELECT * FROM LINEITEM 
                      WHERE L_ORDERKEY = O_ORDERKEY 
                            AND L_COMMITDATE < L_RECEIPTDATE)
    GROUP BY O_ORDERPRIORITY
    ORDER BY O_ORDERPRIORITY
*/

void verify(vector<int64_t>& o_orderpriority, vector<int64_t>& o_orderdate, 
            vector<int64_t>& l_orderkey, vector<int64_t>& o_orderkey,
            vector<int64_t>& l_commitdate, vector<int64_t>& l_receiptdate, 
            vector<Ciphertext>& ctxt_order_count, 
            BatchEncoder& batch_encoder, Decryptor& decryptor) {

    map<int, int64_t> order_date, order_priority, sol_order_count;
    for(size_t i = 0; i < o_orderdate.size(); i++) {
        order_date[o_orderkey[i]] = o_orderdate[i];
        order_priority[o_orderkey[i]] = o_orderpriority[i];
    }
    for(int i = 1; i <= NUM_PRIORITY_GROUP; i++)
        sol_order_count[i] = 0;

    for(size_t i = 0; i < l_commitdate.size(); i++) {
        int cond = (l_commitdate[i] < l_receiptdate[i]) &&
                    (order_date[l_orderkey[i]] >= toDays(Date{1993, 7, 1})) &&
                    (order_date[l_orderkey[i]] < toDays(Date{1993, 10, 1}));
        if(cond)
            sol_order_count[order_priority[l_orderkey[i]]]++;
    }
    
    for (int i = 0; i < NUM_PRIORITY_GROUP; i++) {
        vector<int64_t> decrypted = print_dec_<int64_t>("VERIFY", 
                                    ctxt_order_count[i], l_orderkey.size(),
                                    batch_encoder, decryptor);
        if (sol_order_count[i+1]!= decrypted[0]){
            printf("[%d] Expected: %ld Result: %ld\n", 
                    i, sol_order_count[i+1], decrypted[0]);
            exit(1);
        }
    }
}


void verify_filtered_ordertable(vector<int64_t> &o_orderpriority, vector<int64_t> &o_orderdate,
                                vector<Ciphertext> &ctxt_grp_resp, 
                                BatchEncoder &batch_encoder, Decryptor &decryptor) {

    vector<vector<int64_t>> group(NUM_PRIORITY_GROUP);
    for (int i = 0; i < NUM_PRIORITY_GROUP; i++) {
        group[i].resize(o_orderpriority.size());
    }
    for(size_t i = 0; i < o_orderdate.size(); i++) {
        int cond = (o_orderdate[i] >= toDays(Date{1993, 7, 1})) &&
                   (o_orderdate[i] < toDays(Date{1993, 10, 1}));
        if (cond) group[o_orderpriority[i]-1][i]+=1;
    }
    
    for (int i = 0; i < NUM_PRIORITY_GROUP; i++) {
        vector<int64_t> decrypted = print_dec_<int64_t>("verify_filtered_ordertable", 
                                    ctxt_grp_resp[i], o_orderdate.size(),
                                    batch_encoder, decryptor);
        for(size_t j = 0; j < o_orderdate.size(); j++) {
            if (group[i][j]!= decrypted[j]){
                printf("[%d][%ld] Expected: %ld Result: %ld\n", 
                        i, j, group[i][j], decrypted[j]);
                exit(1);
            }
        }
    }
}


void verify_exist(vector<int64_t> &l_commitdate, vector<int64_t> &l_receiptdate,
                  vector<int64_t> &l_orderkey, int orderkey,
                  Ciphertext &ctxt_key_match, 
                  BatchEncoder &batch_encoder, Decryptor &decryptor) {

    vector<int64_t> decrypted = print_dec_<int64_t>("verify_exist", 
                                ctxt_key_match, l_orderkey.size(),
                                batch_encoder, decryptor);
    for(size_t i = 0; i < l_commitdate.size(); i++) {
        bool cond = l_commitdate[i] < l_receiptdate[i] && l_orderkey[i] == orderkey;
        if(cond != decrypted[i]) {
            printf("[%ld] Expected: %d Result: %ld\n", 
                    i, cond, decrypted[i]);
            exit(1);
        }
    }

}

// ==============================================================================
// Main App
// ==============================================================================

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        printf("./tpch4 < # runs > < debug mode > < # numEle for Lineitem> <# numEle for Orders> <numThread>\n"
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
    Timer time_tpch("[TPCH-4] ");
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
    Comparator comparator(context);
    Evaluator evaluator(context);
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
        vector<int64_t> o_orderkey, o_orderpriority, o_orderdate;
        vector<int64_t> l_orderkey, l_commitdate, l_receiptdate;

        srand (time(NULL));
        for (int i = 0; i < numEle_l; i++) {
            l_orderkey.push_back((rand()%numEle_o)+1);
            l_commitdate.push_back(rand()%1000-3000);
            l_receiptdate.push_back(rand()%1000-3000);
        }

        for (int i = 0; i < numEle_o; i++) {
            o_orderpriority.push_back((rand()%NUM_PRIORITY_GROUP)+1);
            o_orderkey.push_back(i+1);
            o_orderdate.push_back(rand()%1000-3000);
        }


        if(debug == 2) {
            print_vec("PRIORITY", o_orderpriority, o_orderdate.size());
            print_vec("DATE", o_orderdate, o_orderdate.size());
            print_vec("l_commitdate", l_commitdate, l_commitdate.size());
            print_vec("l_receiptdate", l_receiptdate, l_receiptdate.size());
            print_vec("l_orderkey", l_orderkey, l_orderkey.size());
        }

        Plaintext ptxt_o_orderpriority, ptxt_o_orderkey, ptxt_o_orderdate;
        Plaintext ptxt_l_orderkey, ptxt_l_commitdate, 
                  ptxt_l_receiptdate;
        Ciphertext ctxt_o_orderpriority, ctxt_o_orderkey, ctxt_o_orderdate;
        Ciphertext ctxt_l_orderkey, ctxt_l_commitdate, 
                   ctxt_l_receiptdate;

        batch_encoder.encode(o_orderpriority, ptxt_o_orderpriority);
        batch_encoder.encode(o_orderkey, ptxt_o_orderkey);
        batch_encoder.encode(o_orderdate, ptxt_o_orderdate);
        batch_encoder.encode(l_orderkey, ptxt_l_orderkey);
        batch_encoder.encode(l_commitdate, ptxt_l_commitdate);
        batch_encoder.encode(l_receiptdate, ptxt_l_receiptdate);
        
        encryptor.encrypt(ptxt_o_orderpriority, ctxt_o_orderpriority);
        encryptor.encrypt(ptxt_o_orderkey, ctxt_o_orderkey);
        encryptor.encrypt(ptxt_o_orderdate, ctxt_o_orderdate);
        encryptor.encrypt(ptxt_l_orderkey, ctxt_l_orderkey);
        encryptor.encrypt(ptxt_l_commitdate, ctxt_l_commitdate);
        encryptor.encrypt(ptxt_l_receiptdate, ctxt_l_receiptdate);

        vector<Ciphertext> ctxt_order_count(NUM_PRIORITY_GROUP);//, ctxt_zero);
        vector<Ciphertext> ctxt_grp_res;
        Ciphertext ctxt_gte_date, ctxt_lt_date, ctxt_exist;
        Ciphertext ctxt_eq_orderkey, ctxt_lt_receiptdate;
        Ciphertext tmp_filter;

        time_tpch.start();
#pragma omp parallel
#pragma omp single
{
    #pragma omp task
    {
        vector<Plaintext> ptxt_priority_group;
        for (int i = 1; i <= NUM_PRIORITY_GROUP; i++)
            ptxt_priority_group.push_back(Plaintext(to_string(i)));                 
        ctxt_grp_res = GROUPBY(comparator, evaluator, relin_keys, 
                               ctxt_o_orderpriority, ptxt_priority_group);
    }
    
    #pragma omp task
    {
        //O_ORDERDATE >= '1993-07-01
        Plaintext tgt_date(intToHex(toDays(Date{1993,7,1}), plaintext_modulus));
        ctxt_gte_date = GTE(comparator, evaluator, relin_keys, ctxt_o_orderdate, tgt_date);
    }

    #pragma omp task
    {
        //O_ORDERDATE < dateadd(mm,3, cast('1993-07-01' as date))
        Plaintext tgt_date_lt(intToHex(toDays(Date{1993,10,1}), plaintext_modulus));
        ctxt_lt_date = LT(comparator, evaluator, relin_keys, ctxt_o_orderdate, tgt_date_lt);
    }

    #pragma omp task
    {
        //L_COMMITDATE < L_RECEIPTDATE
        ctxt_lt_receiptdate = LT(comparator, evaluator, relin_keys, 
                                 ctxt_l_commitdate, ctxt_l_receiptdate);
    }

        #pragma omp taskwait
        // Combine all filters related to ORDER TABLE including GROUP BY
        evaluator.multiply(ctxt_gte_date, ctxt_lt_date, tmp_filter);
        evaluator.relinearize_inplace(tmp_filter, relin_keys);

        #pragma omp taskloop nogroup
        for(size_t i = 0; i < ctxt_grp_res.size(); i++) {
            evaluator.multiply_inplace(ctxt_grp_res[i], tmp_filter);
            evaluator.relinearize_inplace(ctxt_grp_res[i], relin_keys);
        }
        #pragma omp taskwait
        
        if(debug == 2) {
            verify_filtered_ordertable(o_orderpriority, o_orderdate,
                                      ctxt_grp_res, batch_encoder, decryptor);
        }

        // SELECT O_ORDERPRIORITY, COUNT(*)
        #pragma omp taskloop nogroup
        for(int p = 0; p < NUM_PRIORITY_GROUP; p++) {
            //L_ORDERKEY = O_ORDERKEY
            for (int i = 0; i < numEle_o; i++) {
                Plaintext ptxt_orderkey(intToHex(i+1, plaintext_modulus));
                Ciphertext ctxt_exist, ctxt_key_match, ctxt_ordertable;
                ctxt_key_match = comparator.isEqual(evaluator, relin_keys,
                                                    ctxt_l_orderkey, ptxt_orderkey);
                evaluator.multiply(ctxt_lt_receiptdate, ctxt_key_match, ctxt_exist);
                evaluator.relinearize_inplace(ctxt_exist, relin_keys);
                if(debug == 2)
                {
                    verify_exist(l_commitdate, l_receiptdate, l_orderkey,
                                i+1, ctxt_exist, batch_encoder, decryptor);
                }

                vector<int64_t> mask(numEle_o, 0);
                Plaintext ptxt_mask;
                mask[i] = 1;
                batch_encoder.encode(mask, ptxt_mask);
                evaluator.multiply_plain(ctxt_grp_res[p], ptxt_mask, ctxt_ordertable);
                ctxt_ordertable = SUM(evaluator, ctxt_ordertable, //BATCH -> SINGLE
                                    slot_count, galois_keys);
                evaluator.multiply_inplace(ctxt_ordertable, ctxt_exist);
                evaluator.relinearize_inplace(ctxt_ordertable, relin_keys);

                if (i == 0) ctxt_order_count[p] = ctxt_ordertable;
                else evaluator.add_inplace(ctxt_order_count[p], ctxt_ordertable);

            }
            ctxt_order_count[p] = SUM(evaluator, ctxt_order_count[p], slot_count, 
                                      galois_keys);

        }
        #pragma taskwait
}
        time_tpch.pause();
        if(debug != 0) {
            verify(o_orderpriority, o_orderdate, l_orderkey, o_orderkey,
                   l_commitdate, l_receiptdate, ctxt_order_count, 
                   batch_encoder, decryptor);
        }
        
    }
    // time_select.stop(runs);
    // time_where.stop(runs);
    // time_join.stop(runs);
    // time_groupby.stop(runs);
    time_tpch.stop(runs);
    if(debug != 0)
        cout << "Verified. Done.\n";
    else
        cout << "Verify skipped. Done" << endl;
    return 0;
}