#include "nshedb/nshedb.h"
#include "nshedb/utils/seal_examples.h"
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

// ==============================================================================
// Constants
// ==============================================================================

#define NUM_NATION 25
#define NUM_TYPE 150
#define NUM_REGION 5
#define BRAZIL 20
#define AMERICA 5
#define ECONOMY_ANODIZED_STEEL 25

// SELECT O_YEAR, SUM(CASE WHEN NATION = 'BRAZIL' THEN VOLUME ELSE 0 END)/SUM(VOLUME) AS MKT_SHARE
// FROM (SELECT datepart(yy,O_ORDERDATE) AS O_YEAR, L_EXTENDEDPRICE*(1-L_DISCOUNT) AS VOLUME, N2.N_NAME AS NATION
//  FROM PART, SUPPLIER, LINEITEM, ORDERS, CUSTOMER, NATION N1, NATION N2, REGION
//  WHERE P_PARTKEY = L_PARTKEY AND S_SUPPKEY = L_SUPPKEY AND L_ORDERKEY = O_ORDERKEY
//  AND O_CUSTKEY = C_CUSTKEY AND C_NATIONKEY = N1.N_NATIONKEY AND
//  N1.N_REGIONKEY = R_REGIONKEY AND R_NAME = 'AMERICA' AND S_NATIONKEY = N2.N_NATIONKEY
//  AND O_ORDERDATE BETWEEN '1995-01-01' AND '1996-12-31' AND P_TYPE= 'ECONOMY ANODIZED STEEL') AS ALL_NATIONS
// GROUP BY O_YEAR
// ORDER BY O_YEAR

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
// Verification Logic (TPC-H 8 Specific)
// ==============================================================================

void verify(vector<int64_t> p_type, vector<int64_t> l_partkey, 
            vector<int64_t> o_orderdate, vector<int64_t> l_orderkey,
            vector<int64_t> r_name, vector<int64_t> n_regionkey, 
            vector<int64_t> c_nationkey, vector<int64_t> o_custkey, 
            vector<int64_t> n_name, vector<int64_t> s_nationkey,
            vector<int64_t> l_suppkey, vector<int64_t> l_charge,
            Ciphertext ctxt_mkt_share96, 
            Ciphertext ctxt_mkt_share95,
            int numEle_l,
            BatchEncoder& batch_encoder, Decryptor &decryptor) {

    vector<int64_t> res95(2, 0), res96(2, 0);
    for(int i = 0; i < numEle_l; i++) {
        bool cond1 = p_type[l_partkey[i]-1] == ECONOMY_ANODIZED_STEEL;
        bool cond2 = o_orderdate[l_orderkey[i]-1] >= toDays(Date{1995, 01, 01});
        bool cond3 = o_orderdate[l_orderkey[i]-1] <= toDays(Date{1996, 12, 31});
        bool cond4 = r_name[n_regionkey[c_nationkey[o_custkey[l_orderkey[i]-1]-1]-1]-1] == AMERICA;
        if(cond1 & cond2 & cond4 & cond3) {
            if(o_orderdate[l_orderkey[i]-1] <= toDays(Date{1995, 12, 31})) {
                if(n_name[s_nationkey[l_suppkey[i]-1]-1] == BRAZIL) {
                    res95[0] += l_charge[i];
                }
                res95[1] += l_charge[i];
            }
            else {
                if(n_name[s_nationkey[l_suppkey[i]-1]-1] == BRAZIL) {
                    res96[0] += l_charge[i];
                }
                res96[1] += l_charge[i];
            }
        }
    }

    vector<int64_t> decrypted;
    decrypted = print_dec_<int64_t>("ctxt_mkt_share96", 
                                    ctxt_mkt_share96, numEle_l, 
                                    batch_encoder, decryptor);
    if(res96[0] != decrypted[0]) {
        printf("ctxt_mkt_share96 res Expected: %ld Result: %ld\n", 
                 res96[0], decrypted[0]);
        exit(1);
    }

    decrypted = print_dec_<int64_t>("ctxt_mkt_share95", 
                                    ctxt_mkt_share95, numEle_l, 
                                    batch_encoder, decryptor);
    if(res95[0] != decrypted[0]) {
        printf("ctxt_mkt_share95 res Expected: %ld Result: %ld\n", 
                 res95[0], decrypted[0]);
        exit(1);
    }
}

// ==============================================================================
// Main App
// ==============================================================================

int main(int argc, char *argv[])
{
    if (argc != 8)
    {
        printf("./tpch8 < # runs > < debug mode > <# LINEITEM> < # PART > \
         < # SUPPLIER> < # ORDERS> < # CUSTOMER>\n"
        "[Debug Mode]\n0 - no debug\n"
        "1 - verify once at the end\n"
        "2 - verify each operation\n");
        exit(1);
    }

    int runs = stoi(argv[1]);
    int debug = stoi(argv[2]);
    int numEle_l = stoi(argv[3]);
    int numEle_p = stoi(argv[4]);
    int numEle_s = stoi(argv[5]);
    int numEle_o = stoi(argv[6]);
    int numEle_c = stoi(argv[7]);
    Timer time_parm("[PARAM SETUP] ");
    Timer time_keygen("[KEY GEN] ");
    Timer time_tpch("[TPCH-8] ");
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
    cout << "Total elements: " << numEle_l << endl;

   for(int k = 0; k < runs; k++)
    {

        /*
         Prepare Data...
        */
        vector<int64_t> l_extendedprice, l_suppkey, l_partkey, l_discount,
                        l_orderkey, l_charge;
        vector<int64_t> n_nationkey, n_name, n_regionkey;
        vector<int64_t> o_orderkey, o_custkey, o_orderdate;
        vector<int64_t> c_custkey, c_nationkey;
        vector<int64_t> s_suppkey, s_nationkey;
        vector<int64_t> p_partkey, p_type;
        vector<int64_t> r_regionkey, r_name;

        srand (time(NULL));
        for (int i = 0; i < numEle_l; i++) {
            l_extendedprice.push_back(rand()%201);
            l_partkey.push_back((rand()%numEle_p)+1);
            l_discount.push_back(rand()%11);
            l_suppkey.push_back((rand()%numEle_s)+1);
            l_orderkey.push_back((rand()%numEle_o)+1);
            l_charge.push_back(l_extendedprice[i]*(100-l_discount[i])/100);
        }
        for (int i = 0; i < NUM_NATION; i++) {
            n_nationkey.push_back(i+1);
            n_name.push_back(i+1);
            n_regionkey.push_back((rand()%NUM_REGION)+1);
        }
        for (int i = 0; i < numEle_o; i++) {
            o_orderkey.push_back(i+1);
            o_orderdate.push_back((-1)*rand()%3000);
            o_custkey.push_back((rand()%numEle_c)+1);
        }
        for (int i = 0; i < numEle_c; i++) {
            c_custkey.push_back(i+1);
            c_nationkey.push_back((rand()%NUM_NATION)+1);
        }
        for (int i = 0; i < numEle_s; i++) {
            s_suppkey.push_back(i+1);
            s_nationkey.push_back((rand()%NUM_NATION)+1);
        }
        for (int i = 0; i < numEle_p; i++) {
            p_partkey.push_back(i+1);
            p_type.push_back((rand()%NUM_TYPE)+1);
        }
        for (int i = 0; i < NUM_REGION; i++) {
            r_regionkey.push_back(i+1);
            r_name.push_back(i+1);
        }


        Plaintext ptxt_l_suppkey, ptxt_l_partkey, ptxt_l_orderkey, ptxt_l_charge;
        Plaintext ptxt_n_nationkey, ptxt_n_name, ptxt_n_regionkey;
        Plaintext ptxt_o_orderkey, ptxt_o_custkey, ptxt_o_orderdate, ptxt_o_year;
        Plaintext ptxt_c_custkey, ptxt_c_nationkey;
        Plaintext ptxt_s_suppkey, ptxt_s_nationkey;
        Plaintext ptxt_p_partkey, ptxt_p_type;
        Plaintext ptxt_r_regionkey, ptxt_r_name;
                  
        batch_encoder.encode(l_suppkey, ptxt_l_suppkey);
        batch_encoder.encode(l_partkey, ptxt_l_partkey);
        batch_encoder.encode(l_orderkey, ptxt_l_orderkey);
        batch_encoder.encode(l_charge, ptxt_l_charge);

        batch_encoder.encode(n_nationkey, ptxt_n_nationkey);
        batch_encoder.encode(n_name, ptxt_n_name);
        batch_encoder.encode(n_regionkey, ptxt_n_regionkey);

        batch_encoder.encode(o_orderkey, ptxt_o_orderkey);
        batch_encoder.encode(o_custkey, ptxt_o_custkey);
        batch_encoder.encode(o_orderdate, ptxt_o_orderdate);

        batch_encoder.encode(c_custkey, ptxt_c_custkey);
        batch_encoder.encode(c_nationkey, ptxt_c_nationkey);

        batch_encoder.encode(s_suppkey, ptxt_s_suppkey);
        batch_encoder.encode(s_nationkey, ptxt_s_nationkey);

        batch_encoder.encode(p_partkey, ptxt_p_partkey);
        batch_encoder.encode(p_type, ptxt_p_type);

        batch_encoder.encode(r_name, ptxt_r_name);
        batch_encoder.encode(r_regionkey, ptxt_r_regionkey);

        Ciphertext ctxt_l_suppkey, ctxt_l_partkey,
                  ctxt_l_orderkey, ctxt_l_charge,
                  ctxt_n_nationkey, ctxt_n_name,
                  ctxt_n_regionkey, ctxt_o_orderkey,
                  ctxt_o_custkey, ctxt_o_orderdate,
                  ctxt_c_custkey, ctxt_c_nationkey,
                  ctxt_s_suppkey, ctxt_s_nationkey,
                  ctxt_p_partkey, ctxt_p_type,
                  ctxt_r_name, ctxt_r_regionkey;

        encryptor.encrypt(ptxt_l_suppkey, ctxt_l_suppkey);
        encryptor.encrypt(ptxt_l_partkey, ctxt_l_partkey);
        encryptor.encrypt(ptxt_l_orderkey, ctxt_l_orderkey);
        encryptor.encrypt(ptxt_l_charge, ctxt_l_charge);

        encryptor.encrypt(ptxt_n_nationkey, ctxt_n_nationkey);
        encryptor.encrypt(ptxt_n_name, ctxt_n_name);
        encryptor.encrypt(ptxt_n_regionkey, ctxt_n_regionkey);

        encryptor.encrypt(ptxt_o_orderkey, ctxt_o_orderkey);
        encryptor.encrypt(ptxt_o_custkey, ctxt_o_custkey);
        encryptor.encrypt(ptxt_o_orderdate, ctxt_o_orderdate);

        encryptor.encrypt(ptxt_c_custkey, ctxt_c_custkey);
        encryptor.encrypt(ptxt_c_nationkey, ctxt_c_nationkey);

        encryptor.encrypt(ptxt_s_suppkey, ctxt_s_suppkey);
        encryptor.encrypt(ptxt_s_nationkey, ctxt_s_nationkey);

        encryptor.encrypt(ptxt_p_partkey, ctxt_p_partkey);
        encryptor.encrypt(ptxt_p_type, ctxt_p_type);

        encryptor.encrypt(ptxt_r_name, ctxt_r_name);
        encryptor.encrypt(ptxt_r_regionkey, ctxt_r_regionkey);


        /*
         Query Starts...
        */
        Plaintext tgt_type(intToHex(ECONOMY_ANODIZED_STEEL, plaintext_modulus));
        Plaintext tgt_date1(intToHex(toDays(Date{1995,1,1}), plaintext_modulus));
        Plaintext tgt_date2(intToHex(toDays(Date{1996,12,31}), plaintext_modulus));
        Plaintext tgt_date3(intToHex(toDays(Date{1996,1,1}), plaintext_modulus));
        Plaintext tgt_r_name(intToHex(AMERICA, plaintext_modulus));
        Plaintext tgt_n_nation(intToHex(BRAZIL, plaintext_modulus));

        time_tpch.start();
        Ciphertext ctxt_joined_regionkey, ctxt_joined_nation,
                    ctxt_joined_nation_supply, ctxt_joined_supkey,
                    ctxt_tgt_95, ctxt_tgt_96, ctxt_tgt_type,
                    ctxt_tgt_r_name, ctxt_tgt_n_nation;
        Ciphertext ctxt_total_vol95, ctxt_total_vol96;
        Ciphertext ctxt_mkt_share95, ctxt_mkt_share96;
        vector<Ciphertext> ctxt_joined_custkey(2), ctxt_joined_orderkey(2);

        // P_TYPE= 'ECONOMY ANODIZED STEEL'
        ctxt_tgt_type = comparator.isEqual(evaluator, relin_keys, 
                                            ctxt_p_type, tgt_type);

        // p_partkey = l_partkey
        vector<int64_t> mask(numEle_p, 0);
        vector<Ciphertext> ctxt_joined_partkey(numEle_p);
        for(int i = 0; i < numEle_p; i++) {
            Plaintext ptxt_mask;
            Ciphertext ctxt_masked_partkey, ctxt_masked_filter;
            mask[i] = 1;
            batch_encoder.encode(mask, ptxt_mask);
            // BATCH -> SINGLE
            evaluator.multiply_plain(ctxt_p_partkey, ptxt_mask, ctxt_masked_partkey);
            ctxt_masked_partkey = SUM(evaluator, ctxt_masked_partkey, slot_count, galois_keys);
            ctxt_joined_partkey[i] = comparator.isEqual(evaluator, relin_keys,
                                                   ctxt_masked_partkey, ctxt_l_partkey);
            
            // BATCH -> SINGLE
            evaluator.multiply_plain(ctxt_tgt_type, ptxt_mask, ctxt_masked_filter);
            ctxt_masked_filter = SUM(evaluator, ctxt_masked_filter, slot_count, galois_keys);
            evaluator.multiply_inplace(ctxt_joined_partkey[i], ctxt_masked_filter);
            evaluator.relinearize_inplace(ctxt_joined_partkey[i], relin_keys);
            mask[i] = 0;
        }
        for(size_t i = 1; i < ctxt_joined_partkey.size(); i++) 
            evaluator.add_inplace(ctxt_joined_partkey[0], ctxt_joined_partkey[i]);


        // O_ORDERDATE BETWEEN '1995-01-01' AND '1996-12-31' 
        {   
            Plaintext ptxt_one("1");
            Ciphertext tmp1 = GTE(comparator, evaluator, relin_keys,
                                ctxt_o_orderdate, tgt_date1);
            Ciphertext tmp3 = GTE(comparator, evaluator, relin_keys,
                                ctxt_o_orderdate, tgt_date3);
            Ciphertext tmp2 = LTE(comparator, evaluator, relin_keys,
                                ctxt_o_orderdate, tgt_date2);
            evaluator.multiply(tmp3, tmp2, ctxt_tgt_96);
            evaluator.relinearize_inplace(ctxt_tgt_96, relin_keys);
            evaluator.negate_inplace(tmp3);
            evaluator.add_plain_inplace(tmp3, ptxt_one);
            evaluator.multiply(tmp3, tmp1, ctxt_tgt_95);
            evaluator.relinearize_inplace(ctxt_tgt_95, relin_keys);
        }
        

        // R_NAME = 'AMERICA'
        ctxt_tgt_r_name = comparator.isEqual(evaluator, relin_keys, 
                                          ctxt_r_name, tgt_r_name);

        // NATION = 'BRAZIL'
        ctxt_tgt_n_nation = comparator.isEqual(evaluator, relin_keys, 
                                          ctxt_n_name, tgt_n_nation);

        {
            vector<int64_t> mask(NUM_REGION, 0);
            vector<Ciphertext> ctxt_joined_regionkey_(NUM_REGION);
            for(int i = 0; i < NUM_REGION; i++) {
                Plaintext ptxt_mask;
                Ciphertext ctxt_masked_region, ctxt_masked_filter;
                mask[i] = 1;
                batch_encoder.encode(mask, ptxt_mask);
                // BATCH -> SINGLE
                evaluator.multiply_plain(ctxt_r_regionkey, ptxt_mask, ctxt_masked_region);
                ctxt_masked_region = SUM(evaluator, ctxt_masked_region, slot_count, galois_keys);
                ctxt_joined_regionkey_[i] = comparator.isEqual(evaluator, relin_keys,
                                                    ctxt_masked_region, ctxt_n_regionkey);
                
                // BATCH -> SINGLE
                evaluator.multiply_plain(ctxt_tgt_r_name, ptxt_mask, ctxt_masked_filter);
                ctxt_masked_filter = SUM(evaluator, ctxt_masked_filter, slot_count, galois_keys);
                evaluator.multiply_inplace(ctxt_joined_regionkey_[i], ctxt_masked_filter);
                evaluator.relinearize_inplace(ctxt_joined_regionkey_[i], relin_keys);
                mask[i] = 0;
            }
            for(size_t i = 1; i < ctxt_joined_regionkey_.size(); i++) 
                evaluator.add_inplace(ctxt_joined_regionkey_[0], ctxt_joined_regionkey_[i]);
            ctxt_joined_regionkey = ctxt_joined_regionkey_[0];
        }

        {
            vector<int64_t> mask(NUM_NATION, 0);
            vector<Ciphertext> ctxt_joined_nation_(NUM_NATION);
            vector<Ciphertext> ctxt_joined_nation_supply_(NUM_NATION);
            for(int i = 0; i < NUM_NATION; i++) {
                Plaintext ptxt_mask;
                Ciphertext ctxt_masked_nation, ctxt_masked_filter, ctxt_masked_filter2;
                mask[i] = 1;
                batch_encoder.encode(mask, ptxt_mask);
                // BATCH -> SINGLE
                evaluator.multiply_plain(ctxt_n_nationkey, ptxt_mask, ctxt_masked_nation);
                ctxt_masked_nation = SUM(evaluator, ctxt_masked_nation, slot_count, galois_keys);
                ctxt_joined_nation_[i] = comparator.isEqual(evaluator, relin_keys,
                                                    ctxt_masked_nation, ctxt_c_nationkey);
                ctxt_joined_nation_supply_[i] = comparator.isEqual(evaluator, relin_keys,
                                                    ctxt_masked_nation, ctxt_s_nationkey);
                
                // BATCH -> SINGLE
                evaluator.multiply_plain(ctxt_joined_regionkey, ptxt_mask, ctxt_masked_filter);
                ctxt_masked_filter = SUM(evaluator, ctxt_masked_filter, slot_count, galois_keys);
                evaluator.multiply_inplace(ctxt_joined_nation_[i], ctxt_masked_filter);
                evaluator.relinearize_inplace(ctxt_joined_nation_[i], relin_keys);

                evaluator.multiply_plain(ctxt_tgt_n_nation, ptxt_mask, ctxt_masked_filter2);
                ctxt_masked_filter2 = SUM(evaluator, ctxt_masked_filter2, slot_count, galois_keys);
                evaluator.multiply_inplace(ctxt_joined_nation_supply_[i], ctxt_masked_filter2);
                evaluator.relinearize_inplace(ctxt_joined_nation_supply_[i], relin_keys);

                mask[i] = 0;
            }
            for(int i = 1; i < NUM_NATION; i++) {
                evaluator.add_inplace(ctxt_joined_nation_[0], ctxt_joined_nation_[i]);
                evaluator.add_inplace(ctxt_joined_nation_supply_[0], ctxt_joined_nation_supply_[i]);
            }
            ctxt_joined_nation = ctxt_joined_nation_[0];
            ctxt_joined_nation_supply = ctxt_joined_nation_supply_[0];
        }


        {
            vector<int64_t> mask(numEle_c, 0);
            vector<Ciphertext> ctxt_joined_custkey_(numEle_c);
            for(int i = 0; i < numEle_c; i++) {
                Plaintext ptxt_mask;
                Ciphertext ctxt_masked_custkey, ctxt_masked_filter;
                mask[i] = 1;
                batch_encoder.encode(mask, ptxt_mask);
                // BATCH -> SINGLE
                evaluator.multiply_plain(ctxt_c_custkey, ptxt_mask, ctxt_masked_custkey);
                ctxt_masked_custkey = SUM(evaluator, ctxt_masked_custkey, slot_count, galois_keys);
                ctxt_joined_custkey_[i] = comparator.isEqual(evaluator, relin_keys,
                                                    ctxt_masked_custkey, ctxt_o_custkey);
                
                // BATCH -> SINGLE
                evaluator.multiply_plain(ctxt_joined_nation, ptxt_mask, ctxt_masked_filter);
                ctxt_masked_filter = SUM(evaluator, ctxt_masked_filter, slot_count, galois_keys);
                evaluator.multiply_inplace(ctxt_joined_custkey_[i], ctxt_masked_filter);
                evaluator.relinearize_inplace(ctxt_joined_custkey_[i], relin_keys);
                mask[i] = 0;
            }
            for(size_t i = 1; i < ctxt_joined_custkey_.size(); i++) 
                evaluator.add_inplace(ctxt_joined_custkey_[0], ctxt_joined_custkey_[i]);
            evaluator.multiply(ctxt_joined_custkey_[0], ctxt_tgt_95, ctxt_joined_custkey[0]);
            evaluator.relinearize_inplace(ctxt_joined_custkey[0], relin_keys);
            evaluator.multiply(ctxt_joined_custkey_[1], ctxt_tgt_96, ctxt_joined_custkey[1]);
            evaluator.relinearize_inplace(ctxt_joined_custkey[1], relin_keys);
        }


        {
            vector<int64_t> mask(numEle_o, 0);
            vector<vector<Ciphertext>> ctxt_joined_orderkey_(numEle_o);
            
            for(int i = 0; i < numEle_o; i++) {
                Plaintext ptxt_mask;
                Ciphertext ctxt_masked_orderkey, ctxt_masked_filter;
                mask[i] = 1;
                batch_encoder.encode(mask, ptxt_mask);
                ctxt_joined_orderkey_[i].resize(2);
                // BATCH -> SINGLE
                evaluator.multiply_plain(ctxt_o_orderkey, ptxt_mask, ctxt_masked_orderkey);
                ctxt_masked_orderkey = SUM(evaluator, ctxt_masked_orderkey, slot_count, galois_keys);
                Ciphertext ctxt_joined = comparator.isEqual(evaluator, relin_keys,
                                                    ctxt_masked_orderkey, ctxt_l_orderkey);
                
                // BATCH -> SINGLE
                evaluator.multiply_plain(ctxt_joined_custkey[0], ptxt_mask, ctxt_masked_filter);
                ctxt_masked_filter = SUM(evaluator, ctxt_masked_filter, slot_count, galois_keys);
                evaluator.multiply(ctxt_joined, ctxt_masked_filter, ctxt_joined_orderkey_[i][0]);
                evaluator.relinearize_inplace(ctxt_joined_orderkey_[i][0], relin_keys);

                evaluator.multiply_plain(ctxt_joined_custkey[1], ptxt_mask, ctxt_masked_filter);
                ctxt_masked_filter = SUM(evaluator, ctxt_masked_filter, slot_count, galois_keys);
                evaluator.multiply(ctxt_joined, ctxt_masked_filter, ctxt_joined_orderkey_[i][1]);
                evaluator.relinearize_inplace(ctxt_joined_orderkey_[i][1], relin_keys);
                mask[i] = 0;
            }
            for(size_t i = 1; i < ctxt_joined_orderkey_.size(); i++) {
                evaluator.add_inplace(ctxt_joined_orderkey_[0][0], ctxt_joined_orderkey_[i][0]);
                evaluator.add_inplace(ctxt_joined_orderkey_[0][1], ctxt_joined_orderkey_[i][1]);
            }
            ctxt_joined_orderkey[0] = ctxt_joined_orderkey_[0][0];
            ctxt_joined_orderkey[1] = ctxt_joined_orderkey_[0][1];
        }


        {
            vector<int64_t> mask(numEle_s, 0);
            vector<Ciphertext> ctxt_joined_supkey_(numEle_s);
            for(int i = 0; i < numEle_s; i++) {
                Plaintext ptxt_mask;
                Ciphertext ctxt_masked_supkey, ctxt_masked_filter;
                mask[i] = 1;
                batch_encoder.encode(mask, ptxt_mask);
                // BATCH -> SINGLE
                evaluator.multiply_plain(ctxt_s_suppkey, ptxt_mask, ctxt_masked_supkey);
                ctxt_masked_supkey = SUM(evaluator, ctxt_masked_supkey, slot_count, galois_keys);
                ctxt_joined_supkey_[i] = comparator.isEqual(evaluator, relin_keys,
                                                    ctxt_masked_supkey, ctxt_l_orderkey);
                
                // BATCH -> SINGLE
                evaluator.multiply_plain(ctxt_joined_nation_supply, ptxt_mask, ctxt_masked_filter);
                ctxt_masked_filter = SUM(evaluator, ctxt_masked_filter, slot_count, galois_keys);
                evaluator.multiply_inplace(ctxt_joined_supkey_[i], ctxt_masked_filter);
                evaluator.relinearize_inplace(ctxt_joined_supkey_[i], relin_keys);
                mask[i] = 0;
            }
            for(size_t i = 1; i < ctxt_joined_supkey_.size(); i++) 
                evaluator.add_inplace(ctxt_joined_supkey_[0], ctxt_joined_supkey_[i]);
            ctxt_joined_supkey = ctxt_joined_supkey_[0];
        }

        evaluator.multiply(ctxt_joined_orderkey[0], ctxt_l_charge, ctxt_total_vol95);
        evaluator.relinearize_inplace(ctxt_total_vol95, relin_keys);
        evaluator.multiply(ctxt_joined_orderkey[1], ctxt_l_charge, ctxt_total_vol96);      
        evaluator.relinearize_inplace(ctxt_total_vol96, relin_keys);


        Ciphertext tmp;
        evaluator.multiply(ctxt_joined_supkey, ctxt_l_charge, tmp);
        evaluator.relinearize_inplace(tmp, relin_keys);
        evaluator.multiply(tmp, ctxt_joined_orderkey[0], ctxt_mkt_share95);
        evaluator.relinearize_inplace(ctxt_mkt_share95, relin_keys);
        evaluator.multiply(tmp, ctxt_joined_orderkey[1], ctxt_mkt_share96);        
        evaluator.relinearize_inplace(ctxt_mkt_share96, relin_keys);  

        ctxt_mkt_share95 = SUM(evaluator, ctxt_mkt_share95, slot_count, galois_keys);
        ctxt_mkt_share96 = SUM(evaluator, ctxt_mkt_share96, slot_count, galois_keys);
        ctxt_total_vol95 = SUM(evaluator, ctxt_total_vol95, slot_count, galois_keys);
        ctxt_total_vol96 = SUM(evaluator, ctxt_total_vol96, slot_count, galois_keys);
        auto context_data = context.first_context_data();
        while (context_data->next_context_data())
        {
            evaluator.mod_switch_to_next_inplace(ctxt_mkt_share95);
            evaluator.mod_switch_to_next_inplace(ctxt_mkt_share96);
            evaluator.mod_switch_to_next_inplace(ctxt_total_vol95);
            evaluator.mod_switch_to_next_inplace(ctxt_total_vol96);
            context_data = context_data->next_context_data();
        }
        time_tpch.pause();
        cout << "    + noise budget left: " << decryptor.invariant_noise_budget(ctxt_mkt_share95) << " bits"
         << endl;
        cout << "    + noise budget left: " << decryptor.invariant_noise_budget(ctxt_joined_supkey) << " bits"
         << endl;
        cout << "    + noise budget left: " << decryptor.invariant_noise_budget(ctxt_total_vol95) << " bits"
         << endl;
        cout << "    + noise budget left: " << decryptor.invariant_noise_budget(ctxt_joined_orderkey[0]) << " bits"
         << endl;
        if(debug != 0) {
            verify(p_type, l_partkey, o_orderdate, l_orderkey, r_name, n_regionkey, 
                    c_nationkey, o_custkey, n_name, s_nationkey, l_suppkey, l_charge,
                    ctxt_mkt_share96, ctxt_mkt_share95, numEle_l, batch_encoder, decryptor);
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