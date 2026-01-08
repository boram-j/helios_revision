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

#define NUM_CONTAINER 40
#define NUM_BRAND 25
#define NUM_SHIPMODE 7
#define NUM_SHIPINSTRUCT 4
#define NUM_SIZE 50

#define AIR 3
#define AIR_REG 5
#define DELIVER_IN_PERSON 1

#define BRAND12 12
#define BRAND23 23
#define BRAND34 34

#define SM_CASE 1
#define SM_BOX 2
#define SM_PACK 3
#define SM_PKG 4

#define MED_BAG 15
#define MED_BOX 12
#define MED_PACK 13
#define MED_PKG 14

#define LG_CASE 21
#define LG_BOX 22
#define LG_PACK 23
#define LG_PKG 24

#define QUANTITY_1 1
#define QUANTITY_11 11
#define QUANTITY_10 10
#define QUANTITY_20 20
#define QUANTITY_30 30

#define SIZE_1 1
#define SIZE_5 5
#define SIZE_10 10
#define SIZE_15 15

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
// Verification Logic (TPC-H 19 Specific)
// ==============================================================================

void verify(vector<int64_t> l_quantity, vector<int64_t> l_shipmode,
            vector<int64_t> l_shipinstruct, vector<int64_t> l_charge,
            vector<int64_t> p_partkey, vector<int64_t> l_partkey,
            vector<int64_t> p_brand, vector<int64_t> p_size,
            vector<int64_t> p_container,
            int64_t res)
{
    int64_t sol = 0;
    for(size_t i = 0; i < l_quantity.size(); i++) {
        bool cond1 = l_shipmode[i] == AIR || l_shipmode[i] == AIR_REG;
        bool cond2 = l_shipinstruct[i] == DELIVER_IN_PERSON;

        int idx = l_partkey[i]-1;
        bool cond3 = l_quantity[i]>=1 && l_quantity[i]<=10 
                    && p_brand[idx] == BRAND12 
                    && (p_container[idx] == SM_CASE || 
                        p_container[idx] == SM_BOX ||
                        p_container[idx] == SM_PACK ||
                        p_container[idx] == SM_PKG);
        bool cond4 = l_quantity[i]>=10 && l_quantity[i]<=20
                    && p_brand[idx] == BRAND23
                    && (p_container[idx] == MED_BAG || 
                        p_container[idx] == MED_BOX ||
                        p_container[idx] == MED_PACK ||
                        p_container[idx] == MED_PKG);
        bool cond5 = l_quantity[i]>=20 && l_quantity[i]<=30
                    && p_brand[idx] == BRAND34
                    && (p_container[idx] == LG_CASE || 
                        p_container[idx] == LG_BOX ||
                        p_container[idx] == LG_PACK ||
                        p_container[idx] == LG_PKG);
        if((cond1&cond2&cond3) || (cond1&cond2&cond4) || (cond1&cond2&cond5))
            sol += l_charge[i];
    }

    if (sol != res) {
        printf("SUM Expected: %ld Result: %ld\n", sol, res);
        exit(1);
    }
}


void verify_shipmode_in(Decryptor &decryptor, BatchEncoder &batch_encoder, int numEle_l, 
                        Ciphertext ctxt_shipmode_in, vector<int64_t> l_shipmode)
{
    vector<int64_t> decrypted;
    decrypted = print_dec_<int64_t>("l_shipmode in ('AIR', 'AIR REG')", 
                                    ctxt_shipmode_in, numEle_l, batch_encoder, decryptor);
    for(int i = 0; i < numEle_l; i++) {
        int64_t res = l_shipmode[i] == AIR || l_shipmode[i] == AIR_REG;
        if (res != decrypted[i]){
            printf("res[%d] Expected: %ld Result: %ld\n", i, res, decrypted[i]);
            exit(1);
        }
    }
    cout << "shipmode verified" << endl;
}


void verify_shipinstruct(Decryptor &decryptor, BatchEncoder &batch_encoder, int numEle_l, 
                        Ciphertext ctxt_shipinst_inperson, vector<int64_t> l_shipinstruct)
{
    vector<int64_t> decrypted;
    decrypted = print_dec_<int64_t>("l_shipinstruct = 'DELIVER IN PERSON'", 
                                    ctxt_shipinst_inperson, numEle_l, batch_encoder, decryptor);
    for(int i = 0; i < numEle_l; i++) {
        int64_t res = l_shipinstruct[i] == DELIVER_IN_PERSON;
        if (res != decrypted[i]){
            printf("res[%d] Expected: %ld Result: %ld\n", i, res, decrypted[i]);
            exit(1);
        }
    }
}

void verify_filter(Decryptor &decryptor, BatchEncoder &batch_encoder, int numEle_l, 
                    Ciphertext ctxt_filtered, vector<int64_t> l_quantity, 
                    vector<int64_t> l_shipmode, vector<int64_t> l_shipinstruct)
{
    vector<int64_t> decrypted;
    decrypted = print_dec_<int64_t>("FINAL FILTER'", 
                                    ctxt_filtered, numEle_l, batch_encoder, decryptor);
    for(int i = 0; i < numEle_l; i++) {
        bool cond = ((l_quantity[i] >= 1 && l_quantity[i] <= 1+10) &&
                    (l_shipmode[i] == AIR | l_shipmode[i] == AIR_REG) &&
                    (l_shipinstruct[i] == DELIVER_IN_PERSON))
                    || ((l_quantity[i] >= 10 && l_quantity[i] <= 20) &&
                    (l_shipmode[i] == AIR | l_shipmode[i] == AIR_REG) &&
                    (l_shipinstruct[i] == DELIVER_IN_PERSON))
                    ||((l_quantity[i] >= 20 && l_quantity[i] <= 20+10) &&
                    (l_shipmode[i] == AIR | l_shipmode[i] == AIR_REG) &&
                    (l_shipinstruct[i] == DELIVER_IN_PERSON));
        if (cond != decrypted[i]){
            printf("res[%d] Expected: %d Result: %ld\n", i, cond, decrypted[i]);
            exit(1);
        }
    }
}

// ==============================================================================
// Main App
// ==============================================================================

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        printf("./tpch19 < # runs > < debug mode > < # numEle_l for Lineitem> <# numEle_p for Part> \n"
            "[Debug Mode]\n0 - no debug\n"
            "1 - verify once at the end\n"
            "2 - verify each operation\n");
        exit(1);
    }

    int runs = stoi(argv[1]);
    int debug = stoi(argv[2]);
    int numEle_l = stoi(argv[3]);
    int numEle_p = stoi(argv[4]);


    Timer time_parm("[PARAM SETUP] ");
    Timer time_keygen("[KEY GEN] ");
    Timer time_tpch("[TPCH-19] ");
    Timer time_select("[SELECT] ");
    Timer time_where("[WHERE] ");
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
        // input variables
        vector<int64_t> l_quantity, l_shipinstruct, l_partkey,
            l_shipmode, l_extendedprice, l_discount;
        vector<int64_t> l_charge;
        vector<int64_t> p_brand, p_container, p_partkey, p_size;

        Plaintext ptxt_l_quantity, ptxt_l_shipinstruct,
            ptxt_l_shipmode, ptxt_l_extendedprice, 
            ptxt_l_discount, ptxt_l_partkey;
        Plaintext ptxt_l_charge;
        Plaintext ptxt_p_brand, ptxt_p_container, 
                  ptxt_p_partkey, ptxt_p_size;


        Ciphertext ctxt_l_quantity, ctxt_l_shipinstruct,
            ctxt_l_shipmode, ctxt_l_extendedprice, 
            ctxt_l_discount, ctxt_l_partkey;
        Ciphertext ctxt_l_charge;
        Ciphertext ctxt_p_brand, ctxt_p_container, 
                   ctxt_p_partkey, ctxt_p_size;

        // variables to hold the immediate cmp results
        Ciphertext ctxt_shipmode_in, ctxt_shipinst_inperson,
            ctxt_quantity_gte_1, ctxt_quantity_gte_10,
            ctxt_quantity_gte_20, ctxt_quantity_lte_30,
            ctxt_quantity_lte_11, ctxt_quantity_lte_20,
            ctxt_quantity_1_11, ctxt_quantity_10_20,
            ctxt_quantity_20_30,
            ctxt_revenue, ctxt_eq_brand_12,
            ctxt_eq_brand_23, ctxt_eq_brand_34,
            ctxt_container_in_sm, ctxt_container_in_med,
            ctxt_container_in_lg, ctxt_size_gte_1,
            ctxt_size_lte_5, ctxt_size_lte_10,
            ctxt_size_lte_15,
            ctxt_size_1_5, ctxt_size_1_10,
            ctxt_p_12_sm, ctxt_p_23_med, ctxt_p_34_lg,
            ctxt_l_filter_1, ctxt_l_filter_2, ctxt_l_filter_3,
            ctxt_p_filter_1, ctxt_p_filter_2, ctxt_p_filter_3,
            ctxt_joined1, ctxt_joined2, ctxt_joined3,
            ctxt_size_1_15;

        // will hold (p_partkey=l_partkey)*charge
        vector<Ciphertext> ctxt_key_match1(numEle_p),
                           ctxt_key_match2(numEle_p),
                           ctxt_key_match3(numEle_p);

        // variables to hold combining cmp results
        Ciphertext ctxt_cond1, ctxt_cond2, ctxt_cond3;
        Ciphertext ctxt_common_cond, cond1;

        srand (time(NULL));
        for(int i = 0; i < numEle_l; i++) {
            l_partkey.push_back((rand()%numEle_p)+1);
            l_quantity.push_back(rand()%50);
            l_shipinstruct.push_back((rand()%NUM_SHIPINSTRUCT)+1);
            l_shipmode.push_back((rand()%NUM_SHIPMODE)+1);
            l_extendedprice.push_back(rand()%201);
            l_discount.push_back(rand()%11);
            l_charge.push_back(l_extendedprice[i]*(100-l_discount[i])/100);
        }

        for(int i = 0; i < numEle_p; i++) {
            p_brand.push_back((rand()%NUM_BRAND)+1);
            p_container.push_back((rand()%NUM_CONTAINER)+1);
            p_size.push_back((rand()%NUM_SIZE)+1);
            p_partkey.push_back(i+1);
        }
        batch_encoder.encode(l_partkey, ptxt_l_partkey);
        batch_encoder.encode(l_quantity, ptxt_l_quantity);
        batch_encoder.encode(l_shipinstruct, ptxt_l_shipinstruct);
        batch_encoder.encode(l_shipmode, ptxt_l_shipmode);
        batch_encoder.encode(l_extendedprice, ptxt_l_extendedprice);
        batch_encoder.encode(l_discount, ptxt_l_discount);
        batch_encoder.encode(l_charge, ptxt_l_charge);
        batch_encoder.encode(p_brand, ptxt_p_brand);
        batch_encoder.encode(p_size, ptxt_p_size);
        batch_encoder.encode(p_partkey, ptxt_p_partkey);
        batch_encoder.encode(p_container, ptxt_p_container);

        encryptor.encrypt(ptxt_l_partkey, ctxt_l_partkey);
        encryptor.encrypt(ptxt_l_quantity, ctxt_l_quantity);
        encryptor.encrypt(ptxt_l_shipinstruct, ctxt_l_shipinstruct);
        encryptor.encrypt(ptxt_l_shipmode, ctxt_l_shipmode);
        encryptor.encrypt(ptxt_l_extendedprice, ctxt_l_extendedprice);
        encryptor.encrypt(ptxt_l_discount, ctxt_l_discount);
        encryptor.encrypt(ptxt_l_charge, ctxt_l_charge);
        encryptor.encrypt(ptxt_p_container, ctxt_p_container);
        encryptor.encrypt(ptxt_p_partkey, ctxt_p_partkey);
        encryptor.encrypt(ptxt_p_size, ctxt_p_size);
        encryptor.encrypt(ptxt_p_brand, ctxt_p_brand);

        time_tpch.start();
        time_where.start();

        // l_quantity >= 1  
        Plaintext tgt_quantity1(to_string(QUANTITY_1));
        ctxt_quantity_gte_1 = GTE(comparator, evaluator, relin_keys, 
                                    ctxt_l_quantity, tgt_quantity1);
 
        // l_quantity >= 10 
        Plaintext tgt_quantity10(intToHex(QUANTITY_10, plaintext_modulus));
        ctxt_quantity_gte_10 = GTE(comparator, evaluator, relin_keys, 
                                    ctxt_l_quantity, tgt_quantity10);

        // l_quantity >= 20
        Plaintext tgt_quantity20(intToHex(QUANTITY_20, plaintext_modulus));
        ctxt_quantity_gte_20 = GTE(comparator, evaluator, relin_keys, 
                                    ctxt_l_quantity, tgt_quantity20);

        // l_quantity <= 11
        Plaintext tgt_quantity11(intToHex(QUANTITY_11, plaintext_modulus));
        ctxt_quantity_lte_11 = LTE(comparator, evaluator, relin_keys,
                                    ctxt_l_quantity, tgt_quantity11);

        // l_quantity <= 20
        Ciphertext gt20;
        Plaintext one("1");
        ctxt_quantity_lte_20 = comparator.isEqual(evaluator, relin_keys, 
                                                  ctxt_l_quantity, tgt_quantity20);
        evaluator.negate(ctxt_quantity_gte_20, gt20);
        evaluator.add_plain_inplace(gt20, one);
        evaluator.add_inplace(ctxt_quantity_lte_20, gt20);

        ctxt_quantity_lte_20 = LTE(comparator, evaluator, relin_keys,
                                    ctxt_l_quantity, tgt_quantity20);
 
        // l_quantity <= 30
        Plaintext tgt_quantity30(intToHex(QUANTITY_30, plaintext_modulus));
        ctxt_quantity_lte_30 = LTE(comparator, evaluator, relin_keys,
                                    ctxt_l_quantity, tgt_quantity30);

        //p_size between 1 AND 10
        Plaintext tgt_size_lte_10(intToHex(SIZE_10, plaintext_modulus));
        ctxt_size_lte_10 = LTE(comparator, evaluator, relin_keys, ctxt_p_size, tgt_size_lte_10);

        //p_size between 1 AND 15
        Plaintext tgt_size_lte_15(intToHex(SIZE_15, plaintext_modulus));
        ctxt_size_lte_15 = LTE(comparator, evaluator, relin_keys, ctxt_p_size, tgt_size_lte_15);

        //p_size between 1 AND 5
        Plaintext tgt_size_gte_1(intToHex(SIZE_1, plaintext_modulus));
        ctxt_size_gte_1 = GTE(comparator, evaluator, relin_keys, ctxt_p_size, tgt_size_gte_1);

        Plaintext tgt_size_lte_5(intToHex(SIZE_5, plaintext_modulus));
        ctxt_size_lte_5 = LTE(comparator, evaluator, relin_keys, ctxt_p_size, tgt_size_lte_5);

        //p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
        vector<Plaintext> tgt_container = {Plaintext(intToHex(SM_CASE, plaintext_modulus)),
                                           Plaintext(intToHex(SM_BOX, plaintext_modulus)),
                                           Plaintext(intToHex(SM_PACK, plaintext_modulus)),
                                           Plaintext(intToHex(SM_PKG, plaintext_modulus))};
        ctxt_container_in_sm = IN(comparator, evaluator, relin_keys, ctxt_p_container, tgt_container);

        //p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
        vector<Plaintext> tgt_container_med = {Plaintext(intToHex(MED_BAG, plaintext_modulus)),
                                           Plaintext(intToHex(MED_BOX, plaintext_modulus)),
                                           Plaintext(intToHex(MED_PKG, plaintext_modulus)),
                                           Plaintext(intToHex(MED_PACK, plaintext_modulus))};
        ctxt_container_in_med = IN(comparator, evaluator, relin_keys, ctxt_p_container, tgt_container_med);

        //p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
        vector<Plaintext> tgt_container_lg = {Plaintext(intToHex(LG_CASE, plaintext_modulus)),
                                           Plaintext(intToHex(LG_BOX, plaintext_modulus)),
                                           Plaintext(intToHex(LG_PACK, plaintext_modulus)),
                                           Plaintext(intToHex(LG_PKG, plaintext_modulus))};
        ctxt_container_in_lg = IN(comparator, evaluator, relin_keys, ctxt_p_container, tgt_container_lg);

        // l_shipmode in ('AIR', 'AIR REG')
        vector<Plaintext> ptxt_shipmode = {Plaintext(intToHex(AIR, plaintext_modulus)),
                                           Plaintext(intToHex(AIR_REG, plaintext_modulus))};
        ctxt_shipmode_in = IN(comparator, evaluator, relin_keys, ctxt_l_shipmode, ptxt_shipmode);
        if(debug == 2) {
            verify_shipmode_in(decryptor, batch_encoder, numEle_l, 
                        ctxt_shipmode_in, l_shipmode);
        }

        // l_shipinstruct = 'DELIVER IN PERSON'
        Plaintext ptxt0(intToHex(DELIVER_IN_PERSON, plaintext_modulus));
        ctxt_shipinst_inperson = comparator.isEqual(evaluator, relin_keys, 
                                                    ctxt_l_shipinstruct, ptxt0);
        if(debug == 2) {
            verify_shipinstruct(decryptor, batch_encoder, numEle_l, 
                        ctxt_shipinst_inperson, l_shipinstruct);
        }

        //p_brand = 'Brand#12'
        Plaintext tgt_brand12(intToHex(BRAND12, plaintext_modulus));
        ctxt_eq_brand_12 = comparator.isEqual(evaluator, relin_keys, ctxt_p_brand, tgt_brand12);

        //p_brand = 'Brand#23'
        Plaintext tgt_brand23(intToHex(BRAND23, plaintext_modulus));
        ctxt_eq_brand_23 = comparator.isEqual(evaluator, relin_keys, ctxt_p_brand, tgt_brand23);

        // p_brand = 'Brand#34'
        Plaintext tgt_brand34(intToHex(BRAND34, plaintext_modulus));
        ctxt_eq_brand_34 = comparator.isEqual(evaluator, relin_keys, ctxt_p_brand, tgt_brand34);

        //l_quantity >= 1 AND l_quantity <= 1 + 10
        evaluator.multiply(ctxt_quantity_gte_1, ctxt_quantity_lte_11,
                           ctxt_quantity_1_11);
        evaluator.relinearize_inplace(ctxt_quantity_1_11, relin_keys);

        //l_quantity >= 10 AND l_quantity <= 10 + 10
        evaluator.multiply(ctxt_quantity_gte_10, ctxt_quantity_lte_20,
                           ctxt_quantity_10_20);
        evaluator.relinearize_inplace(ctxt_quantity_10_20, relin_keys);

        //AND l_quantity >= 20 AND l_quantity <= 20 + 10
        evaluator.multiply(ctxt_quantity_gte_20, ctxt_quantity_lte_30,
                           ctxt_quantity_20_30);
        evaluator.relinearize_inplace(ctxt_quantity_20_30, relin_keys);

        //p_size between 1 AND 5
        evaluator.multiply(ctxt_size_gte_1, ctxt_size_lte_5, ctxt_size_1_5);
        evaluator.relinearize_inplace(ctxt_size_1_5, relin_keys);

        //p_size between 1 AND 10
        evaluator.multiply(ctxt_size_gte_1, ctxt_size_lte_10, ctxt_size_1_10);
        evaluator.relinearize_inplace(ctxt_size_1_10, relin_keys);

        //p_size between 1 AND 15
        evaluator.multiply(ctxt_size_gte_1, ctxt_size_lte_15, ctxt_size_1_15);
        evaluator.relinearize_inplace(ctxt_size_1_15, relin_keys);

        //l_shipmode in ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON'
        evaluator.multiply(ctxt_shipmode_in, ctxt_shipinst_inperson, ctxt_common_cond);
        evaluator.relinearize_inplace(ctxt_common_cond, relin_keys);

        // p_brand = 'Brand#12'
        // AND p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
        evaluator.multiply(ctxt_eq_brand_12, ctxt_container_in_sm, ctxt_p_12_sm);
        evaluator.relinearize_inplace(ctxt_p_12_sm, relin_keys);

        // p_brand = 'Brand#23'
        // AND p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
        evaluator.multiply(ctxt_eq_brand_23, ctxt_container_in_med, ctxt_p_23_med);
        evaluator.relinearize_inplace(ctxt_p_23_med, relin_keys);

        // p_brand = 'Brand#34'
        // AND p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
        evaluator.multiply(ctxt_eq_brand_34, ctxt_container_in_lg, ctxt_p_34_lg);
        evaluator.relinearize_inplace(ctxt_p_34_lg, relin_keys);

        // l_quantity >= 1 AND l_quantity <= 1 + 10 AND l_shipmode in ('AIR', 'AIR REG')
        // AND l_shipinstruct = 'DELIVER IN PERSON'
        evaluator.multiply(ctxt_quantity_1_11, ctxt_common_cond, ctxt_l_filter_1);
        evaluator.relinearize_inplace(ctxt_l_filter_1, relin_keys);

        // l_quantity >= 10 AND l_quantity <= 10 + 10 AND l_shipmode in ('AIR', 'AIR REG')
        // AND l_shipinstruct = 'DELIVER IN PERSON'
        evaluator.multiply(ctxt_quantity_10_20, ctxt_common_cond, ctxt_l_filter_2);
        evaluator.relinearize_inplace(ctxt_l_filter_2, relin_keys);

        // l_quantity >= 20 AND l_quantity <= 20 + 10 AND l_shipmode in ('AIR', 'AIR REG')
        // AND l_shipinstruct = 'DELIVER IN PERSON'
        evaluator.multiply(ctxt_quantity_20_30, ctxt_common_cond, ctxt_l_filter_3);
        evaluator.relinearize_inplace(ctxt_l_filter_3, relin_keys);

        // p_brand = 'Brand#12'
        // AND p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
        // AND p_size between 1 AND 5
        evaluator.multiply(ctxt_p_12_sm, ctxt_size_1_5, ctxt_p_filter_1);
        evaluator.relinearize_inplace(ctxt_p_filter_1, relin_keys);

        // AND p_brand = 'Brand#23'
        // AND p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
        // AND p_size between 1 AND 10
        evaluator.multiply(ctxt_p_23_med, ctxt_size_1_10, ctxt_p_filter_2);
        evaluator.relinearize_inplace(ctxt_p_filter_2, relin_keys);

        // AND p_brand = 'Brand#34'
        // AND p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
        // AND p_size between 1 AND 15
        evaluator.multiply(ctxt_p_34_lg, ctxt_size_1_15, ctxt_p_filter_3);
        evaluator.relinearize_inplace(ctxt_p_filter_3, relin_keys);

        // (p_partkey == l_partkey )* l_extendedprice * (1 - l_discount)
        for(int i = 0; i < numEle_p; i++) {
            Plaintext ptxt_mask;
            Ciphertext ctxt_partkey; // hold one single partkey
            Ciphertext ctxt_p_filtered;
            vector<int64_t> mask(numEle_p, 0);
            mask[i] = 1;
            batch_encoder.encode(mask, ptxt_mask);
            evaluator.multiply(ctxt_p_partkey, ctxt_p_filter_1, ctxt_p_filtered);
            evaluator.relinearize_inplace(ctxt_p_filtered, relin_keys);
            evaluator.multiply_plain_inplace(ctxt_p_filtered, ptxt_mask);
            // BATCH -> SINGLE
            ctxt_p_filtered = SUM(evaluator, ctxt_p_filtered, slot_count, galois_keys);

            evaluator.multiply_plain(ctxt_p_partkey, ptxt_mask, ctxt_partkey);
            // BATCH -> SINGLE
            ctxt_partkey = SUM(evaluator, ctxt_partkey, slot_count, galois_keys);
            Ciphertext tmp = comparator.isEqual(evaluator, relin_keys, 
                                                ctxt_partkey, ctxt_l_partkey);
            evaluator.multiply_inplace(tmp, ctxt_p_filtered);
            evaluator.relinearize_inplace(tmp, relin_keys);
            evaluator.multiply_inplace(tmp, ctxt_l_charge);
            evaluator.relinearize_inplace(tmp, relin_keys);
            evaluator.multiply_inplace(tmp, ctxt_l_filter_1);
            evaluator.relinearize_inplace(tmp, relin_keys);
            ctxt_key_match1[i] = tmp;
        }


        for(int i = 0; i < numEle_p; i++) {
            Plaintext ptxt_mask;
            Ciphertext ctxt_partkey; // hold one single partkey
            Ciphertext ctxt_p_filtered;
            vector<int64_t> mask(numEle_p, 0);
            mask[i] = 1;
            batch_encoder.encode(mask, ptxt_mask);
            evaluator.multiply(ctxt_p_partkey, ctxt_p_filter_2, ctxt_p_filtered);
            evaluator.multiply_plain_inplace(ctxt_p_filtered, ptxt_mask);
            evaluator.relinearize_inplace(ctxt_p_filtered, relin_keys);
            // BATCH -> SINGLE
            ctxt_p_filtered = SUM(evaluator, ctxt_p_filtered, slot_count, galois_keys);

            evaluator.multiply_plain(ctxt_p_partkey, ptxt_mask, ctxt_partkey);
            // BATCH -> SINGLE
            ctxt_partkey = SUM(evaluator, ctxt_partkey, slot_count, galois_keys);
            Ciphertext tmp = comparator.isEqual(evaluator, relin_keys, 
                                                ctxt_partkey, ctxt_l_partkey);
            evaluator.multiply_inplace(tmp, ctxt_p_filtered);
            evaluator.relinearize_inplace(tmp, relin_keys);
            evaluator.multiply_inplace(tmp, ctxt_l_charge);
            evaluator.relinearize_inplace(tmp, relin_keys);
            evaluator.multiply_inplace(tmp, ctxt_l_filter_2);
            evaluator.relinearize_inplace(tmp, relin_keys);
            ctxt_key_match2[i] = tmp;
        }

        for(int i = 0; i < numEle_p; i++) {
            Plaintext ptxt_mask;
            Ciphertext ctxt_partkey; // hold one single partkey
            Ciphertext ctxt_p_filtered;
            vector<int64_t> mask(numEle_p, 0);
            mask[i] = 1;
            batch_encoder.encode(mask, ptxt_mask);
            evaluator.multiply(ctxt_p_partkey, ctxt_p_filter_3, ctxt_p_filtered);
            evaluator.relinearize_inplace(ctxt_p_filtered, relin_keys);
            evaluator.multiply_plain_inplace(ctxt_p_filtered, ptxt_mask);
            
            // BATCH -> SINGLE
            ctxt_p_filtered = SUM(evaluator, ctxt_p_filtered, slot_count, galois_keys);

            evaluator.multiply_plain(ctxt_p_partkey, ptxt_mask, ctxt_partkey);
            // BATCH -> SINGLE
            ctxt_partkey = SUM(evaluator, ctxt_partkey, slot_count, galois_keys);
            Ciphertext tmp = comparator.isEqual(evaluator, relin_keys, 
                                                ctxt_partkey, ctxt_l_partkey);
            evaluator.multiply_inplace(tmp, ctxt_p_filtered);
            evaluator.relinearize_inplace(tmp, relin_keys);
            evaluator.multiply_inplace(tmp, ctxt_l_charge);
            evaluator.relinearize_inplace(tmp, relin_keys);
            evaluator.multiply_inplace(tmp, ctxt_l_filter_3);
            evaluator.relinearize_inplace(tmp, relin_keys);
            ctxt_key_match3[i] = tmp;
        }

        for(size_t i = 1; i < ctxt_key_match1.size(); i++) {
            evaluator.add_inplace(ctxt_key_match1[0], ctxt_key_match1[i]);
        }
        ctxt_joined1 = ctxt_key_match1[0];

        for(size_t i = 1; i < ctxt_key_match2.size(); i++) {
            evaluator.add_inplace(ctxt_key_match2[0], ctxt_key_match2[i]);
        }
        ctxt_joined2 = ctxt_key_match2[0];

        for(size_t i = 1; i < ctxt_key_match3.size(); i++) {
            evaluator.add_inplace(ctxt_key_match3[0], ctxt_key_match3[i]);
        }
        ctxt_joined3 = ctxt_key_match3[0];
    

        ctxt_joined1 = SUM(evaluator, ctxt_joined1, slot_count, galois_keys);

        ctxt_joined2 = SUM(evaluator, ctxt_joined2, slot_count, galois_keys);

        ctxt_joined3 = SUM(evaluator, ctxt_joined3, slot_count, galois_keys);

        evaluator.add_inplace(ctxt_joined1, ctxt_joined2);
        evaluator.add(ctxt_joined1, ctxt_joined3, ctxt_revenue);

        auto context_data = context.first_context_data();
        while (context_data->next_context_data())
        {
            evaluator.mod_switch_to_next_inplace(ctxt_revenue);
            context_data = context_data->next_context_data();
        }
        time_tpch.pause();

        if(debug != 0) {
            cout << "    + noise budget left: " << decryptor.invariant_noise_budget(ctxt_revenue) << " bits"
             << endl;
            int64_t revenue = dec<int64_t>(ctxt_revenue, numEle_l, batch_encoder, decryptor)[0];
            verify(l_quantity, l_shipmode, l_shipinstruct, l_charge,  p_partkey,
                   l_partkey, p_brand, p_size, p_container, revenue);
        }

    }
    time_tpch.stop(runs);
    time_select.stop(runs);
    time_where.stop(runs);
    if(debug != 0)
        cout << "Verified. Done.\n";
    else
        cout << "Verify skipped. Done" << endl;
    return 0;
}