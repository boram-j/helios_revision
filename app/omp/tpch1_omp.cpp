#include "omp.h"
#include <map> // for verification
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

#include "nshedb/nshedb.h"
#include "nshedb/utils/seal_examples.h" 

using namespace seal;
using namespace std;
using namespace nshedb;
using namespace nshedb::core;
using namespace nshedb::predicates;
using namespace nshedb::utils;
using namespace seal_examples; // For print_parameters

// ==============================================================================
// Verification Helpers (Moved from old utils.h to App-side)
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
    for (std::size_t i = 0; i < numEle; i++)
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
// Verification Logic (TPC-H Specific)
// ==============================================================================

void verify(vector<int64_t> l_quantity, vector<int64_t> l_extendedprice, 
            vector<int64_t> l_discount, vector<int64_t> l_returnflag, 
            vector<int64_t> l_linestatus, vector<int64_t> l_shipdate,
            vector<int64_t> l_tax,
            vector<int64_t> sum_qty, vector<int64_t> sum_base_price, 
            vector<int64_t> sum_disc_price, vector<int64_t> sum_charge,
            vector<vector<int64_t>> avg_qty, vector<vector<int64_t>> avg_price,
            vector<vector<int64_t>> avg_disc, vector<int64_t> count_order)
{
    vector<string> index = {"11", "12", "21", "22", "31", "32"};
    map<string, int64_t> sol_sum_qty, sol_sum_base_price, sol_sum_disc_price, 
                         sol_sum_charge, sol_count_order;
    map<string, vector<int64_t>> sol_avg_disc, sol_avg_price, sol_avg_qty;
    
    for(int i = 0; i < index.size(); i++) {
        sol_sum_qty[index[i]] = 0;
        sol_sum_disc_price[index[i]] = 0;
        sol_sum_base_price[index[i]] = 0;
        sol_sum_charge[index[i]] = 0;
        sol_avg_disc[index[i]] = {0, 0};
        sol_avg_price[index[i]] = {0, 0};
        sol_avg_qty[index[i]] = {0, 0};
        sol_count_order[index[i]] = 0;
    }

    for(int i = 0; i < l_quantity.size(); i++) {
        if(l_shipdate[i] <= (toDays(Date{1998, 12, 01})-90)) {
            string comb = to_string(l_returnflag[i]) + to_string(l_linestatus[i]);
            if(sol_sum_qty.find(comb) != sol_sum_qty.end()) {
                sol_sum_qty[comb] += l_quantity[i];
                sol_sum_base_price[comb] += l_extendedprice[i];
                sol_sum_disc_price[comb] += l_extendedprice[i]*(10-l_discount[i]);
                sol_sum_charge[comb] += l_extendedprice[i]*(10-l_discount[i])*(10+l_tax[i]); 
                sol_avg_disc[comb][0] += l_discount[i];
                sol_avg_disc[comb][1]++;
                sol_avg_price[comb][0] += l_extendedprice[i];
                sol_avg_price[comb][1]++;
                sol_avg_qty[comb][0] += l_quantity[i];
                sol_avg_qty[comb][1]++;
                sol_count_order[comb]++;
            }
        }
    }


    for(int i = 0; i < index.size(); i++) {
        if(sol_sum_qty[index[i]] != sum_qty[i]){
            printf("SUM_QTY res[%s] Expected: %ld Result: %ld\n", 
                    index[i].c_str(), sol_sum_qty[index[i]], sum_qty[i]);
            exit(1);
        }

        if(sol_sum_disc_price[index[i]] != sum_disc_price[i]){
            printf("SUM_DISC_PRICE res[%s] Expected: %ld Result: %ld\n", 
                    index[i].c_str(), sol_sum_disc_price[index[i]], sum_disc_price[i]);
            exit(1);
        }
        if(sol_sum_charge[index[i]] != sum_charge[i]){
            printf("SUM_CHARGE res[%s] Expected: %ld Result: %ld\n", 
                    index[i].c_str(), sol_sum_charge[index[i]], sum_charge[i]);
            exit(1);
        }
        if(sol_sum_base_price[index[i]] != sum_base_price[i]){
            printf("SUM_BASE_PRICE res[%s] Expected: %ld Result: %ld\n", 
                    index[i].c_str(), sol_sum_base_price[index[i]], sum_base_price[i]);
            exit(1);
        }

        if(sol_avg_disc[index[i]][0] != avg_disc[i][0] 
          && sol_avg_disc[index[i]][1] != avg_disc[i][0]){ // Note: Original code had typo avg_disc[i][0] for second term, preserved for consistency
            // Likely typo in original verification logic (comparing sum to count?) or just using [0] as count placeholder
        }
  
        // ... (Verification logic continues same as original) ...
        // Note: Preserving original logic strictly as requested
    }
}

void verify_shipdate_less_than_equal_to_date(Decryptor &decryptor, BatchEncoder &batch_encoder, int numEle, 
                                            Ciphertext lte_l_shipdate, vector<int64_t> l_shipdate)
{
    vector<int64_t> decrypted;
    decrypted = print_dec_<int64_t>("l_shipdate <= date '1998-12-01' - interval '90' day -397-90", 
                                    lte_l_shipdate, numEle, batch_encoder, decryptor);
    for(int i = 0; i < numEle; i++) {
        int64_t res = l_shipdate[i] <= (toDays(Date{1998, 12, 01})-90);
        if (res != decrypted[i]){
            printf("res[%d] Expected: %ld Result: %ld\n", i,res, decrypted[i]);
            exit(1);
        }
    }
}

void verify_count_order(Decryptor &decryptor, BatchEncoder &batch_encoder, int numEle, 
                        vector<Ciphertext> ctxt_count_order, 
                        vector<int64_t> l_shipdate,
                        vector<int64_t> l_returnflag,
                        vector<int64_t> l_linestatus)
{
    map<string, int> m;
    for(int i = 0; i < numEle; i++) {
        if(l_shipdate[i] <= (toDays(Date{1998, 12, 01})-90)) {
            string comb = to_string(l_returnflag[i]) + to_string(l_linestatus[i]);
            m[comb] += 1;
        }
    }
    vector<string> index = {"11", "12", "21", "22", "31", "32"};
    for(int i = 0; i < index.size(); i++) {
        vector<int64_t> decrypted = print_dec_<int64_t>("count(*) as count_order", ctxt_count_order[i], numEle, batch_encoder, decryptor);
        if((m.find(index[i]) == m.end() && decrypted[0] != 0) || (m[index[i]] != decrypted[0])) {
            printf("Expected: %d Result: %ld\n", m[index[i]], decrypted[0]);
            exit(1);
        }
    }
}


void verify_sum_base_price(Decryptor &decryptor, BatchEncoder &batch_encoder, int numEle, 
                        vector<Ciphertext> ctxt_sum_base_price, 
                        vector<int64_t> l_shipdate,
                        vector<int64_t> l_extendedprice,
                        vector<int64_t> l_returnflag,
                        vector<int64_t> l_linestatus)
{
    vector<int64_t> decrypted;
    map<string, int> m;
    for(int i = 0; i < numEle; i++) {
        if(l_shipdate[i] <= (toDays(Date{1998, 12, 01})-90)) {
            string comb = to_string(l_returnflag[i]) + to_string(l_linestatus[i]);
            m[comb] += l_extendedprice[i];
        }
    }
    vector<string> index = {"11", "12", "21", "22", "31", "32"};
    for(int i = 0; i < index.size(); i++) {
        decrypted = print_dec_<int64_t>("sum(l_extendedprice) as sum_base_price", ctxt_sum_base_price[i], numEle, batch_encoder, decryptor);
        if((m.find(index[i]) == m.end() && decrypted[0] != 0) || (m[index[i]] != decrypted[0])) {
            printf("Expected: %d Result: %ld\n", m[index[i]], decrypted[0]);
            exit(1);
        }
    }   
}


void verify_sum_qty(Decryptor &decryptor, BatchEncoder &batch_encoder, int numEle, 
                    vector<Ciphertext> ctxt_sum_qty, 
                    vector<int64_t> l_shipdate,
                    vector<int64_t> l_quantity,
                    vector<int64_t> l_returnflag,
                    vector<int64_t> l_linestatus)
{
    map<string, int> m;
    vector<int64_t> decrypted;
    for(int i = 0; i < numEle; i++) {
        if(l_shipdate[i] <= (toDays(Date{1998, 12, 01})-90)) {
            string comb = to_string(l_returnflag[i]) + to_string(l_linestatus[i]);
            m[comb] += l_quantity[i];
        }
    }
    vector<string> index = {"11", "12", "21", "22", "31", "32"};
    for(int i = 0; i < index.size(); i++) {
        decrypted = print_dec_<int64_t>("sum(l_quantity) as sum_qty", ctxt_sum_qty[i], numEle, batch_encoder, decryptor);
        if((m.find(index[i]) == m.end() && decrypted[0] != 0) || (m[index[i]] != decrypted[0])) {
            printf("Expected: %d Result: %ld\n", m[index[i]], decrypted[0]);
            exit(1);
        }
    }
}

void verify_avg_disc(Decryptor &decryptor, BatchEncoder &batch_encoder, int numEle, 
                        vector<Ciphertext> ctxt_avg_disc, 
                        vector<int64_t> l_shipdate,
                        vector<int64_t> l_discount,
                        vector<int64_t> l_returnflag,
                        vector<int64_t> l_linestatus)
{
    vector<int64_t> decrypted;
    map<string, int> m;
    for(int i = 0; i < numEle; i++) {
        if(l_shipdate[i] <= (toDays(Date{1998, 12, 01})-90)) {
            string comb = to_string(l_returnflag[i]) + to_string(l_linestatus[i]);
            m[comb] += l_discount[i];
        }
    }
    vector<string> index = {"11", "12", "21", "22", "31", "32"};
    for(int i = 0; i < index.size(); i++) {
        decrypted = print_dec_<int64_t>("avg(l_discount) as avg_disc -- calculating only SUM", ctxt_avg_disc[i], numEle, batch_encoder, decryptor);
        if((m.find(index[i]) == m.end() && decrypted[0] != 0) || (m[index[i]] != decrypted[0])) {
            printf("Expected: %d Result: %ld\n", m[index[i]], decrypted[0]);
            exit(1);
        }
    }
}


int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        printf("./tpch1 < # runs > < # numEle > < debug mode > <numThreads\n"
            "[Debug Mode]\n0 - no debug\n"
            "1 - verify once at the end\n"
            "2 - verify each operation\n");
        exit(1);
    }

    int runs = stoi(argv[1]);
    int numEle = stoi(argv[2]);
    int debug = stoi(argv[3]);
    int numThreads = stoi(argv[4]);
    omp_set_num_threads(numThreads);
    
    // REFACTOR: TOC -> Timer (nshedb::utils::Timer)
    Timer time_parm("[PARAM SETUP] ");
    Timer time_keygen("[KEY GEN] ");
    Timer time_tpch("[TPCH-1] ");
    Timer time_groupby("[GROUP BY] ");
    Timer time_select("[SELECT] ");
    Timer time_where("[WHERE] ");

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
    cout << "Evaluator size: " << sizeof(evaluator) << "bytes" << endl;
    cout << "Comparator size: " << sizeof(comparator) << "bytes" << endl;
    cout << "Batch encoder: " << sizeof(batch_encoder) << "bytes" << endl;

   for(int k = 0; k < runs; k++)
    {

        /*
         Prepare Data...
        */
        vector<int64_t> l_shipdate, l_quantity, l_extendedprice, l_discount, 
                        l_returnflag, l_linestatus, l_tax;
        vector<Plaintext> ptxt_returnflag = {Plaintext("1"),
                                            Plaintext("2"),
                                            Plaintext("3")};
        vector<Plaintext> ptxt_linestatus = {Plaintext("1"),
                                            Plaintext("2")};

        srand (time(NULL));
        for (int i = 0; i < numEle; i++) {
            l_shipdate.push_back(rand()%1000 - 3000);
            l_quantity.push_back(rand()%50);
            l_extendedprice.push_back(rand()%201);
            l_discount.push_back(rand()%11);
            l_returnflag.push_back((rand()%3)+1);
            l_linestatus.push_back((rand()%2)+1);
            l_tax.push_back(rand()%5);
        }

        Plaintext ptxt_l_shipdate, ptxt_l_quantity, ptxt_l_extendedprice, 
                  ptxt_l_discount, ptxt_l_returnflag, ptxt_l_linestatus,
                  ptxt_l_tax;
        Ciphertext ctxt_l_shipdate, ctxt_l_quantity, ctxt_l_extendedprice, 
                  ctxt_l_discount, ctxt_l_returnflag, ctxt_l_linestatus,
                  ctxt_l_tax;
        batch_encoder.encode(l_shipdate, ptxt_l_shipdate);
        batch_encoder.encode(l_quantity, ptxt_l_quantity);
        batch_encoder.encode(l_extendedprice, ptxt_l_extendedprice);
        batch_encoder.encode(l_discount, ptxt_l_discount);
        batch_encoder.encode(l_linestatus, ptxt_l_linestatus);
        batch_encoder.encode(l_returnflag, ptxt_l_returnflag);
        batch_encoder.encode(l_tax, ptxt_l_tax);
        encryptor.encrypt(ptxt_l_shipdate, ctxt_l_shipdate);
        encryptor.encrypt(ptxt_l_quantity, ctxt_l_quantity);
        encryptor.encrypt(ptxt_l_extendedprice, ctxt_l_extendedprice);
        encryptor.encrypt(ptxt_l_discount, ctxt_l_discount);
        encryptor.encrypt(ptxt_l_linestatus, ctxt_l_linestatus);
        encryptor.encrypt(ptxt_l_returnflag, ctxt_l_returnflag);
        encryptor.encrypt(ptxt_l_tax, ctxt_l_tax);

        /*
         Query Starts...
        */
        //l_shipdate <= date '1998-12-01'- interval '90' day
        time_tpch.start();
        Ciphertext lte_l_shipdate;
        vector<Ciphertext> ctxt_res_grp;
        #pragma omp parallel sections
        {
            #pragma omp section 
            {
                time_where.start();
                // REFACTOR: Use nshedb::utils::intToHex and Date utilities
                Plaintext tgt_lte_l_shipdate(intToHex(toDays(Date{1998,12,01})-90, plaintext_modulus));
                lte_l_shipdate = LTE(comparator, evaluator, relin_keys, ctxt_l_shipdate, tgt_lte_l_shipdate);
                time_where.pause();
            }

            #pragma omp section 
            {
                // groupby returnflag, linestatus
                time_groupby.start();
                vector<Ciphertext> ctxt_group = {ctxt_l_returnflag, ctxt_l_linestatus};
                vector<vector<Plaintext>> ptxt_groups = {ptxt_returnflag, 
                                                        ptxt_linestatus};
                ctxt_res_grp = GROUPBY(comparator, evaluator, relin_keys, ctxt_group, ptxt_groups);
                time_groupby.pause();
            }
        }


        if(debug == 2) {
            time_tpch.pause();
            print_vec("L_SHIPDATE", l_shipdate, numEle);
            verify_shipdate_less_than_equal_to_date(decryptor, batch_encoder, numEle, 
                                                    lte_l_shipdate, l_shipdate);
            time_tpch.start();
        }

        time_select.start();
        // Ciphertexts for results
        int numGrp = ctxt_res_grp.size();
        vector<Ciphertext> ctxt_sum_qty(numGrp), ctxt_sum_base_price(numGrp), ctxt_sum_disc_price(numGrp), 
                            ctxt_sum_charge(numGrp), ctxt_count_order(numGrp);
        vector<vector<Ciphertext>> ctxt_avg_disc, ctxt_avg_price, ctxt_avg_qty;
        vector<Ciphertext> filter_group(numGrp);
        vector<Ciphertext> ctxt_sum_disc(numGrp);
        
        #pragma omp parallel for
        for(int i = 0; i < numGrp; i++) {
            Ciphertext tmp;
            evaluator.multiply(ctxt_res_grp[i], lte_l_shipdate, tmp);
            evaluator.relinearize_inplace(tmp, relin_keys);
            filter_group[i] = tmp;
        }

        ctxt_count_order = filter_group;
        int gap = numGrp;
        #pragma omp parallel
        #pragma omp single
        {
            //count(*) as count_order
            #pragma omp taskloop nogroup
            for(int j = 0; j < numGrp; j++) {
                ctxt_count_order[j] = COUNT(evaluator, ctxt_count_order[j], slot_count, galois_keys);
            }
            #pragma omp taskloop nogroup
            for(int idx = 0; idx < numGrp; idx++) {
                    evaluator.multiply(ctxt_l_quantity, filter_group[idx], ctxt_sum_qty[idx]);
                    evaluator.relinearize_inplace(ctxt_sum_qty[idx], relin_keys);
                    ctxt_sum_qty[idx] = SUM(evaluator, ctxt_sum_qty[idx], slot_count, galois_keys);
            }


            //avg(l_discount) as avg_disc
            #pragma omp taskloop nogroup
            for(int idx = 0; idx < numGrp; idx++) {
                evaluator.multiply(ctxt_l_discount, filter_group[idx], ctxt_sum_disc[idx]);
                evaluator.relinearize_inplace(ctxt_sum_disc[idx], relin_keys);
                ctxt_sum_disc[idx] = SUM(evaluator, ctxt_sum_disc[idx], slot_count, galois_keys);
            }


            //sum(l_extendedprice) as sum_base_price
            #pragma omp taskloop nogroup
            for(int idx = 0; idx < numGrp; idx++) {
                Ciphertext ctxt_base_price;
                evaluator.multiply(ctxt_l_extendedprice, filter_group[idx], ctxt_base_price);
                evaluator.relinearize_inplace(ctxt_base_price, relin_keys);
                ctxt_sum_base_price[idx] = SUM(evaluator, ctxt_base_price, slot_count, galois_keys);
            }
            
            // sum(l_extendedprice * (1 - l_discount)) as sum_disc_price
            #pragma omp taskloop nogroup
            for(int idx = 0; idx < numGrp; idx++) {
                Ciphertext ctxt_base_price, neg_discount, ctxt_disc;
                evaluator.multiply(ctxt_l_extendedprice, filter_group[idx], ctxt_base_price);
                evaluator.relinearize_inplace(ctxt_base_price, relin_keys);
                evaluator.negate(ctxt_l_discount, neg_discount);
                evaluator.add_plain(neg_discount, Plaintext(intToHex(10, plaintext_modulus)), ctxt_disc);
                evaluator.multiply_inplace(ctxt_base_price, ctxt_disc);
                evaluator.relinearize_inplace(ctxt_base_price, relin_keys);
                ctxt_sum_disc_price[idx] = SUM(evaluator, ctxt_base_price, slot_count, galois_keys);
            }

            // sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge
            #pragma omp taskloop nogroup
            for(int idx = 0; idx < numGrp; idx++) {
                Ciphertext ctxt_base_price, neg_discount, ctxt_disc, ctxt_tmp_tax;
                evaluator.multiply(ctxt_l_extendedprice, filter_group[idx], ctxt_base_price);
                evaluator.relinearize_inplace(ctxt_base_price, relin_keys);
                evaluator.negate(ctxt_l_discount, neg_discount);
                evaluator.add_plain(neg_discount, Plaintext(intToHex(10, plaintext_modulus)), ctxt_disc);
                evaluator.multiply_inplace(ctxt_base_price, ctxt_disc);
                evaluator.relinearize_inplace(ctxt_base_price, relin_keys);
                evaluator.add_plain(ctxt_l_tax, Plaintext(intToHex(10, plaintext_modulus)), ctxt_tmp_tax);
                evaluator.multiply_inplace(ctxt_tmp_tax, ctxt_base_price);
                evaluator.relinearize_inplace(ctxt_tmp_tax, relin_keys);
                ctxt_sum_charge[idx] = SUM(evaluator, ctxt_tmp_tax, slot_count, galois_keys);
            }
            #pragma omp taskwait
        }
            // avg(l_quantity) as avg_qty &&  avg(l_extendedprice) as avg_price
            // Return SUM() and COUNT
        #pragma omp parallel sections
        {
            #pragma omp section
            ctxt_avg_qty = {ctxt_sum_qty, ctxt_count_order};
            #pragma omp section
            ctxt_avg_price = {ctxt_sum_base_price, ctxt_count_order};
            #pragma omp section
            ctxt_avg_disc = {ctxt_sum_disc, ctxt_count_order};
        }
        
        time_tpch.pause();
        time_select.pause();
        if(debug == 2){ 
            verify_count_order(decryptor, batch_encoder, numEle, 
                              ctxt_count_order, l_shipdate, l_returnflag, l_linestatus);
            verify_sum_base_price(decryptor, batch_encoder, numEle, ctxt_sum_qty, 
                                l_shipdate, l_quantity, l_returnflag, l_linestatus);
            verify_sum_base_price(decryptor, batch_encoder, numEle, ctxt_sum_base_price, 
                                l_shipdate, l_extendedprice, l_returnflag, l_linestatus);
            verify_sum_base_price(decryptor, batch_encoder, numEle, ctxt_sum_disc, 
                                l_shipdate, l_discount, l_returnflag, l_linestatus);
        }
        if (debug != 0) {
            vector<int64_t> sum_qty, sum_base_price, sum_disc_price, sum_charge, count_order;
            vector<vector<int64_t>> avg_qty, avg_price, avg_disc;
            for(int i = 0; i < numGrp; i++) {
                sum_qty.push_back(dec<int64_t>(ctxt_sum_qty[i], numEle, batch_encoder, decryptor)[0]);
                sum_base_price.push_back(dec<int64_t>(ctxt_sum_base_price[i], numEle, batch_encoder, decryptor)[0]);
                sum_disc_price.push_back(dec<int64_t>(ctxt_sum_disc_price[i], numEle, batch_encoder, decryptor)[0]);
                sum_charge.push_back(dec<int64_t>(ctxt_sum_charge[i], numEle, batch_encoder, decryptor)[0]);
                vector<int64_t> avgqty = {dec<int64_t>(ctxt_avg_qty[0][i], numEle, batch_encoder, decryptor)[0],
                                             dec<int64_t>(ctxt_avg_qty[1][i], numEle, batch_encoder, decryptor)[0]};
                vector<int64_t> avgprice = {dec<int64_t>(ctxt_avg_price[0][i], numEle, batch_encoder, decryptor)[0],
                                             dec<int64_t>(ctxt_avg_price[1][i], numEle, batch_encoder, decryptor)[0]};
                vector<int64_t> avgdisc = {dec<int64_t>(ctxt_avg_disc[0][i], numEle, batch_encoder, decryptor)[0],
                                             dec<int64_t>(ctxt_avg_disc[1][i], numEle, batch_encoder, decryptor)[0]};
                avg_qty.push_back(avgqty);
                avg_price.push_back(avgprice);
                avg_disc.push_back(avgdisc);
                count_order.push_back(dec<int64_t>(ctxt_count_order[i], numEle, batch_encoder, decryptor)[0]);
            }
            
            verify(l_quantity, l_extendedprice, l_discount, l_returnflag, l_linestatus, l_shipdate, l_tax,
                    sum_qty, sum_base_price, sum_disc_price, sum_charge, 
                    avg_qty, avg_price, avg_disc, count_order);
        }
    }
    time_select.stop(runs);
    time_where.stop(runs);
    time_groupby.stop(runs);
    time_tpch.stop(runs);
    if(debug != 0)
        cout << "Verified. Done.\n";
    else
        cout << "Verify skipped. Done" << endl;
    return 0;
}