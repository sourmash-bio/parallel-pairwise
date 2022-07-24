#include <iostream>
#include <cstdint>
#include <chrono>
#include "parallel_hashmap/phmap.h"
#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/functional/hash.hpp>
#include <ctime>
#include "combination.hpp"
#include<omp.h>
#include "RSJparser.tcc"
#include <glob.h>
#include <string>
#include <stdexcept>

using boost::adaptors::transformed;
using boost::algorithm::join;
using namespace std;
using namespace phmap;
using JSON = RSJresource;


#define SIGS_LOAD_THREADS 64
#define PAIRWISE_THREADS 64

#define KSIZE 51

typedef std::chrono::high_resolution_clock Time;


inline uint64_t unrolled(std::string const& value) {
    uint64_t result = 0;

    size_t const length = value.size();
    switch (length) {
    case 20:    result += (value[length - 20] - '0') * 10000000000000000000ULL;
    case 19:    result += (value[length - 19] - '0') * 1000000000000000000ULL;
    case 18:    result += (value[length - 18] - '0') * 100000000000000000ULL;
    case 17:    result += (value[length - 17] - '0') * 10000000000000000ULL;
    case 16:    result += (value[length - 16] - '0') * 1000000000000000ULL;
    case 15:    result += (value[length - 15] - '0') * 100000000000000ULL;
    case 14:    result += (value[length - 14] - '0') * 10000000000000ULL;
    case 13:    result += (value[length - 13] - '0') * 1000000000000ULL;
    case 12:    result += (value[length - 12] - '0') * 100000000000ULL;
    case 11:    result += (value[length - 11] - '0') * 10000000000ULL;
    case 10:    result += (value[length - 10] - '0') * 1000000000ULL;
    case  9:    result += (value[length - 9] - '0') * 100000000ULL;
    case  8:    result += (value[length - 8] - '0') * 10000000ULL;
    case  7:    result += (value[length - 7] - '0') * 1000000ULL;
    case  6:    result += (value[length - 6] - '0') * 100000ULL;
    case  5:    result += (value[length - 5] - '0') * 10000ULL;
    case  4:    result += (value[length - 4] - '0') * 1000ULL;
    case  3:    result += (value[length - 3] - '0') * 100ULL;
    case  2:    result += (value[length - 2] - '0') * 10ULL;
    case  1:    result += (value[length - 1] - '0');
    }
    return result;
}

using SIGS_MAP = parallel_flat_hash_map<std::string, phmap::flat_hash_set<uint64_t>,
    phmap::priv::hash_default_hash<std::string>,
    phmap::priv::hash_default_eq<std::string>,
    std::allocator<std::pair<const std::string,
    phmap::flat_hash_set<uint64_t>>>,
    12
    >;

using PAIRWISE_MAP = parallel_flat_hash_map<std::pair<uint32_t, uint32_t>,
    float, boost::hash<pair<uint32_t, uint32_t>>,
    std::equal_to<std::pair<uint32_t, uint32_t>>,
    std::allocator<std::pair<const std::pair<uint32_t, uint32_t>, float>>,
    12, std::mutex>;


template <typename T>
void ascending(T& dFirst, T& dSecond)
{
    if (dFirst > dSecond)
        std::swap(dFirst, dSecond);
}

template<>
uint64_t RSJresource::as<uint64_t>(const uint64_t& def) {
    if (!exists()) return (0); // required
    return (unrolled(data)); // example
}


std::vector<std::string> glob2(const std::string& pattern) {
    using namespace std;

    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // do the glob operation
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if (return_value != 0) {
        globfree(&glob_result);
        stringstream ss;
        ss << "glob() failed with return_value " << return_value << endl;
        throw std::runtime_error(ss.str());
    }

    // collect all the filenames into a std::list<std::string>
    vector<string> filenames;
    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
        filenames.push_back(string(glob_result.gl_pathv[i]));
    }

    // cleanup
    globfree(&glob_result);

    // done
    return filenames;
}


int main(int argc, char** argv) {

    if (argc != 4) {
        cout << "run: ./pairwise <sigs_directory> <threads> <output>" << endl;
        exit(1);
    }
    string sigs_dir = argv[1];
    int threads = stoi(argv[2]);
    string output = argv[3];


    // 1. Scan all sigs in a directory
    vector<string> sigs_paths;
    vector<string> sig_names;

    int total_sigs_number = 0;
    for (const auto& dirEntry : glob2(sigs_dir + "/*")) {
        string file_name = (string)dirEntry;
        size_t lastindex = file_name.find_last_of(".");
        string sig_prefix = file_name.substr(0, lastindex);
        std::string sig_basename = sig_prefix.substr(sig_prefix.find_last_of("/\\") + 1);
        std::string::size_type idx;
        idx = file_name.rfind('.');
        std::string extension = "";
        if (idx != std::string::npos) extension = file_name.substr(idx + 1);
        if (extension != "sig" && extension != "gz") continue;

        sig_names.push_back(sig_basename);
        sigs_paths.push_back(file_name);
        total_sigs_number++;
    }

    cout << "[i] Loading " << total_sigs_number << " sigs using " << threads << " cores..." << endl;

    // 2. Load all sigs in parallel
    SIGS_MAP sig_to_hashes;
    int thread_num, num_threads, start, end, vec_i;
    int n = total_sigs_number;
    omp_set_num_threads(threads);
    auto begin_time = Time::now();
#pragma omp parallel private(vec_i,thread_num,num_threads,start,end)
    {
        thread_num = omp_get_thread_num();
        num_threads = omp_get_num_threads();
        start = thread_num * n / num_threads;
        end = (thread_num + 1) * n / num_threads;

        for (vec_i = start; vec_i != end; ++vec_i) {
            flat_hash_set<uint64_t> tmp_hashes;
            auto sig_path = sigs_paths[vec_i];
            std::ifstream sig_stream(sig_path);
            JSON sig(sig_stream);
            int number_of_sub_sigs = sig[0]["signatures"].size();
            for (int i = 0; i < number_of_sub_sigs; i++) {
                int current_kSize = sig[0]["signatures"][i]["ksize"].as<int>();
                auto loaded_sig_it = sig[0]["signatures"][i]["mins"].as_array().begin();
                if (current_kSize == KSIZE) {
                    while (loaded_sig_it != sig[0]["signatures"][i]["mins"].as_array().end()) {
                        tmp_hashes.insert(loaded_sig_it->as<uint64_t>());
                        loaded_sig_it++;
                    }
                    break;
                }
            }
            sig_to_hashes.insert(pair(sig_names[vec_i], tmp_hashes));
        }
    }
    cout << "Loaded all signatures in " << std::chrono::duration<double, std::milli>(Time::now() - begin_time).count() / 1000 << " secs" << endl;
    cout << "Performing pairwise comparisons using " << threads << " cores ..." << endl;
    Combo combo = Combo();
    combo.combinations(total_sigs_number);
    PAIRWISE_MAP pairwise_hashtable;
    begin_time = Time::now();

    int thread_num_1, num_threads_1, start_1, end_1, vec_i_1;
    int n_1 = combo.combs.size();
    omp_set_num_threads(threads);

#pragma omp parallel private(vec_i_1,thread_num_1,num_threads_1,start_1,end_1)
    {

        thread_num_1 = omp_get_thread_num();
        num_threads_1 = omp_get_num_threads();
        start_1 = thread_num_1 * n_1 / num_threads_1;
        end_1 = (thread_num_1 + 1) * n_1 / num_threads_1;

        for (vec_i = start_1; vec_i_1 != end_1; ++vec_i_1) {
            auto const& seq_pair = combo.combs[vec_i_1];
            uint32_t sig_1_idx = seq_pair.first;
            uint32_t sig_2_idx = seq_pair.second;
            auto sig_1_size = sig_to_hashes[sig_names[sig_1_idx]].size();
            auto sig_2_size = sig_to_hashes[sig_names[sig_2_idx]].size();
            auto& sig1_hashes = sig_to_hashes[sig_names[sig_1_idx]];
            auto& sig2_hashes = sig_to_hashes[sig_names[sig_2_idx]];
            uint64_t shared_hashes = count_if(sig1_hashes.begin(), sig1_hashes.end(), [&](uint64_t k) {return sig2_hashes.find(k) != sig2_hashes.end();});
            if (!shared_hashes) continue;
            float max_containment = (float)shared_hashes / std::min(sig_1_size, sig_2_size);
            pairwise_hashtable.insert(pair(pair(sig_1_idx, sig_2_idx), max_containment));
        }
    }

    cout << "Pairwise comparisons done in " << std::chrono::duration<double, std::milli>(Time::now() - begin_time).count() / 1000 << " secs" << endl;
    cout << "writing pairwise matrix to" << output << ".csv" << endl;

    std::ofstream myfile;
    myfile.open(output + ".csv");
    myfile << "sig_1" << ',' << "sig_2" << ',' << "max_containment" << '\n';
    for (const auto& edge : pairwise_hashtable)
        myfile << sig_names[edge.first.first] << ',' << sig_names[edge.first.second] << ',' << edge.second << '\n';
    myfile.close();
}