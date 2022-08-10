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
#include <glob.h>
#include <string>
#include <stdexcept>
#include "cpp_json.h"
#include "zstr.hpp"
#include "progressbar.hpp"

using boost::adaptors::transformed;
using boost::algorithm::join;
using namespace std;
using namespace phmap;


#define SIGS_LOAD_THREADS 64
#define PAIRWISE_THREADS 64

#define KSIZE 51

typedef std::chrono::high_resolution_clock Time;

using SIGS_MAP = parallel_flat_hash_map<std::string, phmap::flat_hash_set<uint64_t>,
    phmap::priv::hash_default_hash<std::string>,
    phmap::priv::hash_default_eq<std::string>,
    std::allocator<std::pair<const std::string, phmap::flat_hash_set<uint64_t>>>,
    1,
    std::mutex
>;

using PAIRWISE_MAP = parallel_flat_hash_map<std::pair<uint32_t, uint32_t>,
    float, boost::hash<pair<uint32_t, uint32_t>>,
    std::equal_to<std::pair<uint32_t, uint32_t>>,
    std::allocator<std::pair<const std::pair<uint32_t, uint32_t>, float>>,
    12, std::mutex>;


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

    vector<string> filenames;
    for (size_t i = 0; i < glob_result.gl_pathc; ++i)
        filenames.push_back(string(glob_result.gl_pathv[i]));

    // cleanup
    globfree(&glob_result);

    // done
    return filenames;
}


int main(int argc, char** argv) {

    if (argc != 5) {
        cout << "run: ./pairwise <sigs_directory> <threads> <kSize> <output>" << endl;
        exit(1);
    }
    string sigs_dir = argv[1];
    int threads = stoi(argv[2]);
    int kSize = stoi(argv[3]);
    string output = argv[4];
    int loading_cores = (int)threads * 0.5;


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

    // 2. Load all sigs in parallel
    cout << "Loading signatures using " << loading_cores << " cores..." << endl;

    auto* sig_to_hashes = new SIGS_MAP();
    auto begin_time = Time::now();
    int sigIdx = 0;
    int N = sigs_paths.size();
    // Loading all bins
    progressbar bar(sigs_paths.size());
    bar.set_todo_char(" ");
    bar.set_done_char("█");
    bar.set_opening_bracket_char("{");
    bar.set_closing_bracket_char("}");
#pragma omp parallel num_threads(loading_cores)
    {
#pragma omp for
        for (int l = 0; l < sigs_paths.size(); l++) {
            auto& sig_path = sigs_paths[l];
            ++sigIdx;
            // cout << "\r" << "loading " << sigIdx << "/" << N;
            flat_hash_set<uint64_t> tmp_hashes;
            zstr::ifstream sig_stream(sig_path);
            json::value json = json::parse(sig_stream);
            auto sourmash_sig = json[0]["signatures"];
            auto sig_name = sig_names[l];
            const json::array& sig_array = as_array(sourmash_sig);
            for (auto it = sig_array.begin(); it != sig_array.end(); ++it) {
                const json::value& v = *it;
                if (v["ksize"] == kSize) {
                    const json::array& mins = as_array(v["mins"]);
                    auto mins_it = mins.begin();
                    while (mins_it != mins.end()) {
                        tmp_hashes.insert(json::to_number<uint64_t>(*mins_it));
                        mins_it++;
                    }
                }
                break;
            }

            sig_to_hashes->try_emplace_l(sig_name,
                [](SIGS_MAP::value_type& v) {},
                tmp_hashes
                );

#pragma omp critical
                bar.update();
        }
    }

    cout << endl;
    cout << "Loaded all signatures in " << std::chrono::duration<double, std::milli>(Time::now() - begin_time).count() / 1000 << " secs" << endl;
    Combo combo = Combo();
    combo.combinations(total_sigs_number);
    PAIRWISE_MAP pairwise_hashtable;
    begin_time = Time::now();
    int thread_num_1, num_threads_1, start_1, end_1, vec_i_1;
    int n_1 = combo.combs.size();
    cout << "Performing " << n_1 * n_1 << " pairwise comparisons using " << threads << " cores ..." << endl;
    omp_set_num_threads(threads);

#pragma omp parallel private(vec_i_1,thread_num_1,num_threads_1,start_1,end_1)
    {

        thread_num_1 = omp_get_thread_num();
        num_threads_1 = omp_get_num_threads();
        start_1 = thread_num_1 * n_1 / num_threads_1;
        end_1 = (thread_num_1 + 1) * n_1 / num_threads_1;

        for (vec_i_1 = start_1; vec_i_1 != end_1; ++vec_i_1) {
            auto const& seq_pair = combo.combs[vec_i_1];
            uint32_t sig_1_idx = seq_pair.first;
            uint32_t sig_2_idx = seq_pair.second;
            auto& sig1_hashes = sig_to_hashes->operator[](sig_names[sig_1_idx]);
            auto& sig2_hashes = sig_to_hashes->operator[](sig_names[sig_2_idx]);
            uint64_t shared_hashes = count_if(sig1_hashes.begin(), sig1_hashes.end(), [&](uint64_t k) {return sig2_hashes.find(k) != sig2_hashes.end();});
            if (!shared_hashes) continue;
            pairwise_hashtable.insert(pair(pair(sig_1_idx, sig_2_idx), shared_hashes));
        }
    }
    cout << "Pairwise comparisons done in " << std::chrono::duration<double, std::milli>(Time::now() - begin_time).count() / 1000 << " secs" << endl;
    cout << "writing pairwise matrix to " << output << endl;

    std::ofstream myfile;
    myfile.open(output + ".csv");
    myfile << "sig_1" << ',' << "sig_2" << ',' << "shared_kmers" << ',' << "max_containment" << '\n';
    for (const auto& edge : pairwise_hashtable) {
        auto sig_1_size = sig_to_hashes->operator[](sig_names[edge.first.first]).size();
        auto sig_2_size = sig_to_hashes->operator[](sig_names[edge.first.second]).size();
        float max_containment = (float)edge.second / std::min(sig_1_size, sig_2_size);
        myfile << sig_names[edge.first.first] << ',' << sig_names[edge.first.second] << ',' << edge.second << ',' << max_containment << '\n';
    }
    myfile.close();
}