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
#include "progressbar.hpp"
#include "parallel_hashmap/phmap_dump.h"

using boost::adaptors::transformed;
using boost::algorithm::join;
using namespace std;
using namespace phmap;


#define BINS_LOAD_THREADS 64
#define PAIRWISE_THREADS 64

#define KSIZE 51

typedef std::chrono::high_resolution_clock Time;

using BINS_MAP = parallel_flat_hash_map<std::string, phmap::flat_hash_set<uint64_t>,
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

    if (argc != 4) {
        cout << "run: ./pairwise_bins <bins_directory> <threads> <output>" << endl;
        exit(1);
    }
    string bins_dir = argv[1];
    int threads = stoi(argv[2]);
    string output = argv[3];
    int loading_cores = (int)threads * 0.5;


    // 1. Scan all bins in a directory
    vector<string> bins_paths;
    vector<string> bin_names;

    int total_bins_number = 0;
    for (const auto& dirEntry : glob2(bins_dir + "/*")) {
        string file_name = (string)dirEntry;
        size_t lastindex = file_name.find_last_of(".");
        string bin_prefix = file_name.substr(0, lastindex);
        std::string bin_basename = bin_prefix.substr(bin_prefix.find_last_of("/\\") + 1);
        std::string::size_type idx;
        idx = file_name.rfind('.');
        std::string extension = "";
        if (idx != std::string::npos) extension = file_name.substr(idx + 1);
        if (extension != "bin") continue;

        bin_names.push_back(bin_basename);
        bins_paths.push_back(file_name);
        total_bins_number++;
    }

    // 2. Load all bins in parallel
    cout << "Loading binnatures using " << loading_cores << " cores..." << endl;

    auto* bin_to_hashes = new BINS_MAP();
    auto begin_time = Time::now();
    int binIdx = 0;
    int N = bins_paths.size();
    // Loading all bins
    progressbar bar(bins_paths.size());
    bar.set_todo_char(" ");
    bar.set_done_char("█");
    bar.set_opening_bracket_char("{");
    bar.set_closing_bracket_char("}");
#pragma omp parallel num_threads(loading_cores)
    {
#pragma omp for
        for (int l = 0; l < bins_paths.size(); l++) {
            auto& bin_path = bins_paths[l];
            ++binIdx;
            // cout << "\r" << "loading " << binIdx << "/" << N;
            flat_hash_set<uint64_t> tmp_hashes;

            phmap::BinaryInputArchive ar_in(bin_path.c_str());
            tmp_hashes.phmap_load(ar_in);

            bin_to_hashes->try_emplace_l(bin_names[l],
                [](BINS_MAP::value_type& v) {},
                tmp_hashes
            );

#pragma omp critical
            bar.update();
        }
    }

    cout << endl;
    cout << "Loaded all binnatures in " << std::chrono::duration<double, std::milli>(Time::now() - begin_time).count() / 1000 << " secs" << endl;
    Combo combo = Combo();
    combo.combinations(total_bins_number);
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
            uint32_t bin_1_idx = seq_pair.first;
            uint32_t bin_2_idx = seq_pair.second;
            auto& bin1_hashes = bin_to_hashes->operator[](bin_names[bin_1_idx]);
            auto& bin2_hashes = bin_to_hashes->operator[](bin_names[bin_2_idx]);
            uint64_t shared_hashes = count_if(bin1_hashes.begin(), bin1_hashes.end(), [&](uint64_t k) {return bin2_hashes.find(k) != bin2_hashes.end();});
            if (!shared_hashes) continue;
            pairwise_hashtable.insert(pair(pair(bin_1_idx, bin_2_idx), shared_hashes));
        }
    }
    cout << "Pairwise comparisons done in " << std::chrono::duration<double, std::milli>(Time::now() - begin_time).count() / 1000 << " secs" << endl;
    cout << "writing pairwise matrix to " << output << endl;

    std::ofstream myfile;
    myfile.open(output + ".csv");
    myfile << "bin_1" << ',' << "bin_2" << ',' << "shared_kmers" << ',' << "max_containment" << '\n';
    for (const auto& edge : pairwise_hashtable) {
        auto bin_1_size = bin_to_hashes->operator[](bin_names[edge.first.first]).size();
        auto bin_2_size = bin_to_hashes->operator[](bin_names[edge.first.second]).size();
        float max_containment = (float)edge.second / std::min(bin_1_size, bin_2_size);
        myfile << bin_names[edge.first.first] << ',' << bin_names[edge.first.second] << ',' << edge.second << ',' << max_containment << '\n';
    }
    myfile.close();
}