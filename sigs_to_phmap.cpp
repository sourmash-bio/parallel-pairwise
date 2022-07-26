#include <iostream>
#include <cstdint>
#include <chrono>
#include "parallel_hashmap/phmap.h"
#include <ctime>
#include "combination.hpp"
#include<omp.h>
#include "RSJparser.tcc"
#include <glob.h>
#include <string>
#include <stdexcept>
#include "parallel_hashmap/phmap_dump.h"
#include <cstdlib>

using namespace std;
// using namespace phmap;
using JSON = RSJresource;

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

template<>
uint64_t RSJresource::as<uint64_t>(const uint64_t& def) {
    if (!exists()) return (0); return (unrolled(data));
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

    vector<string> filenames;
    for (size_t i = 0; i < glob_result.gl_pathc; ++i)
        filenames.push_back(string(glob_result.gl_pathv[i]));

    // cleanup
    globfree(&glob_result);

    // done
    return filenames;
}


int main(int argc, char** argv) {

    if (argc != 6) {
        cout << "run: ./pairwise <sigs_directory> <kSize> <output_dir> <threads> <chunks>" << endl;
        exit(1);
    }
    string sigs_dir = argv[1];
    int kSize = stoi(argv[2]);
    string output_dir = argv[3];
    int user_threads = stoi(argv[4]);
    int user_chunks = stoi(argv[5]);

    string cmd = "mkdir -p " + output_dir;

    const int dir_err = system(cmd.c_str());
    if (-1 == dir_err)
    {
        printf("Error creating directory!n");
        exit(1);
    }


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


    vector<tuple<string, JSON>> json_map;
    int chunk =  user_chunks;
    if (user_chunks > sig_names.size()){
        cout << "chunk size set to " << sig_names.size();
        chunk = sig_names.size();
    }
    cout << "chunk size: " << chunk << endl;
    int chunk_count = 0;
    int done_chunks = 0;
    cout << "load without parsing" << endl;
    auto begin_time = Time::now();


    for (int j = 0; j < sigs_paths.size(); j++) {
        cout << "\r" << "loading " << chunk_count + 1 << "/" << chunk;
        string& sig_path = sigs_paths[j];
        std::ifstream sig_stream(sig_path);
        JSON sig(sig_stream);
        json_map.push_back(make_tuple(sig_names[j], sig));
        chunk_count++;

        if (chunk_count < chunk) continue;

#pragma omp parallel num_threads(user_threads)
        {
#pragma omp for
            for (int m = 0; m < chunk; m++) {
                auto& _pair = json_map[m];
                auto& sig_name = get<0>(_pair);
                JSON& sig = get<1>(_pair);
                phmap::flat_hash_set<uint64_t> tmp_hashes;
                int number_of_sub_sigs = sig[0]["signatures"].size();
                for (int i = 0; i < number_of_sub_sigs; i++) {
                    int current_kSize = sig[0]["signatures"][i]["ksize"].as<int>();
                    auto loaded_sig_it = sig[0]["signatures"][i]["mins"].as_array().begin();
                    if (current_kSize == kSize) {
                        while (loaded_sig_it != sig[0]["signatures"][i]["mins"].as_array().end()) {
                            tmp_hashes.insert(loaded_sig_it->as<uint64_t>());
                            loaded_sig_it++;
                        }
                        break;
                    }
                }
                string out_path = output_dir + "/" + sig_name + ".bin";
                phmap::BinaryOutputArchive ar_out(out_path.c_str());
                tmp_hashes.phmap_dump(ar_out);
            }
        }
        chunk_count = 0;
        cout << "\nprocessed " << ++done_chunks << "/" << (int)(sig_names.size() / chunk) << " chunks..." << endl;
        json_map.clear();
    }

    if (chunk_count) {
        for (auto& _pair : json_map) {
            auto& sig_name = get<0>(_pair);
            JSON& sig = get<1>(_pair);
            phmap::flat_hash_set<uint64_t> tmp_hashes;
            int number_of_sub_sigs = sig[0]["signatures"].size();
            for (int i = 0; i < number_of_sub_sigs; i++) {
                int current_kSize = sig[0]["signatures"][i]["ksize"].as<int>();
                auto loaded_sig_it = sig[0]["signatures"][i]["mins"].as_array().begin();
                if (current_kSize == kSize) {
                    while (loaded_sig_it != sig[0]["signatures"][i]["mins"].as_array().end()) {
                        tmp_hashes.insert(loaded_sig_it->as<uint64_t>());
                        loaded_sig_it++;
                    }
                    break;
                }
            }
            string out_path = output_dir + "/" + sig_name + ".bin";
            phmap::BinaryOutputArchive ar_out(out_path.c_str());
            tmp_hashes.phmap_dump(ar_out);
        }
    }

    cout << endl;
    cout << "Loaded all signatures in " << std::chrono::duration<double, std::milli>(Time::now() - begin_time).count() / 1000 << " secs" << endl;

}