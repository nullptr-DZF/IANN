#include "../hpp/PurePathSearch.h"
#include "../hpp/SpaceFactory.h"

#include "../hpp/ReadExpected.h"
#include "../hpp/VectorDataLoader.h"
#include "../hpp/SecondaryRuntime.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace {

bool file_exists(const std::string &path) {
    std::ifstream f(path.c_str());
    return f.good();
}

void create_directory_if_not_exists(const std::string &path) {
    if (file_exists(path)) {
        return;
    }
#if defined(_WIN32)
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0777);
#endif
}

std::string get_env_or_default(const char *key, const std::string &fallback) {
    const char *v = std::getenv(key);
    if (!v || std::string(v).empty()) {
        return fallback;
    }
    return std::string(v);
}

std::string join_labels(const std::vector<hnswlib::labeltype> &xs) {
    std::ostringstream oss;
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i) {
            oss << " ";
        }
        oss << xs[i];
    }
    return oss.str();
}

std::string escape_csv(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '"') {
            out.push_back('"');
        }
        out.push_back(s[i]);
    }
    out.push_back('"');
    return out;
}

PurePathSearcher<float>::SearchMode parse_mode(const std::string &mode_str) {
    if (mode_str == "single") {
        return PurePathSearcher<float>::SearchMode::Single;
    }
    if (mode_str == "bi_serial") {
        return PurePathSearcher<float>::SearchMode::BidirectionalSerial;
    }
    throw std::runtime_error("PURE_MODE must be one of: single, bi_serial, collect_unreachable");
}

void run_collect_unreachable(
    PurePathSearcher<float>               &searcher,
    const ExpectedDataReader<float>::ExpectedData &expected_data,
    const float                           *queries,
    size_t                                 d_q,
    size_t                                 n_q,
    int                                    max_queries,
    const std::string                     &input_expected_path,
    const std::string                     &output_root)
{

    auto derive_output_path = [&]() -> std::string {
        
        std::string dir  = output_root + "/pure_path_search";
        create_directory_if_not_exists(output_root);
        create_directory_if_not_exists(dir);

        size_t slash = input_expected_path.find_last_of("/\\");
        std::string fname = (slash == std::string::npos)
                                ? input_expected_path
                                : input_expected_path.substr(slash + 1);
        size_t dot = fname.rfind('.');
        std::string stem = (dot == std::string::npos) ? fname : fname.substr(0, dot);
        std::string ext  = (dot == std::string::npos) ? ""    : fname.substr(dot);

        std::string input_dir = (slash == std::string::npos)
                                    ? "."
                                    : input_expected_path.substr(0, slash);
        return input_dir + "/" + stem + "_unreachable" + ext;
    };

    const std::string out_path = derive_output_path();

    int efs = 500, k = 100;
    {
        auto it = expected_data.params.find("ef_search");
        if (it != expected_data.params.end()) efs = it->second;
        it = expected_data.params.find("k");
        if (it != expected_data.params.end()) k = it->second;
    }

    size_t total_queries = expected_data.expected_vectors_per_query.size();
    std::vector<std::vector<std::tuple<size_t, size_t, std::vector<float>>>>
        unreachable(total_queries);

    size_t total_cases       = 0;
    size_t unreachable_cases = 0;

    for (size_t q = 0; q < total_queries; ++q) {
        if (q >= n_q) break;
        if (max_queries > 0 && static_cast<int>(q) >= max_queries) break;

        const float *q_data = queries + q * d_q;
        const auto  &items  = expected_data.expected_vectors_per_query[q];

        for (const auto &item : items) {
            size_t gt_rank = std::get<1>(item);
            if (gt_rank == 0) continue; 

            hnswlib::labeltype omega_id =
                static_cast<hnswlib::labeltype>(std::get<0>(item));
            const std::vector<float> &omega_vec = std::get<2>(item);
            ++total_cases;

            bool unreachable_flag = false;
            try {
                
                searcher.run(q, q_data, omega_id,
                             PurePathSearcher<float>::SearchMode::BidirectionalSerial);
            } catch (const std::exception &) {
                unreachable_flag = true;
            }

            if (unreachable_flag) {
                ++unreachable_cases;
                unreachable[q].emplace_back(
                    static_cast<size_t>(omega_id), gt_rank, omega_vec);
            }
        }

        if ((q + 1) % 100 == 0) {
            std::cout << "  [collect_unreachable] processed queries: "
                      << (q + 1) << ", unreachable so far: "
                      << unreachable_cases << std::endl;
        }
    }

    std::cout << "collect_unreachable: " << unreachable_cases
              << " unreachable / " << total_cases << " total pairs" << std::endl;

    std::ofstream out(out_path.c_str());
    if (!out.is_open())
        throw std::runtime_error("Failed to open output file: " + out_path);

    out << "ef_search -- k\n";
    out << efs << " " << k << "\n";
    out << std::fixed << std::setprecision(8);

    for (size_t q = 0; q < unreachable.size(); ++q) {
        if (unreachable[q].empty()) continue;
        out << "Query " << q << "\n";
        for (const auto &entry : unreachable[q]) {
            size_t              omega_id = std::get<0>(entry);
            size_t              rank     = std::get<1>(entry);
            const std::vector<float> &vec = std::get<2>(entry);
            out << omega_id << " " << rank;
            for (float val : vec) {
                out << " " << val;
            }
            out << "\n";
        }
    }
    out.close();
    std::cout << "Unreachable expected file written to: " << out_path << std::endl;
}

}  

int main() {
    try {
        const std::string dataset_name  = get_env_or_default("DATASET_NAME", "GLOVE100");
        const std::string dataset_dir   = get_env_or_default("DATASET_DIR", "datasets/processed/");
        const std::string noise_perc    = get_env_or_default("NOISE_PERC", "0");
        const std::string index_path    = get_env_or_default("INDEX_PATH", "IndexSave/" + dataset_name + ".index");
        const std::string expected_path = get_env_or_default("EXPECTED_FILE", "");
        const std::string output_root   = get_env_or_default("OUTPUT_DIR", "ExperResults");
        const std::string mode_str      = get_env_or_default("PURE_MODE", "bi_serial");

        std::string ekind = "kind";
        {
            std::string b = expected_path.substr(expected_path.find_last_of("/\\") + 1);
            size_t dot = b.rfind(".txt"); if (dot != std::string::npos) b = b.substr(0, dot);
            size_t kp = b.find("_k");
            if (kp != std::string::npos) {
                size_t us = b.find('_', kp + 2);
                if (us != std::string::npos) ekind = b.substr(us + 1);
            }
        }
        const std::string mtag = "_M" + get_env_or_default("M", "16") + "_" + ekind;

        int max_queries = -1;
        if (const char *v = std::getenv("MAX_QUERIES")) {
            max_queries = std::atoi(v);
        }

        VectorDataLoader loader(dataset_name, TESTING, noise_perc, dataset_dir);
        loader.initializeDataMaps();

        size_t d_query = 0, n_query = 0;
        float *queries = loader.loadQueries(&d_query, &n_query);
        size_t d = d_query;

        ExpectedDataReader<float> expected_reader(expected_path);
        auto expected_data = expected_reader.read();

        std::unique_ptr<hnswlib::SpaceInterface<float>> space_ptr = whynot::create_space_from_env(d);
        hnswlib::SpaceInterface<float>& space = *space_ptr;
        const size_t M = std::stoul(get_env_or_default("M", "16"));
        const int    ef_construction = 500;
        const size_t random_seed    = 100;

        PurePathSearcher<float> searcher(&space, 16, M, ef_construction, random_seed, false);
        if (!file_exists(index_path)) {
            throw std::runtime_error("index file not found: " + index_path);
        }
        searcher.loadIndex(index_path, &space, 0);
        searcher.buildIncomingNeighbors();

        if (mode_str == "collect_unreachable") {
            run_collect_unreachable(searcher, expected_data,
                                    queries, d_query, n_query,
                                    max_queries,
                                    expected_path, output_root);
            return 0;
        }

        const PurePathSearcher<float>::SearchMode mode = parse_mode(mode_str);

        const std::string output_dir = output_root + "/pure_path_search";
        create_directory_if_not_exists(output_root);
        create_directory_if_not_exists(output_dir);

        const std::string out_csv = output_dir + "/paths_" + dataset_name + mtag + "_" + mode_str + ".csv";
        std::ofstream out(out_csv.c_str());
        if (!out.is_open()) {
            throw std::runtime_error("failed to open output file: " + out_csv);
        }

        out << "query_id,expected_id,time_ms,dist_comps,path_external_ids,run_status\n";
        iann_secondary::Runtime runtime;

        size_t total_cases   = 0;
        size_t success_cases = 0;
        double total_time_ms = 0.0;

        for (size_t q = 0; q < expected_data.expected_vectors_per_query.size(); ++q) {
            if (q >= n_query) break;
            if (max_queries > 0 && static_cast<int>(q) >= max_queries) break;

            const float *query_data = queries + q * d_query;
            const auto  &items      = expected_data.expected_vectors_per_query[q];
            for (size_t i = 0; i < items.size(); ++i) {
                if (std::get<1>(items[i]) == 0) continue;

                hnswlib::labeltype target_id =
                    static_cast<hnswlib::labeltype>(std::get<0>(items[i]));
                ++total_cases;
                runtime.begin(q,target_id);
                try {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    auto result = searcher.run(q, query_data, target_id, mode);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double elapsed_ms =
                        std::chrono::duration<double, std::milli>(t1 - t0).count();
                    bool timed_out=runtime.finish();
                    if(timed_out)elapsed_ms=runtime.budget_ms();
                    total_time_ms += elapsed_ms;
                    ++success_cases;

                    out << result.query_id << ","
                        << result.target_id << ","
                        << elapsed_ms << ","
                        << result.dist_computations << ","
                        << escape_csv(join_labels(result.path_external_ids)) << ","
                        << (timed_out?"timeout":"ok") << "\n";
                    out.flush();
                } catch (const std::exception &e) {
                    runtime.fail(e.what());
                }
            }
        }

        std::cout << "pure_path_search done. success/total="
                  << success_cases << "/" << total_cases << std::endl;
        std::cout << "average_time_ms="
                  << (success_cases > 0
                          ? total_time_ms / static_cast<double>(success_cases)
                          : 0.0)
                  << std::endl;
        std::cout << "mode=" << mode_str << std::endl;
        std::cout << "output=" << out_csv << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "pure_path_search failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
