#pragma once

#include <string>
#include <vector>
#include <tuple>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <stdexcept>

template<typename dist_t>
class ExpectedDataReader {
public:
    using labeltype = size_t;

    struct ExpectedData {
        std::map<std::string, int> params;
        std::vector<std::vector<std::tuple<labeltype, size_t, std::vector<dist_t>>>> expected_vectors_per_query;
    };

    ExpectedDataReader(const std::string& filepath) : filepath_(filepath) {}

    ExpectedData read() {
        ExpectedData data;
        std::ifstream file(filepath_);
        if (!file.is_open()) {
            throw std::runtime_error("Error: Could not open file: " + filepath_);
        }

        std::string line;

        if (!std::getline(file, line)) {
            throw std::runtime_error("Error: File is empty or header line is missing.");
        }

        if (std::getline(file, line)) {
            std::stringstream param_ss(line);
            int ef_search_val, k_val;
            if (param_ss >> ef_search_val >> k_val) {
                data.params["ef_search"] = ef_search_val;
                data.params["k"] = k_val;
            }
            else {
                throw std::runtime_error("Error: Could not parse parameter values line.");
            }
        }
        else {
            throw std::runtime_error("Error: Parameter values line is missing.");
        }

        int current_query_idx = -1;
        while (std::getline(file, line)) {
            if (line.empty()) continue;

            std::stringstream ss(line);
            std::string first_token;
            ss >> first_token;

            if (first_token == "Query") {
                ss >> current_query_idx;
                if (current_query_idx >= 0 && data.expected_vectors_per_query.size() <= current_query_idx) {
                    data.expected_vectors_per_query.resize(current_query_idx + 1);
                }
            }
            else if (current_query_idx != -1) {
                
                std::stringstream data_ss(line);

                labeltype id;
                size_t rank;
                std::vector<dist_t> vec;
                dist_t val;

                if (!(data_ss >> id >> rank)) {
                    std::cerr << "Warning: Skipping malformed data line: " << line << std::endl;
                    continue;
                }

                while (data_ss >> val) {
                    vec.push_back(val);
                }

                if (current_query_idx < data.expected_vectors_per_query.size()) {
                    data.expected_vectors_per_query[current_query_idx].emplace_back(id, rank, vec);
                }
            }
        }

        return data;
    }

private:
    std::string filepath_;
};