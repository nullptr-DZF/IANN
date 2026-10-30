#pragma once

#include "hnswlib/hnswlib.h"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace whynot {

inline std::string metric_from_env() {
    const char* value = std::getenv("METRIC");
    std::string metric = (value && *value) ? std::string(value) : "l2";
    if (metric != "l2" && metric != "ip") {
        throw std::runtime_error("METRIC must be 'l2' or 'ip', got: " + metric);
    }
    return metric;
}

inline std::unique_ptr<hnswlib::SpaceInterface<float>> create_space_from_env(size_t dim) {
    std::string metric = metric_from_env();
    std::cout << ">> Distance metric (METRIC env): " << metric << std::endl;
    if (metric == "ip") {
        return std::unique_ptr<hnswlib::SpaceInterface<float>>(new hnswlib::InnerProductSpace(dim));
    }
    return std::unique_ptr<hnswlib::SpaceInterface<float>>(new hnswlib::L2Space(dim));
}

} 
