#pragma once
#include <cstddef>
#include <string>

enum query_type_t { TESTING = 2 };

float* fvecs_read(const char* path, size_t* dimension, size_t* count, size_t limit=0);
int* ivecs_read(const char* path, size_t* dimension, size_t* count);

class VectorDataLoader {
    std::string dataset_name;
    std::string directory_path;
    std::string path(const std::string& filename) const;
public:
    VectorDataLoader(std::string dataset, query_type_t type);
    VectorDataLoader(std::string dataset, query_type_t type,
                     std::string noise, std::string directory);
    void initializeDataMaps();
    float* loadDB(size_t* dimension, size_t* count);
    float* loadQueries(size_t* dimension, size_t* count);
    int* loadQueriesGroundtruths(size_t* dimension, size_t* count);
};
