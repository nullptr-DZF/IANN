#include "../hpp/VectorDataLoader.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {
template<class T>
T* read_xvecs(const char* filename, size_t* dimension, size_t* count, size_t limit) {
    static_assert(sizeof(T)==4, "xvecs values must occupy four bytes");
    std::ifstream input(filename, std::ios::binary | std::ios::ate);
    if(!input) throw std::runtime_error(std::string("cannot open ")+filename);
    const std::streamoff bytes=input.tellg();
    input.seekg(0);
    int32_t dim=0;
    input.read(reinterpret_cast<char*>(&dim),sizeof(dim));
    if(!input || dim<=0 || dim>=1000000)
        throw std::runtime_error(std::string("invalid xvecs dimension: ")+filename);
    const size_t stride=size_t(dim)+1;
    if(bytes<=0 || uint64_t(bytes)%(stride*4)!=0)
        throw std::runtime_error(std::string("truncated xvecs file: ")+filename);
    size_t n=uint64_t(bytes)/(stride*4);
    if(limit && n>limit) n=limit;
    if(n>std::numeric_limits<size_t>::max()/stride/sizeof(T))
        throw std::runtime_error("xvecs allocation size overflow");
    std::unique_ptr<T[]> values(new T[n*stride]);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(values.get()),n*stride*sizeof(T));
    if(!input) throw std::runtime_error(std::string("xvecs read failed: ")+filename);
    // Compact the dimension-prefixed rows in place to avoid a second full copy.
    for(size_t i=0;i<n;++i) {
        int32_t row_dimension=0;
        std::memcpy(&row_dimension,values.get()+i*stride,sizeof(row_dimension));
        if(row_dimension!=dim)
            throw std::runtime_error(std::string("mixed xvecs dimensions: ")+filename);
        std::memmove(values.get()+i*size_t(dim),values.get()+i*stride+1,size_t(dim)*sizeof(T));
    }
    *dimension=size_t(dim);*count=n;
    return values.release();
}
}

float* fvecs_read(const char* file,size_t* dimension,size_t* count,size_t limit) {
    return read_xvecs<float>(file,dimension,count,limit);
}
int* ivecs_read(const char* file,size_t* dimension,size_t* count) {
    return read_xvecs<int>(file,dimension,count,0);
}

VectorDataLoader::VectorDataLoader(std::string dataset,query_type_t type)
    :VectorDataLoader(std::move(dataset),type,"0","datasets/processed/") {}
VectorDataLoader::VectorDataLoader(std::string dataset,query_type_t type,
                                 std::string noise,std::string directory)
    :dataset_name(std::move(dataset)),directory_path(std::move(directory)) {
    if(type!=TESTING || (!noise.empty() && noise!="0"))
        throw std::invalid_argument("only the paper query workload is supported");
    if(directory_path.empty()) throw std::invalid_argument("empty dataset directory");
    if(directory_path.back()!='/' && directory_path.back()!='\\') directory_path+='/';
}
void VectorDataLoader::initializeDataMaps() {
    if(dataset_name!="GLOVE100" && dataset_name!="DEEP10M" && dataset_name!="MSONG420" &&
       dataset_name!="LASTFM65" && dataset_name!="IMAGENET" && dataset_name!="GLINK")
        throw std::invalid_argument("unsupported paper dataset: "+dataset_name);
}
std::string VectorDataLoader::path(const std::string& filename) const {
    return directory_path+dataset_name+"/"+filename;
}
float* VectorDataLoader::loadDB(size_t* dimension,size_t* count) {
    return fvecs_read(path("base.fvecs").c_str(),dimension,count);
}
float* VectorDataLoader::loadQueries(size_t* dimension,size_t* count) {
    return fvecs_read(path("query.fvecs").c_str(),dimension,count);
}
int* VectorDataLoader::loadQueriesGroundtruths(size_t* dimension,size_t* count) {
    return ivecs_read(path("query.groundtruth.k1000.ivecs").c_str(),dimension,count);
}
