#include "../hpp/GenerageExpectedDatasets.h"
#include "../hpp/VectorDataLoader.h"
#include "../hpp/SpaceFactory.h"
#include "../hpp/PurePathSearch.h"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

static std::string setting(const char* name,const std::string& fallback) {
    const char* value=std::getenv(name);return value?value:fallback;
}
static size_t number(const char* name,size_t fallback) {
    return std::stoull(setting(name,std::to_string(fallback)));
}
static bool exists(const std::string& name) {return bool(std::ifstream(name));}

int main() try {
    const std::string key=setting("DATASET_NAME","LASTFM65");
    const std::string data=setting("DATASET_DIR","datasets/processed/");
    const std::string path=setting("INDEX_PATH","IndexSave/"+key+".index");
    const std::string outdir=setting("EXPECTED_DIR",data+key);
    const size_t M=number("M",16),efc=number("EF_CONSTRUCTION",500);
    const size_t ef=number("EF_SEARCH",500),k=number("K",100),seed=number("RANDOM_SEED",100);
    const bool index_only=number("INDEX_ONLY",0)!=0;
    VectorDataLoader loader(key,TESTING,"0",data);loader.initializeDataMaps();
    size_t dim,nq;
    std::unique_ptr<float[]> raw_queries(loader.loadQueries(&dim,&nq));
    auto space=whynot::create_space_from_env(dim);
    std::unique_ptr<PurePathSearcher<float>> index;
    if(exists(path)) {
        index.reset(new PurePathSearcher<float>(space.get(),16,M,efc,seed,false));
        index->loadIndex(path,space.get(),0);
        if(index->M_!=M || index->label_offset_-index->offsetData_!=dim*sizeof(float))
            throw std::runtime_error("Index parameters do not match the prepared vectors");
    } else {
        size_t dbdim,n;
        std::unique_ptr<float[]> base(loader.loadDB(&dbdim,&n));
        if(dbdim!=dim) throw std::runtime_error("Base/query dimension mismatch");
        index.reset(new PurePathSearcher<float>(space.get(),n,M,efc,seed,false));
        for(size_t i=0;i<n;++i) index->addPoint(base.get()+i*dim,i);
        index->saveIndex(path);
    }
    if(index_only) return 0;
    size_t depth,ngt;
    std::unique_ptr<int[]> raw_gt(loader.loadQueriesGroundtruths(&depth,&ngt));
    if(ngt!=nq) throw std::runtime_error("Neighbor/query count mismatch");
    const size_t use=nq;
    std::vector<std::vector<float>> queries(use,std::vector<float>(dim));
    std::vector<std::vector<int>> neighbors(use,std::vector<int>(depth));
    for(size_t i=0;i<use;++i) {
        std::copy(raw_queries.get()+i*dim,raw_queries.get()+(i+1)*dim,queries[i].begin());
        std::copy(raw_gt.get()+i*depth,raw_gt.get()+(i+1)*depth,neighbors[i].begin());
    }
    raw_queries.reset();raw_gt.reset();
    index->setEf(ef);
    const std::string mode=setting("EXPECTED_MODE","result");
    if(mode!="result" && mode!="unvisited") throw std::runtime_error("Invalid expected-point mode");
    const bool unvisited=mode=="unvisited";
    ExpectedVectorsGenerator<float> generator(*index,queries,neighbors);
    auto selected=generator.generate(k,unvisited);
    const std::string prefix=outdir+"/expected_vectors_"+key+"_M"+std::to_string(M)+
        "_efC"+std::to_string(efc)+"_efS"+std::to_string(ef)+"_k"+std::to_string(k);
    auto write=[&](const std::string& suffix,const std::vector<std::vector<ExpectedVectorsGenerator<float>::Entry>>& entries) {
        std::ofstream reachable_output(prefix+suffix+".txt");
        std::ofstream unreachable_output(prefix+suffix+"_unreachable.txt");
        if(!reachable_output || !unreachable_output) throw std::runtime_error("Cannot write expected points");
        for(auto* output:{&reachable_output,&unreachable_output})
            *output<<"ef_search -- k\n"<<ef<<' '<<k<<'\n'<<std::setprecision(9);
        for(size_t qi=0;qi<use;++qi) {
            reachable_output<<"Query "<<qi<<'\n';
            unreachable_output<<"Query "<<qi<<'\n';
            for(const auto& item:entries[qi]) {
                const bool reachable=index->isReachableFromQuery(queries[qi].data(),std::get<0>(item));
                std::ostream& output=reachable?reachable_output:unreachable_output;
                output<<std::get<0>(item)<<' '<<std::get<1>(item);
                for(float value:std::get<2>(item)) output<<' '<<value;
                output<<'\n';
            }
        }
        if(!reachable_output || !unreachable_output) throw std::runtime_error("Expected-point output failed");
    };
    if(!unvisited) write("_top100",selected.top_miss);
    write(unvisited?"_deep_unvisited":"_deep",selected.deep);
    return 0;
} catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
}
