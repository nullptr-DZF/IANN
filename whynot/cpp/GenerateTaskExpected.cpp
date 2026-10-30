#include "../hpp/GenerateTaskExpected.h"
#include "../hpp/SpaceFactory.h"
#include "../hpp/VectorDataLoader.h"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

static std::string setting(const char* name,const std::string& fallback="") {
    const char* value=std::getenv(name);return value?value:fallback;
}
static size_t number(const char* name,size_t fallback) {
    return std::stoull(setting(name,std::to_string(fallback)));
}

int main() try {
    const std::string key=setting("DATASET_NAME","IMAGENET");
    const std::string data=setting("DATASET_DIR","datasets/processed/");
    const std::string path=setting("INDEX_PATH","IndexSave/"+key+".index");
    const std::string outdir=setting("EXPECTED_DIR",data+key);
    const size_t M=number("M",16),efc=number("EF_CONSTRUCTION",500);
    const size_t ef=number("EF_SEARCH",500),k=number("K",100),seed=number("RANDOM_SEED",100);
    VectorDataLoader loader(key,TESTING,"0",data);loader.initializeDataMaps();
    size_t dimension,nq;
    std::unique_ptr<float[]> raw(loader.loadQueries(&dimension,&nq));
    std::vector<float> queries(raw.get(),raw.get()+dimension*nq);raw.reset();
    auto space=whynot::create_space_from_env(dimension);
    PurePathSearcher<float> index(space.get(),16,M,efc,seed,false);
    index.loadIndex(path,space.get(),0);
    if(index.M_!=M || index.label_offset_-index.offsetData_!=dimension*sizeof(float))
        throw std::runtime_error("Index parameters do not match the prepared vectors");
    auto base_labels=raw_labels_read(setting("TASK_BASE_LABELS",data+key+"/base.labels.u32"),
                                     index.cur_element_count.load());
    auto query_labels=raw_labels_read(setting("TASK_QUERY_LABELS",data+key+"/query.labels.u32"),nq);
    TaskExpectedGenerator<float> generator(index,queries,dimension,std::move(base_labels),
                                           std::move(query_labels),number("TASK_CAP",5));
    auto output=generator.generate(k,ef);
    generator.classifyReachability(output);
    generator.assign_global_ranks(output);
    const std::string prefix=outdir+"/expected_vectors_"+key+"_M"+std::to_string(M)+
        "_efC"+std::to_string(efc)+"_efS"+std::to_string(ef)+"_k"+std::to_string(k);
    auto write=[&](const std::string& suffix,int kind) {
        std::ofstream stream(prefix+suffix+".txt");
        if(!stream) throw std::runtime_error("Cannot write expected points");
        stream<<"ef_search -- k\n"<<ef<<' '<<k<<'\n'<<std::setprecision(9);
        for(size_t qi=0;qi<output.size();++qi) {
            stream<<"Query "<<qi<<'\n';
            const auto& row=output[qi];
            const std::vector<TaskExpectedGenerator<float>::Entry>* entries[]={
                &row.missed,&row.missed_unreach,&row.unvisited,&row.unvisited_unreach};
            for(const auto& entry:*entries[kind]) {
                stream<<std::get<0>(entry)<<' '<<std::get<1>(entry);
                for(float value:std::get<2>(entry)) stream<<' '<<value;
                stream<<'\n';
            }
        }
        if(!stream) throw std::runtime_error("Expected-point output failed");
    };
    write("_missed",0);write("_missed_unreachable",1);
    write("_unvisited",2);write("_unvisited_unreachable",3);
    return 0;
} catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
}
