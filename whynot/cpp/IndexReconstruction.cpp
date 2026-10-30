#include "../hpp/IndexReconstruction.h"
#include "../hpp/SpaceFactory.h"
#include "../hpp/ReadExpected.h"
#include "../hpp/VectorDataLoader.h"
#include "../hpp/SecondaryRuntime.h"
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_set>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

namespace {
using Key=std::pair<size_t,size_t>;
using Searcher=whynot::IndexReconstructionSearcher<float>;
struct SavedRepair {std::string status;whynot::ReconstructResult result;};
bool exists(const std::string& path) {std::ifstream f(path,std::ios::binary);return bool(f);}
void mkdirs(const std::string& path) {
    for(size_t i=1;i<=path.size();++i) if(i==path.size() || path[i]=='/' || path[i]=='\\') {
        std::string part=path.substr(0,i);
#ifdef _WIN32
        int result=_mkdir(part.c_str());
#else
        int result=mkdir(part.c_str(),0755);
#endif
        if(result && errno!=EEXIST)throw std::runtime_error("cannot create directory: "+part);
    }
}
size_t number(const char* name,size_t fallback) {return std::stoull(iann_secondary::env(name,std::to_string(fallback)));}
void set_state_path(const std::string& path) {
#ifdef _WIN32
    _putenv_s("EXPERIMENT_REPAIR_STATE",path.c_str());
#else
    setenv("EXPERIMENT_REPAIR_STATE",path.c_str(),1);
#endif
}
std::map<Key,SavedRepair> read_state(const std::string& path,const std::string& identity,bool& committed) {
    std::map<Key,SavedRepair> saved;
    std::ifstream input(path);
    if(!input)return saved;
    std::string line;
    if(!std::getline(input,line) || line!=identity)throw std::runtime_error("repair checkpoint inputs differ");
    while(std::getline(input,line)) {
        if(input.eof())break;
        std::istringstream in(line);std::string type;size_t q,e;
        if(!(in>>type))continue;
        if(type=="COMMIT") {size_t count;committed=bool(in>>count) && count==saved.size();continue;}
        if(!(in>>q>>e))continue;
        if(type=="DONE") {
            committed=false;
            SavedRepair row;int success,slot;
            if(!(in>>success>>row.result.pass>>row.result.edge.bridge_u>>row.result.edge.removed_v>>slot))continue;
            row.result.success=success!=0;row.result.edge.is_slot_fill=slot!=0;row.result.edge.omega_id=e;
            row.status=row.result.success?"ok":"no_feasible";saved[{q,e}]=row;
        } else if(type=="TIMEOUT" || type=="EXCEPTION") {
            committed=false;
            SavedRepair row;row.status=type=="TIMEOUT"?"timeout":"exception";saved[{q,e}]=row;
        }
    }
    return saved;
}
void persist(std::ofstream& state,size_t q,size_t e,const whynot::ReconstructResult& r) {
    state<<"DONE\t"<<q<<'\t'<<e<<'\t'<<r.success<<'\t'<<r.pass<<'\t'<<r.edge.bridge_u
         <<'\t'<<r.edge.removed_v<<'\t'<<r.edge.is_slot_fill<<'\n';state.flush();
}
void evaluate(Searcher& searcher,const float* queries,size_t dim,size_t nq,
              const int* gt,size_t depth,size_t k,size_t ef,
              const std::string& label,const std::string& filename,
              std::ofstream& summary,size_t total,size_t success,iann_secondary::Runtime& runtime) {
    if(!nq || !depth || !k)throw std::runtime_error("empty standard-query workload");
    searcher.setEf(ef);
    std::ofstream rows(filename);if(!rows)throw std::runtime_error("cannot write evaluation CSV");
    rows<<"query_id,recall_at_k,query_time_us,run_status\n";
    double sum_recall=0,sum_us=0;
    for(size_t q=0;q<nq;++q) {
        std::unordered_set<hnswlib::labeltype> expected;
        size_t count=std::min(depth,k);
        for(size_t j=0;j<count;++j)expected.insert(hnswlib::labeltype(gt[q*depth+j]));
        runtime.begin(q,std::numeric_limits<size_t>::max());
        const auto started=std::chrono::steady_clock::now();
        auto result=searcher.searchKnn(queries+q*dim,k);
        double elapsed=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count();
        bool timeout=runtime.finish();if(timeout)elapsed=runtime.budget_ms()*1000;
        size_t hits=0;
        while(!result.empty()) {hits+=expected.count(result.top().second);result.pop();}
        double recall=double(hits)/count;sum_recall+=recall;sum_us+=elapsed;
        rows<<q<<','<<std::setprecision(12)<<recall<<','<<elapsed<<','<<(timeout?"timeout":"ok")<<'\n';rows.flush();
    }
    summary<<label<<','<<nq<<','<<k<<','<<ef<<','<<std::setprecision(12)<<sum_recall/nq<<','
           <<sum_us/nq<<','<<sum_us/1000<<','<<total<<','<<success<<','<<(total?100.0*success/total:0)<<'\n';
    summary.flush();
}
}

int main() try {
    using iann_secondary::env;
    const std::string dataset=env("DATASET_NAME","GLOVE100");
    const std::string dataset_dir=env("DATASET_DIR","datasets/processed/");
    const std::string index_path=env("INDEX_PATH","IndexSave/"+dataset+".index");
    const std::string expected_path=env("EXPECTED_FILE");
    const std::string root=env("OUTPUT_DIR","ExperResults");
    if(env("STRATEGY","wvd-pareto")!="wvd-pareto")throw std::invalid_argument("only the paper repair strategy is supported");
    if(expected_path.empty())throw std::invalid_argument("EXPECTED_FILE is required");
    const size_t m=number("M",16),efs_q=number("EFS_Q",500),efs_omega=number("EFS_OMEGA",500);
    const size_t k=number("K",100),ef_eval=number("EF_EVAL",500),max_queries=number("MAX_QUERIES",0);
    const std::string output=root+"/index_reconstruction",eval_dir=output+"/eval";
    mkdirs(eval_dir);
    std::string tag="_M"+std::to_string(m)+"_unreach";
    const std::string state_path=env("EXPERIMENT_REPAIR_STATE",output+"/repair_state_"+dataset+tag+".tsv");
    const std::string repaired_path=env("REPAIRED_INDEX_PATH",output+"/repaired_"+dataset+tag+".index");
    set_state_path(state_path);
    const std::string identity="INPUTS\t"+index_path+"\t"+expected_path+"\t"+std::to_string(m);
    bool committed=false;
    auto saved=read_state(state_path,identity,committed);
    if(!exists(state_path)) {std::ofstream state(state_path);state<<identity<<'\n';if(!state)throw std::runtime_error("cannot initialize repair state");}
    std::ofstream state(state_path,std::ios::app);
    iann_secondary::Runtime runtime;
    VectorDataLoader loader(dataset,TESTING,"0",dataset_dir);loader.initializeDataMaps();
    size_t dim,nq,depth,ngt;
    std::unique_ptr<float[]> queries(loader.loadQueries(&dim,&nq));
    std::unique_ptr<int[]> gt(loader.loadQueriesGroundtruths(&depth,&ngt));
    if(ngt!=nq)throw std::runtime_error("ground-truth query count mismatch");
    auto expected=ExpectedDataReader<float>(expected_path).read();
    auto space=whynot::create_space_from_env(dim);
    Searcher searcher(space.get(),16,m,500,100,false);searcher.loadIndex(index_path,space.get(),0);
    std::ofstream summary(output+"/summary_"+dataset+tag+".csv");
    if(!summary)throw std::runtime_error("cannot write repair summary");
    summary<<"strategy,n_queries,k,ef_eval,avg_recall_at_k,avg_query_time_us,total_time_ms,reconstruct_total,reconstruct_success,reconstruct_success_pct\n";
    evaluate(searcher,queries.get(),dim,nq,gt.get(),depth,k,ef_eval,"baseline",eval_dir+"/eval_"+dataset+tag+"_baseline.csv",summary,0,0,runtime);
    bool complete=true;
    for(size_t q=0;q<expected.expected_vectors_per_query.size();++q) {
        if(max_queries && q>=max_queries)break;
        for(const auto& item:expected.expected_vectors_per_query[q])if(std::get<1>(item)>0)
            complete=complete && saved.count({q,std::get<0>(item)});
    }
    const bool cached=complete && committed && exists(repaired_path);
    if(cached)searcher.loadIndex(repaired_path,space.get(),0);
    std::ofstream repairs(output+"/reconstruction_"+dataset+tag+"_wvd-pareto.csv");
    repairs<<"query_id,omega_id,gt_rank,success,pass,bridge_u,removed_v,is_slot_fill,run_status\n";
    size_t total=0,success=0;
    for(size_t q=0;q<expected.expected_vectors_per_query.size();++q) {
        if(max_queries && q>=max_queries)break;
        if(q>=nq)throw std::out_of_range("expected query outside query vectors");
        const float* query=queries.get()+q*dim;
        std::vector<whynot::SearchResultWithOrder> found;
        if(!cached)found=searcher.searchWithInsertOrder(query,efs_q,k);
        for(const auto& item:expected.expected_vectors_per_query[q]) {
            if(!std::get<1>(item))continue;
            const size_t omega=std::get<0>(item);const Key key(q,omega);++total;
            auto previous=saved.find(key);SavedRepair row;
            if(previous!=saved.end())row=previous->second;
            if(!cached && (previous==saved.end() || row.result.success)) {
                runtime.begin(q,omega,previous!=saved.end());
                try {
                    auto result=searcher.reconstruct(query,omega,found,efs_omega,whynot::Strategy::WVD_Pareto);
                    bool timeout=runtime.finish();
                    if(timeout)return 124;
                    if(previous!=saved.end() && row.result.success) {
                        if(timeout || !result.success || result.pass!=row.result.pass || result.edge.bridge_u!=row.result.edge.bridge_u || result.edge.removed_v!=row.result.edge.removed_v)
                            throw std::runtime_error("repair replay differs from checkpoint");
                    } else {
                        row.result=result;row.status=timeout?"timeout":result.success?"ok":"no_feasible";
                        if(!timeout)persist(state,q,omega,result);
                        saved[key]=row;
                    }
                } catch(const std::exception& error) {
                    runtime.fail(error.what());
                    if(previous!=saved.end())throw;
                    row=SavedRepair();row.status="exception";saved[key]=row;
                }
            }
            if(row.status=="exception")continue;
            success+=row.result.success;
            repairs<<q<<','<<omega<<','<<std::get<1>(item)<<','<<row.result.success<<','<<row.result.pass<<','
                   <<row.result.edge.bridge_u<<','<<row.result.edge.removed_v<<','<<row.result.edge.is_slot_fill<<','<<row.status<<'\n';repairs.flush();
        }
    }
    if(!cached) {
        // Commit only a complete repair pass; replay reconstructs an interrupted pass.
        const std::string temporary=repaired_path+".tmp";
        searcher.saveIndex(temporary);
#ifdef _WIN32
        // Windows rename cannot replace a destination; the repair log permits replay.
        if(exists(repaired_path) && std::remove(repaired_path.c_str()))throw std::runtime_error("cannot replace repaired index");
#endif
        if(std::rename(temporary.c_str(),repaired_path.c_str()))throw std::runtime_error("cannot commit repaired index");
        state<<"COMMIT\t"<<saved.size()<<'\n';state.flush();
    }
    evaluate(searcher,queries.get(),dim,nq,gt.get(),depth,k,ef_eval,"repaired",eval_dir+"/eval_"+dataset+tag+"_repaired.csv",summary,total,success,runtime);
    std::ofstream completed(output+"/completed_"+dataset+tag+".json");
    completed<<"{\"queries\":"<<nq<<",\"repair_pairs\":"<<total<<",\"repair_success\":"<<success<<"}\n";
    std::cout<<"repair_pairs="<<total<<" repair_success="<<success<<" standard_queries="<<nq<<'\n';
    return 0;
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
