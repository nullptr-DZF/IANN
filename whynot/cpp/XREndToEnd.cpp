

#include "../hpp/IndexReconstruction.h"
#include "../hpp/SpaceFactory.h"
#include "../hpp/ReadExpected.h"
#include "../hpp/VectorDataLoader.h"
#include "../hpp/SecondaryRuntime.h"
#include <iomanip>
#include <limits>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace {

bool file_exists(const std::string &p) { std::ifstream f(p.c_str()); return f.good(); }
void mkdir_p(const std::string &p) {
    if (p.empty() || file_exists(p)) return;
#if defined(_WIN32)
    _mkdir(p.c_str());
#else
    mkdir(p.c_str(), 0777);
#endif
}
std::string env_str(const char *k, const std::string &d) {
    const char *v = std::getenv(k); return (v && *v) ? std::string(v) : d;
}
size_t env_sz(const char *k, size_t d) {
    const char *v = std::getenv(k); if (!v || !*v) return d;
    try { return static_cast<size_t>(std::stoull(v)); } catch (...) { return d; }
}
whynot::Strategy parse_strategy(const std::string &s) {
    if(s!="wvd-pareto")throw std::invalid_argument("only the paper repair strategy is supported");
    return whynot::Strategy::WVD_Pareto;
}

} // namespace

template <typename dist_t>
class XREndToEndSearcher : public whynot::IndexReconstructionSearcher<dist_t> {
    using Base = whynot::IndexReconstructionSearcher<dist_t>;
    using tableint = hnswlib::tableint;
    using labeltype = hnswlib::labeltype;
    using linklistsizeint = hnswlib::linklistsizeint;

public:
    XREndToEndSearcher(hnswlib::SpaceInterface<dist_t> *s, size_t max_elements,
                       size_t M = 16, size_t ef_construction = 500,
                       size_t random_seed = 100)
        : Base(s, max_elements, M, ef_construction, random_seed, false) {}

    bool internalOf(labeltype label, tableint &out) {
        std::unique_lock<std::mutex> lock(this->label_lookup_lock);
        auto it = this->label_lookup_.find(label);
        if (it == this->label_lookup_.end()) return false;
        out = it->second;
        return true;
    }

    struct LinkSnapshot {
        tableint node;
        std::vector<tableint> nbrs;
    };

    LinkSnapshot snapshotNode(tableint internal) {
        std::unique_lock<std::mutex> lock(this->link_list_locks_[internal]);
        linklistsizeint *ll = this->get_linklist0(internal);
        size_t sz = this->getListCount(ll);
        tableint *data = reinterpret_cast<tableint *>(ll + 1);
        LinkSnapshot snap;
        snap.node = internal;
        snap.nbrs.assign(data, data + sz);
        return snap;
    }

    void restoreNode(const LinkSnapshot &snap) {
        std::unique_lock<std::mutex> lock(this->link_list_locks_[snap.node]);
        linklistsizeint *ll = this->get_linklist0(snap.node);
        tableint *data = reinterpret_cast<tableint *>(ll + 1);
        for (size_t j = 0; j < snap.nbrs.size(); ++j) data[j] = snap.nbrs[j];
        this->setListCount(ll, static_cast<unsigned short>(snap.nbrs.size()));
    }

    void rollbackRepair(const whynot::ReconstructResult &res,
                        tableint omega_internal,
                        const LinkSnapshot &omega_snap) {
        if (res.success) {
            tableint u_internal;
            if (internalOf(res.edge.bridge_u, u_internal)) {
                std::unique_lock<std::mutex> lock(this->link_list_locks_[u_internal]);
                linklistsizeint *ll = this->get_linklist0(u_internal);
                size_t sz = this->getListCount(ll);
                tableint *data = reinterpret_cast<tableint *>(ll + 1);
                if (res.pass == 1) {
                    if (sz > 0 && data[sz - 1] == omega_internal) {
                        this->setListCount(ll, static_cast<unsigned short>(sz - 1));
                    }
                } else {
                    tableint v_far_internal;
                    if (internalOf(res.edge.removed_v, v_far_internal)) {
                        for (size_t j = 0; j < sz; ++j) {
                            if (data[j] == omega_internal) { data[j] = v_far_internal; break; }
                        }
                    }
                }
            }
        }
        restoreNode(omega_snap);
    }

};

int main() try {
    const std::string dataset_name=env_str("DATASET_NAME","GLOVE100");
    const std::string dataset_dir=env_str("DATASET_DIR","datasets/processed/");
    const std::string index_path=env_str("INDEX_PATH","IndexSave/"+dataset_name+".index");
    const std::string expected_path=env_str("EXPECTED_FILE","");
    std::string output_dir=env_str("OUTPUT_DIR","ExperResults");
    const std::string strategy_str=env_str("STRATEGY","wvd-pareto");
    const size_t m=env_sz("M",16),ec=env_sz("EF_CONSTRUCTION",500),seed=env_sz("RANDOM_SEED",100);
    const size_t max_queries=env_sz("MAX_QUERIES",0);
    const double alpha=std::stod(env_str("ALPHA","0.99")),beta=std::stod(env_str("BETA","0.01"));
    const auto strategy=parse_strategy(strategy_str);
    if(expected_path.empty())throw std::invalid_argument("EXPECTED_FILE is required");
    VectorDataLoader loader(dataset_name,TESTING,"0",dataset_dir);loader.initializeDataMaps();
    size_t dim,nq;
    std::unique_ptr<float[]> queries(loader.loadQueries(&dim,&nq));
    const auto expected=ExpectedDataReader<float>(expected_path).read();
    const size_t k=expected.params.at("k"),ef=expected.params.at("ef_search");
    const size_t xr_efs=env_sz("XR_EFS",ef);
    auto space=whynot::create_space_from_env(dim);
    XREndToEndSearcher<float> searcher(space.get(),16,m,ec,seed);
    searcher.loadIndex(index_path,space.get(),0);searcher.setEf(ef);
    PurePathSearcher<float> paths(space.get(),16,m,ec,seed,false);
    paths.loadIndex(index_path,space.get(),0);paths.buildIncomingNeighbors();
    mkdir_p(output_dir);output_dir+="/xr_end_to_end";mkdir_p(output_dir);
    const std::string name=expected_path.substr(expected_path.find_last_of("/\\")+1);
    const std::string filename=output_dir+"/xr_e2e_"+name+"_wvd-pareto_M"+std::to_string(m)+".csv";
    std::ofstream out(filename);
    if(!out)throw std::runtime_error("cannot write repair/refinement results");
    out<<"query_id,omega_id,ground_truth_rank,strategy,repair_success,repair_pass,bridge_u,removed_v,repair_time_ms,pr_found,pr_rank,pr_ef,pr_score,pr_time_ms,total_time_ms,run_status\n";
    iann_secondary::Runtime runtime;
    size_t processed=0,repaired=0;
    for(size_t q=0;q<expected.expected_vectors_per_query.size();++q) {
        if(max_queries && q>=max_queries)break;
        if(q>=nq)throw std::out_of_range("expected query outside query vectors");
        const float* query=queries.get()+q*dim;
        for(const auto& item:expected.expected_vectors_per_query[q]) {
            if(!std::get<1>(item))continue;
            const size_t omega=std::get<0>(item);
            runtime.begin(q,omega);
            hnswlib::tableint internal=0;
            typename XREndToEndSearcher<float>::LinkSnapshot snapshot;
            typename PurePathSearcher<float>::OmegaSnapshot path_snapshot;
            whynot::ReconstructResult repair;
            bool have_snapshot=false,path_applied=false;
            auto restore=[&]() {
                if(path_applied) {
                    paths.rollbackRepair(repair.edge.bridge_u,repair.edge.removed_v,omega,int(repair.pass),path_snapshot);
                    path_applied=false;
                }
                if(have_snapshot) {searcher.rollbackRepair(repair,internal,snapshot);have_snapshot=false;}
            };
            try {
                if(!searcher.internalOf(omega,internal))throw std::out_of_range("expected point not in index");
                snapshot=searcher.snapshotNode(internal);have_snapshot=true;
                path_snapshot=paths.snapshotOmega(omega);
                auto found=searcher.searchWithInsertOrder(query,xr_efs,xr_efs);
                const auto repair_start=std::chrono::steady_clock::now();
                repair=searcher.reconstruct(query,omega,found,xr_efs,strategy);
                double repair_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-repair_start).count();
                size_t best_ef=0,best_k=0;
                float score=0;double pr_ms=0;
                if(repair.success) {
                    if(!paths.applyRepair(repair.edge.bridge_u,repair.edge.removed_v,omega,int(repair.pass)))
                        throw std::runtime_error("cannot synchronize repaired provenance graph");
                    path_applied=true;
                    std::ofstream discard;
                    double path_ms=0,lb_ms=0,pre_ms=0,loop_ms=0;
                    pr_ms=searcher.SearchKnnforExpectedAlgo4(alpha,beta,q,query,item,k,discard,paths,
                        path_ms,lb_ms,pre_ms,loop_ms,true,&best_ef,&best_k,&score);
                }
                // Each expected point starts from the same unmodified graph.
                restore();
                const bool timeout=runtime.finish() || (repair.success && searcher.last_result.timed_out);
                const double total=timeout?runtime.budget_ms():repair_ms+pr_ms;
                out<<q<<','<<omega<<','<<std::get<1>(item)<<",wvd-pareto,"<<repair.success<<','<<repair.pass<<','
                   <<repair.edge.bridge_u<<','<<repair.edge.removed_v<<','<<std::setprecision(12)<<repair_ms<<','
                   <<bool(best_ef)<<','<<best_k<<','<<best_ef<<',';
                if(best_ef)out<<score;
                out<<','<<pr_ms<<','<<total<<','<<(timeout?"timeout":best_ef?"ok":"no_feasible")<<'\n';out.flush();
                ++processed;repaired+=repair.success;
            } catch(const std::exception& error) {
                restore();runtime.fail(error.what());
            }
        }
    }
    std::cout<<"pairs="<<processed<<" repaired="<<repaired<<" output="<<filename<<'\n';
    return 0;
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
