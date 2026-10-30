#pragma once
#include "../../hnswlib/hnswlib.h"
#include "PurePathSearch.h"
#include "PaperRefinement.h"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>
#include <tuple>

inline size_t paper_env_size(const char* name,size_t fallback) {
    const char* s=std::getenv(name); return s && *s ? std::stoull(s) : fallback;
}
template<class D> class IncrementalCalculation : public hnswlib::HierarchicalNSW<D> {
    using Base=hnswlib::HierarchicalNSW<D>;
    using Id=hnswlib::tableint;
public:
    struct Graph {
        IncrementalCalculation& h;
        size_t size() const { return h.cur_element_count.load(); }
        float distance(const void* q,Id id) const {return h.fstdistfunc_(q,h.getDataByInternalId(id),h.dist_func_param_);}
        std::pair<const Id*,size_t> neighbors(Id id) const {
            auto ll=h.get_linklist0(id); return {reinterpret_cast<const Id*>(ll+1),h.getListCount(ll)};
        }
        void prefetch(Id id) const {
#ifdef USE_SSE
            _mm_prefetch(h.getDataByInternalId(id),_MM_HINT_T0);
#endif
        }
    } graph;
    whynot::PaperRefinement<Graph> refinement;
    whynot::RefinementResult last_result;
    IncrementalCalculation(hnswlib::SpaceInterface<D>* s,size_t n,size_t m=16,size_t ec=500,
                           size_t seed=100,bool replace=false)
        :Base(s,n,m,ec,seed,replace),graph{*this},refinement(graph) {}
    // Upper-layer descent fixes the layer-0 entry for this query.
    Id entry(const void* q) const {
        Id curr=this->enterpoint_node_;
        D best=graph.distance(q,curr);
        for(int level=this->maxlevel_;level>0;--level) {
            bool changed=true;
            while(changed) {
                changed=false; auto ll=this->get_linklist(curr,level);
                auto ids=reinterpret_cast<const Id*>(ll+1);
                for(size_t j=0;j<this->getListCount(ll);++j) {
                    D d=graph.distance(q,ids[j]);
                    if(d<best) {best=d;curr=ids[j];changed=true;}
                }
            }
        }
        return curr;
    }
    
    // Time provenance, lower-bound computation, and incremental refinement together.
    double SearchKnnforExpectedAlgo4(const double& a,const double& b,size_t qid,const void* q,
        const std::tuple<hnswlib::labeltype,size_t,std::vector<D>>& item,size_t k,
        std::ofstream& out,PurePathSearcher<D>& paths,double& path_ms,double& lb_ms,
        double& pre_ms,double& loop_ms,bool use_lb=true,size_t* out_ef=nullptr,
        size_t* out_k=nullptr,float* out_score=nullptr) {
        auto begin=std::chrono::steady_clock::now();
        auto label=std::get<0>(item); Id om=this->label_lookup_.at(label);
        Id ep=entry(q);
        bool reachable=true;
        try { paths.run(qid,q,label,PurePathSearcher<D>::SearchMode::BidirectionalSerial); }
        catch(const std::runtime_error& e) {
            if(std::string(e.what()).find("unreachable")==std::string::npos &&
               std::string(e.what()).find("failed to meet")==std::string::npos) throw;
            reachable=false;
        }
        auto after_path=std::chrono::steady_clock::now();
        path_ms=std::chrono::duration<double,std::milli>(after_path-begin).count();
        lb_ms=pre_ms=loop_ms=0;
        
        double da=std::getenv("ALPHA") ? std::stod(std::getenv("ALPHA")) : double(a);
        double db=std::getenv("BETA") ? std::stod(std::getenv("BETA")) : double(b);
        if(reachable) last_result=refinement.run(q,ep,om,this->ef_,k,da,db,
            paper_env_size("ALGO4_JUMP_I",5),paper_env_size("ALGO4_FREEZE_ROUNDS",5),use_lb,
            paper_env_size("ALGO4_FREEZE_EF",0),paper_env_size("MAX_POINT_TIME_MS",600000));
        else last_result=whynot::RefinementResult();
        loop_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-after_path).count();
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        out<<qid<<','<<label<<','<<std::get<1>(item)<<','<<last_result.k<<','<<last_result.ef<<','
           <<std::setprecision(12)<<last_result.score<<','<<path_ms<<','<<lb_ms<<','<<ms<<','
           <<(last_result.ef ? 1:0)<<'\n';
        if(out_ef) *out_ef=last_result.ef;
        if(out_k) *out_k=last_result.k;
        if(out_score) *out_score=last_result.score;
        return ms;
    }
};
