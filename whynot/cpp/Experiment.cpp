#include "../hpp/IncrementalCalculation.h"
#include "../hpp/ColdBeam.h"
#include "../hpp/ExperimentRuntime.h"
#include <memory>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace iann_runtime;
struct Configuration {
    std::string method,index,queries,keys,output,errors,progress;
    size_t ef=500,k=100,gain=5,patience=5;
    double alpha=.99,beta=.01,budget=600000;
};
static void usage() {
    std::cout << "experiment --method iann|inc-only|lb-only|inc-rerun|beams|ground"
              << " --index FILE --queries FILE --keys FILE --output FILE --errors FILE --progress FILE"
              << " [--ef 500 --k 100 --alpha .99 --beta .01 --i 5 --ell 5 --budget-ms 600000]\n";
}
static Configuration arguments(int argc,char** argv) {
    std::map<std::string,std::string> values;
    for(int i=1;i<argc;i+=2) {
        if(i+1>=argc || std::string(argv[i]).substr(0,2)!="--") throw std::invalid_argument("expected --option value");
        if(!values.emplace(argv[i],argv[i+1]).second) throw std::invalid_argument("duplicate option");
    }
    auto get=[&](const std::string& key,const std::string& fallback="") {auto it=values.find(key);return it==values.end()?fallback:it->second;};
    const std::set<std::string> allowed={"--method","--index","--queries","--keys","--output","--errors","--progress","--ef","--k","--alpha","--beta","--i","--ell","--budget-ms"};
    for(const auto& item:values) if(!allowed.count(item.first)) throw std::invalid_argument("unknown option: "+item.first);
    Configuration c; c.method=get("--method"); c.index=get("--index"); c.queries=get("--queries"); c.keys=get("--keys");
    c.output=get("--output");c.errors=get("--errors");c.progress=get("--progress");
    if(!std::set<std::string>{"iann","inc-only","lb-only","inc-rerun","beams","ground"}.count(c.method)) throw std::invalid_argument("invalid method");
    for(const auto& path:{c.index,c.queries,c.keys,c.output,c.errors,c.progress}) if(path.empty()) throw std::invalid_argument("missing required file option");
    c.ef=std::stoull(get("--ef","500"));c.k=std::stoull(get("--k","100"));c.gain=std::stoull(get("--i","5"));c.patience=std::stoull(get("--ell","5"));
    c.alpha=std::stod(get("--alpha",".99"));c.beta=std::stod(get("--beta",".01"));c.budget=std::stod(get("--budget-ms","600000"));
    if(!c.k || c.ef<c.k || !c.gain || !c.patience || !std::isfinite(c.alpha) || !std::isfinite(c.beta) || c.alpha<0 || c.beta<0 || std::abs(c.alpha+c.beta-1)>1e-6 || !std::isfinite(c.budget) || c.budget<=0) throw std::invalid_argument("invalid numerical parameters");
    return c;
}
struct Result {
    size_t ef=0,k=0,lower=0,visited=0,expanded=0,searches=0;
    double score=-std::numeric_limits<double>::infinity();
    bool timeout=false,optimal=false;
    std::string reason;
};
static std::string row(const Configuration& c,const Point& p,const Result& r,double reported,double raw,const std::string& status) {
    std::ostringstream out;
    out<<c.method<<','<<p.query<<','<<p.expected<<','<<p.rank<<','<<r.ef<<','<<r.k<<',';
    if(std::isfinite(r.score)) out<<std::setprecision(17)<<r.score;
    out<<','<<std::setprecision(17)<<reported<<','<<raw<<','<<status<<','<<r.reason<<','<<r.lower<<','<<r.visited<<','<<r.expanded<<','<<r.searches<<','<<int(r.optimal)<<'\n';
    return out.str();
}
template<class Cold,class Graph,class Clock>
static Result cold_scan(Cold& cold,Graph& graph,const void* q,uint32_t entry,uint32_t omega,size_t start,const Configuration& c,bool exact,Clock elapsed) {
    const size_t n=graph.size(); if(!start || start>n || c.k>start) throw std::invalid_argument("invalid initial beam width");
    double raw_step=c.beta>0?c.gain*c.alpha/c.beta:double(n);
    size_t step=raw_step>=double(n)?n:std::max<size_t>(1,size_t(std::floor(raw_step+1e-9)));
    size_t tolerance=c.patience>n/step?n:c.patience*step;
    size_t exact_rank=0;
    if(exact) {
        auto expected_key=std::make_pair(graph.distance(q,omega),omega);
        for(uint32_t id=0;id<n;++id) if(std::make_pair(graph.distance(q,id),id)<expected_key) ++exact_rank;
        ++exact_rank;
    }
    Result r;
    for(size_t w=start;;++w) {
        size_t rank=cold.rank(q,entry,omega,w); ++r.searches;
        if(rank && rank<=w) {
            size_t kp=std::max(c.k,rank);
            double score=c.alpha*(double(kp)-c.k)-c.beta*(double(w)-c.ef);
            if(kp<=w && score>r.score+1e-10) {r.ef=w;r.k=kp;r.score=score;}
        }
        // A cold search completes before its elapsed-time budget is checked.
        if(elapsed()>=c.budget) {r.timeout=true;r.reason="timeout";break;}
        if(exact) {
            double upper=c.alpha*(double(std::max(c.k,exact_rank))-c.k)-c.beta*(double(w+1)-c.ef);
            if(r.ef && upper<=r.score+1e-10) {r.optimal=true;r.reason="optimal_cost_bound";break;}
        } else {
            if(r.ef && c.alpha<c.beta && (c.beta-c.alpha)*(double(w)+1)>c.beta*r.ef-c.alpha*r.k+1e-9) {r.reason="safe_cost_bound";break;}
            if(r.ef && c.alpha==c.beta && r.k==r.ef) {r.reason="equal_weight_optimum";break;}
            if(r.ef && c.alpha>=c.beta && (w-start)%step==0 && w-r.ef>=tolerance) {r.reason="no_improvement";break;}
        }
        if(w==n) {r.optimal=exact && bool(r.ef);r.reason="graph_exhausted";break;}
    }
    return r;
}
int main(int argc,char** argv) try {
    if(argc==2 && std::string(argv[1])=="--help") {usage();return 0;}
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    auto c=arguments(argc,argv);auto points=read_keys(c.keys);
    size_t dim=0;auto queries=read_queries(c.queries,dim);
    hnswlib::L2Space space(dim);
    IncrementalCalculation<float> index(&space,16);index.loadIndex(c.index,&space);index.setEf(c.ef);
    if(c.ef>index.graph.size()) throw std::invalid_argument("original beam width exceeds index size");
    whynot::ColdBeam<IncrementalCalculation<float>::Graph> cold(index.graph);
    std::unique_ptr<PurePathSearcher<float>> paths;
    if(c.method!="beams" && c.method!="ground") {
        paths.reset(new PurePathSearcher<float>(&space,16));paths->loadIndex(c.index,&space);paths->buildIncomingNeighbors();
    }
    std::ofstream output(c.output),errors(c.errors),progress(c.progress);
    if(!output || !errors || !progress) throw std::runtime_error("cannot open result files");
    output<<"method,query_id,expected_id,ground_truth_rank,ef_best,k_best,score,execution_time_ms,raw_execution_time_ms,status,stop_reason,lower_bound,visited,expanded,searches,optimal\n";output.flush();
    BudgetGuard guard(c.output,c.progress);
    size_t complete=0,exceptions=0;
    for(const auto& p:points) {
        progress<<p.query<<'\t'<<p.expected<<"\tBEGIN\n";progress.flush();
        try {
            if(p.query>=queries.size()/dim) throw std::out_of_range("query ID outside query file");
            const float* q=queries.data()+p.query*dim;
            if(c.method=="beams") {
                if(!p.ef || !p.k || p.k>p.ef || p.ef>index.graph.size()) throw std::invalid_argument("BeamS requires feasible ef_best/k_best");
                index.setEf(p.ef);
            }
            Result timeout;timeout.timeout=true;timeout.reason="whole_point_budget";
            const bool hard=c.method!="lb-only" && c.method!="inc-rerun";
            if(hard) guard.arm(c.budget,row(c,p,timeout,c.budget,c.budget,"timeout"));
            auto began=std::chrono::steady_clock::now();
            auto elapsed=[&](){return milliseconds(began);};
            Result result; double raw=0;
            if(c.method=="beams") {
                auto found=index.searchKnn(q,p.k);
                raw=elapsed();guard.cancel();
                result.ef=p.ef;result.k=p.k;result.searches=1;result.visited=found.size();result.reason="query_complete";
            } else {
                uint32_t omega=index.label_lookup_.at(p.expected),entry=index.entry(q);
                if(paths) paths->run(p.query,q,p.expected,PurePathSearcher<float>::SearchMode::BidirectionalSerial);
                if(c.method=="iann" || c.method=="inc-only") {
                    double remaining=std::max(.001,c.budget-elapsed());
                    auto r=index.refinement.run(q,entry,omega,c.ef,c.k,c.alpha,c.beta,c.gain,c.patience,c.method=="iann",0,remaining);
                    result.ef=r.ef;result.k=r.k;result.score=r.score;result.lower=r.lower_bound;result.visited=r.visited;result.expanded=r.expanded;result.timeout=r.timed_out;result.reason=r.timed_out?"timeout":"refinement_complete";
                } else {
                    size_t start=c.ef;
                    if(c.method=="lb-only") {
                        auto bound=index.refinement.run(q,entry,omega,c.ef,c.k,c.alpha,c.beta,c.gain,c.patience,true,0,0,true);
                        start=bound.lower_bound;
                        if(!start) throw std::runtime_error("lower bound did not return a starting width");
                    }
                    result=cold_scan(cold,index.graph,q,entry,omega,start,c,c.method=="ground",elapsed);
                    result.lower=c.method=="lb-only"?start:0;
                }
                raw=elapsed();guard.cancel();
            }
            if(raw>=c.budget) {result.timeout=true;result.optimal=false;result.reason="timeout";}
            std::string status=result.timeout?"timeout":result.ef?"ok":"no_feasible";
            output<<row(c,p,result,result.timeout?c.budget:raw,raw,status);output.flush();
            progress<<p.query<<'\t'<<p.expected<<"\tEND\n";progress.flush();++complete;
        } catch(const std::exception& e) {
            guard.cancel();
            errors<<"{\"method\":"<<json_quote(c.method)<<",\"query_id\":"<<p.query<<",\"expected_id\":"<<p.expected<<",\"error\":"<<json_quote(e.what())<<"}\n";errors.flush();
            progress<<p.query<<'\t'<<p.expected<<"\tEXCEPTION\n";progress.flush();++exceptions;
        }
    }
    std::cout<<"method="<<c.method<<" recorded="<<complete<<" exceptions="<<exceptions<<'\n';return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';usage();return 1;}
