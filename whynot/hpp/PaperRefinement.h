#pragma once
#include "RankIndex.h"
#include <cassert>
#include <limits>
#include <chrono>
#include <cmath>
#include <functional>
#include <queue>
#include <stdexcept>
#include <utility>

namespace whynot {
template<class T, class Compare> struct ReusableHeap : std::priority_queue<T,std::vector<T>,Compare> {
    void clear() { this->c.clear(); }
    const std::vector<T>& values() const { return this->c; }
};

struct RefinementResult {
    size_t ef=0, k=0, lower_bound=0, visited=0, expanded=0, rounds=0;
    double score=-std::numeric_limits<double>::infinity();
    double lower_bound_ms=0, reconstruction_ms=0;
    bool timed_out=false;
};

// All search-state structures use the same distance/ID key order.
template<class Graph> class PaperRefinement {
public:
    using Key=uint64_t;
private:
    Graph& graph;
    ReusableHeap<Key,std::less<Key>> beam, front;
    ReusableHeap<Key,std::greater<Key>> candidates, trans, back;
    using Dropped=std::pair<size_t,Key>;
    ReusableHeap<Dropped,std::greater<Dropped>> drop;
    RankIndex<Key> ranks;
    size_t visit_lambda=0,rank_limit=0;
    std::vector<Key> history;
    std::vector<size_t> deferred;
    ReusableHeap<Key,std::greater<Key>> bst;
    std::vector<Key> arr;
    std::vector<uint32_t> seen;
    uint32_t tag=0, omega=0;
    Key omega_key;
    float omega_distance=0;
    const void* query=nullptr;
    size_t closer=0, width=0, initial=0, initial_k=0;
    double alpha=0, beta=0;
    RefinementResult result;
    std::chrono::steady_clock::time_point started;
    double timeout_ms=600000;

    void checkTime() {
        if ((result.expanded & 1023)==0 && timeout_ms>0 &&
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()>timeout_ms)
            throw std::runtime_error("point timeout");
    }
    Key visit(uint32_t id) {
        seen[id]=tag; ++result.visited;
        Key p=distanceKey(id==omega ? omega_distance : graph.distance(query,id),id);
        history.push_back(p);
        size_t rank=ranks.insertRankBounded(p,rank_limit);
        visit_lambda=rank==SIZE_MAX ? SIZE_MAX : rank+1;
        if(p<omega_key) ++closer;
        return p;
    }
    template<class F> void expand(const Key& p, F&& accept) {
        ++result.expanded; checkTime();
        auto neighbors=graph.neighbors(uint32_t(p));
        for(size_t j=0;j<neighbors.second;++j) {
            uint32_t id=neighbors.first[j];
#ifdef USE_SSE
            if(j+1<neighbors.second) {
                uint32_t next=neighbors.first[j+1];
                _mm_prefetch(reinterpret_cast<const char*>(seen.data()+next),_MM_HINT_T0);
                graph.prefetch(next);
            }
#endif
            if(seen[id]!=tag) accept(visit(id));
        }
    }
    void record(size_t w) {
        if(seen[omega]!=tag || closer+1>w) return;
        size_t kp=std::max(initial_k,closer+1);
        if(kp>w) return;
        double score=alpha*(double(kp)-initial_k)-beta*(double(w)-initial);
        if(score>result.score+1e-10) { result.score=score; result.ef=w; result.k=kp; }
    }
    // W, Arr, BST, and Drop partition visited points; retain each rejection threshold.
    void addRejected(const Key& p) {

        if(visit_lambda==SIZE_MAX)deferred.push_back(history.size()-1);
        else drop.push(Dropped(visit_lambda,p));
    }
    // Deferred thresholds cannot affect earlier widths; recover their original ranks offline.
    void ensureRankLimit(size_t target) {
        if(target<=rank_limit)return;
        rank_limit=std::min(graph.size(),std::max(target,rank_limit+std::min(rank_limit,graph.size()-rank_limit)));

        std::vector<std::pair<Key,size_t>> sorted;sorted.reserve(history.size());
        for(size_t i=0;i<history.size();++i)sorted.emplace_back(history[i],i);
        std::sort(sorted.begin(),sorted.end(),[](const std::pair<Key,size_t>& a,const std::pair<Key,size_t>& b){return a.first<b.first;});
        std::vector<char> pending(history.size(),0);for(size_t i:deferred)pending[i]=1;
        std::vector<uint32_t> counts(history.size()+1,0);
        std::vector<Key> prefix;prefix.reserve(std::min(rank_limit,history.size()));
        for(const auto& p:sorted){
            if(pending[p.second]){size_t rank=0;for(size_t j=p.second;j;j-=j&-j)rank+=counts[j];drop.push(Dropped(rank+1,p.first));}
            for(size_t j=p.second+1;j<counts.size();j+=j&-j)++counts[j];
            if(prefix.size()<rank_limit)prefix.push_back(p.first);
        }
        deferred.clear();ranks.assignSorted(prefix);
    }
    void appendEvicted(const Key& p) {
        assert(arr.empty() || p<arr.back());
        arr.push_back(p);
    }
    void acceptTarget(const Key& p, size_t target, size_t tracked) {
        if(beam.size()<target || p<beam.top()) {
            candidates.push(p); beam.push(p);
            if(front.size()<tracked || p<front.top()) {
                front.push(p);
                if(front.size()>tracked) { back.push(front.top()); front.pop(); }
            } else back.push(p);
            if(beam.size()>target) { appendEvicted(beam.top()); beam.pop(); }
        } else addRejected(p);
    }

    // Observe completed intermediate widths without repeating search from the entry.
    void complete(size_t target, size_t tracked) {
        while(true) {
            if(candidates.empty() || beam.top()<candidates.top()) {
                record(tracked);
                while(!back.empty()) {
                    Key p=back.top(); back.pop();
                    if(beam.top()<p) continue;
                    front.push(p); ++tracked; record(tracked);
                }
                record(target); width=target; return;
            }
            Key p=candidates.top();
            while(front.empty() || front.top()<p) {
                record(tracked);
                while(!back.empty() && beam.top()<back.top()) back.pop();
                if(back.empty()) throw std::logic_error("missing intermediate beam candidate");
                front.push(back.top()); back.pop(); ++tracked;
            }
            candidates.pop();
            expand(p,[&](const Key& v){acceptTarget(v,target,tracked);});
        }
    }
    void reconstruct(size_t target) {
        ensureRankLimit(target);
        while(!drop.empty() && drop.top().first<=target) {
            Key p=drop.top().second;drop.pop();candidates.push(p);bst.push(p);
        }
        while(beam.size()<target && (!arr.empty() || !bst.empty())) {
            Key p;
            if(bst.empty() || (!arr.empty() && arr.back()<bst.top())) {
                p=arr.back(); arr.pop_back();
            } else {p=bst.top();bst.pop();}
            beam.push(p); back.push(p);
        }
    }
    // Stop after scanning omega and transfer the cached state to refinement.
    bool lowerBound(uint32_t entry) {
        Key p=visit(entry); beam.push(p); candidates.push(p);
        while(!candidates.empty()) {
            p=candidates.top(); candidates.pop();
            while(beam.top()<p) {
                ++width;
                if(trans.empty()) throw std::logic_error("empty lower-bound transfer heap");
                beam.push(trans.top()); trans.pop();
            }
            expand(p,[&](const Key& v) {
                candidates.push(v);
                if(beam.size()<width || v<beam.top()) {
                    beam.push(v);
                    if(beam.size()>width) { trans.push(beam.top()); beam.pop(); }
                } else trans.push(v);
            });
            if(uint32_t(p)==omega) {

                arr.assign(trans.values().begin(),trans.values().end());
                std::sort(arr.begin(),arr.end(),std::greater<Key>()); trans.clear();
                front=beam; result.lower_bound=width; return true;
            }
        }
        return false;
    }
public:
    explicit PaperRefinement(Graph& g):graph(g) {}
    
    std::function<void(size_t,const std::vector<Key>&)> completed_hook;
    size_t rank_capacity_hint=8192;
    RefinementResult run(const void* q, uint32_t entry, uint32_t expected,
                         size_t ef, size_t k, double a, double b,
                         size_t stride_i=5, size_t patience=5, bool use_lb=true,
                         size_t freeze_override=0, double time_limit=600000, bool bound_only=false) {
        if(!graph.size() || ef<k || ef>graph.size() || !k || !std::isfinite(a) || !std::isfinite(b) || a<0 || b<0 || std::abs(a+b-1)>1e-6)
            throw std::invalid_argument("require nonempty graph, 1<=k<=ef, alpha+beta=1");
        if(entry>=graph.size() || expected>=graph.size()) throw std::out_of_range("vertex id");
        if(seen.size()!=graph.size()) { seen.assign(graph.size(),0); tag=0; }
        if(++tag==0) { std::fill(seen.begin(),seen.end(),0); tag=1; }
        beam.clear(); front.clear(); candidates.clear(); trans.clear(); back.clear();
        drop.clear();ranks.clear();bst.clear();arr.clear();history.clear();deferred.clear();result=RefinementResult();closer=0;
        rank_limit=std::min(graph.size(),std::max(ef,rank_capacity_hint));
        query=q;omega=expected;omega_distance=graph.distance(q,expected);omega_key=distanceKey(omega_distance,expected);
        width=initial=std::min(ef,graph.size()); initial_k=k; alpha=a; beta=b;
        started=std::chrono::steady_clock::now(); timeout_ms=time_limit;
        double raw_stride=b>0 ? stride_i*a/b : double(graph.size());
        size_t stride=raw_stride>=double(graph.size()) ? graph.size() : std::max<size_t>(1,size_t(std::floor(raw_stride+1e-9)));
        size_t tolerance=freeze_override ? freeze_override :
            (patience>graph.size()/stride ? graph.size() : patience*stride);
        try {
            if(use_lb) {
                if(!lowerBound(entry)) return result;
                result.lower_bound_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
                if(bound_only) return result;
                ensureRankLimit(width);
            }
            else {
                Key p=visit(entry); beam.push(p); front.push(p); candidates.push(p);
                result.lower_bound=width;
            }
            complete(width,width);
            if(completed_hook) completed_hook(width,beam.values());
            while(width<graph.size()) {
                size_t target;
                if(a<b && result.ef) {
                    double cutoff=(b*result.ef-a*result.k)/(b-a);
                    target=cutoff>=double(graph.size()) ? graph.size() : size_t(std::max(0.0,std::floor(cutoff+1e-8)));
                } else {
                    if(result.ef && ((a==b && result.k==result.ef) || width-result.ef>=tolerance)) break;
                    target=width+std::min(stride,graph.size()-width);
                }
                if(target<=width) break;
                size_t old=width;auto reconstruct_start=std::chrono::steady_clock::now();
                reconstruct(target); ++result.rounds;
                result.reconstruction_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-reconstruct_start).count();
                complete(target,old);
                if(completed_hook) completed_hook(width,beam.values());
            }
        } catch(const std::runtime_error& e) {
            if(std::string(e.what())!="point timeout") throw;
            result.timed_out=true;
        }
        return result;
    }
};
} 
