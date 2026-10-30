#pragma once
#include "PaperRefinement.h"
namespace whynot {

// Reset search state on every call; retain only allocated storage between calls.
template<class Graph> class ColdBeam {
public:
    using Key=std::pair<float,uint32_t>;
    Graph& graph;
    std::vector<uint32_t> visited;
    uint32_t tag=0;
    ReusableHeap<Key,std::less<Key>> w;
    ReusableHeap<Key,std::greater<Key>> c;
    explicit ColdBeam(Graph& g):graph(g) {}
    std::vector<Key> run(const void* q,uint32_t ep,size_t ef) {
        if(visited.size()!=graph.size()) {visited.assign(graph.size(),0);tag=0;}
        if(++tag==0) {std::fill(visited.begin(),visited.end(),0);tag=1;}
        w.clear();c.clear(); Key p(graph.distance(q,ep),ep);
        w.push(p);c.push(p);visited[ep]=tag;
        while(!c.empty() && !(w.top()<c.top())) {
            p=c.top();c.pop();auto nb=graph.neighbors(p.second);
            for(size_t j=0;j<nb.second;++j) {
                auto id=nb.first[j];if(visited[id]==tag) continue;
                visited[id]=tag;Key v(graph.distance(q,id),id);
                if(w.size()<ef || v<w.top()) {
                    w.push(v);c.push(v);if(w.size()>ef) w.pop();
                }
            }
        }
        auto out=w.values();std::sort(out.begin(),out.end());return out;
    }
    size_t rank(const void* q,uint32_t ep,uint32_t om,size_t ef) {
        auto r=run(q,ep,ef);
        for(size_t i=0;i<r.size();++i) if(r[i].second==om) return i+1;
        return 0;
    }
};
}
