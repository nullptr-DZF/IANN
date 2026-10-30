#pragma once
#include "hnswlib/hnswlib.h"
#include <algorithm>
#include <random>
#include <tuple>
#include <unordered_set>
#include <vector>

template<typename dist_t>
class ExpectedVectorsGenerator {
public:
    using labeltype = hnswlib::labeltype;
    using Entry = std::tuple<labeltype, size_t, std::vector<dist_t>>;
    struct GenOutputs {
        std::vector<std::vector<Entry>> top_miss;
        std::vector<std::vector<Entry>> deep;
    };

    ExpectedVectorsGenerator(hnswlib::HierarchicalNSW<dist_t>& index,
                             const std::vector<std::vector<dist_t>>& queries,
                             const std::vector<std::vector<int>>& ground_truth)
        : index_(index), queries_(queries), ground_truth_(ground_truth) {}

    GenOutputs generate(size_t k=100, bool unvisited=false) {
        size_t count=std::min(queries_.size(),ground_truth_.size());
        GenOutputs out;
        out.top_miss.resize(count); out.deep.resize(count);
        for(size_t qi=0;qi<count;++qi) {
            std::vector<labeltype> visited;
            auto result=unvisited ? index_.searchKnnWithVisited(queries_[qi].data(),k,visited)
                                  : index_.searchKnn(queries_[qi].data(),k);
            std::unordered_set<labeltype> returned,seen(visited.begin(),visited.end());
            while(!result.empty()) {returned.insert(result.top().second);result.pop();}
            const auto& row=ground_truth_[qi];
            for(size_t r=0;r<std::min(k,row.size()) && out.top_miss[qi].size()<2;++r)
                if(row[r]>=0 && !returned.count(labeltype(row[r])))
                    out.top_miss[qi].emplace_back(labeltype(row[r]),r+1,
                        index_.template getDataByLabel<dist_t>(labeltype(row[r])));
            std::vector<std::pair<labeltype,size_t>> buckets[8];
            for(size_t r=k;r<std::min(row.size(),k+900);++r) {
                if(row[r]<0) continue;
                labeltype id=labeltype(row[r]);
                if(returned.count(id) || (unvisited && seen.count(id))) continue;
                buckets[(r-k)*8/900].emplace_back(id,r+1);
            }
            std::mt19937 rng(static_cast<uint32_t>(qi+12345));
            for(auto& bucket:buckets) {
                if(bucket.empty()) continue;
                std::uniform_int_distribution<size_t> choose(0,bucket.size()-1);
                auto selected=bucket[choose(rng)];
                out.deep[qi].emplace_back(selected.first,selected.second,
                    index_.template getDataByLabel<dist_t>(selected.first));
            }
        }
        return out;
    }

private:
    hnswlib::HierarchicalNSW<dist_t>& index_;
    const std::vector<std::vector<dist_t>>& queries_;
    const std::vector<std::vector<int>>& ground_truth_;
};
