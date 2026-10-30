#pragma once
#include "PurePathSearch.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

inline std::vector<uint32_t> raw_labels_read(const std::string& path,size_t expected) {
    std::ifstream input(path,std::ios::binary);
    if(!input) throw std::runtime_error("Cannot read labels: "+path);
    input.seekg(0,std::ios::end);
    if(input.tellg()!=std::streamoff(expected*sizeof(uint32_t)))
        throw std::runtime_error("Label count mismatch: "+path);
    input.seekg(0);
    std::vector<uint32_t> labels(expected);
    input.read(reinterpret_cast<char*>(labels.data()),expected*sizeof(uint32_t));
    if(!input) throw std::runtime_error("Truncated labels: "+path);
    return labels;
}

template<typename dist_t>
class TaskExpectedGenerator {
public:
    using labeltype=hnswlib::labeltype;
    using tableint=hnswlib::tableint;
    using Entry=std::tuple<labeltype,size_t,std::vector<dist_t>>;
    struct QueryOut {
        std::vector<Entry> missed,missed_unreach,unvisited,unvisited_unreach;
    };

    TaskExpectedGenerator(PurePathSearcher<dist_t>& index,const std::vector<dist_t>& queries,
                          size_t dimension,std::vector<uint32_t> base_labels,
                          std::vector<uint32_t> query_labels,size_t cap)
        : index_(index),queries_(queries),dimension_(dimension),
          base_labels_(std::move(base_labels)),query_labels_(std::move(query_labels)),cap_(cap) {
        for(size_t id=0;id<base_labels_.size();++id)
            by_label_[base_labels_[id]].push_back(labeltype(id));
    }

    std::vector<QueryOut> generate(size_t k,size_t ef) {
        size_t count=query_labels_.size();
        std::vector<QueryOut> output(count);
        index_.setEf(ef);
        for(size_t qi=0;qi<count;++qi) {
            auto candidates=by_label_.find(query_labels_[qi]);
            if(candidates==by_label_.end()) continue;
            const dist_t* query=queries_.data()+qi*dimension_;
            std::vector<labeltype> visits;
            auto result=index_.searchKnnWithVisited(query,k,visits);
            std::unordered_set<labeltype> returned,visited(visits.begin(),visits.end());
            while(!result.empty()) {returned.insert(result.top().second);result.pop();}
            std::vector<std::pair<dist_t,labeltype>> missing;
            for(labeltype id:candidates->second) {
                if(returned.count(id)) continue;
                auto values=index_.template getDataByLabel<dist_t>(id);
                missing.emplace_back(distance(query,values.data()),id);
            }
            std::sort(missing.begin(),missing.end());
            if(missing.size()>cap_) missing.resize(cap_);
            // The cap is applied to the combined candidate set before visit splitting.
            for(const auto& candidate:missing) {
                Entry entry(candidate.second,1,index_.template getDataByLabel<dist_t>(candidate.second));
                (visited.count(candidate.second)?output[qi].missed:output[qi].unvisited).push_back(std::move(entry));
            }
        }
        return output;
    }

    void classifyReachability(std::vector<QueryOut>& output) {
        for(size_t qi=0;qi<output.size();++qi) {
            const dist_t* query=queries_.data()+qi*dimension_;
            split(query,output[qi].missed,output[qi].missed_unreach);
            split(query,output[qi].unvisited,output[qi].unvisited_unreach);
        }
    }

    void assign_global_ranks(std::vector<QueryOut>& output) {
        struct Target {size_t* rank;const dist_t* query;dist_t distance;tableint internal;};
        std::vector<std::vector<Target>> targets(output.size());
        for(size_t qi=0;qi<output.size();++qi) {
            const dist_t* query=queries_.data()+qi*dimension_;
            std::vector<Entry>* lists[]={&output[qi].missed,&output[qi].missed_unreach,
                                        &output[qi].unvisited,&output[qi].unvisited_unreach};
            for(auto* list:lists) for(auto& entry:*list) {
                auto found=index_.label_lookup_.find(std::get<0>(entry));
                if(found==index_.label_lookup_.end()) throw std::runtime_error("Expected ID absent from index");
                targets[qi].push_back(Target{&std::get<1>(entry),query,
                    distance(query,std::get<2>(entry).data()),found->second});
            }
        }
        std::vector<size_t> active;
        for(size_t qi=0;qi<targets.size();++qi) if(!targets[qi].empty()) active.push_back(qi);
        for(size_t first=0;first<active.size();first+=32) {
            std::vector<Target*> group;
            for(size_t j=first;j<std::min(active.size(),first+32);++j)
                for(auto& target:targets[active[j]]) group.push_back(&target);
            std::vector<uint64_t> counts(group.size(),0);
            // Compare the native distance and internal-ID order over the whole base.
            for(tableint id=0;id<index_.cur_element_count.load();++id) {
                if(index_.isMarkedDeleted(id)) continue;
                const void* value=index_.getDataByInternalId(id);
                for(size_t j=0;j<group.size();++j) {
                    dist_t d=index_.fstdistfunc_(group[j]->query,value,index_.dist_func_param_);
                    if(d<group[j]->distance || (d==group[j]->distance && id<group[j]->internal))
                        ++counts[j];
                }
            }
            for(size_t j=0;j<group.size();++j) *group[j]->rank=size_t(counts[j]+1);
        }
    }

private:
    dist_t distance(const dist_t* a,const dist_t* b) const {
        return index_.fstdistfunc_(a,b,index_.dist_func_param_);
    }
    void split(const dist_t* query,std::vector<Entry>& reachable,std::vector<Entry>& unreachable) {
        std::vector<Entry> keep;
        for(auto& entry:reachable) {
            if(index_.isReachableFromQuery(query,std::get<0>(entry))) keep.push_back(std::move(entry));
            else unreachable.push_back(std::move(entry));
        }
        reachable.swap(keep);
    }
    PurePathSearcher<dist_t>& index_;
    const std::vector<dist_t>& queries_;
    size_t dimension_;
    std::vector<uint32_t> base_labels_,query_labels_;
    size_t cap_;
    std::unordered_map<uint32_t,std::vector<labeltype>> by_label_;
};
