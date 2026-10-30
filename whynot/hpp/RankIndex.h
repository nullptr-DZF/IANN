#pragma once
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cstring>

namespace whynot {

// Pack finite float distances and internal IDs into one lexicographic key.
inline uint64_t distanceKey(float distance,uint32_t id) {
    if(distance==0)distance=0.0f;
    uint32_t bits;std::memcpy(&bits,&distance,sizeof(bits));
    bits=(bits&0x80000000u) ? ~bits : (bits^0x80000000u);
    return (uint64_t(bits)<<32)|id;
}
}

// Sorted blocks plus Fenwick counts retain exact first-visit insertion ranks.
template<class Key> class RankIndex {
    static const size_t capacity=256;
    using Block=std::vector<Key>;
    std::vector<Block> blocks,spare;
    std::vector<size_t> counts;
    size_t total=0;
    Block fresh() {
        Block b;if(!spare.empty()){b=std::move(spare.back());spare.pop_back();}
        b.clear();b.reserve(capacity+1);return b;
    }
    void countAdd(size_t i,size_t n) {for(++i;i<counts.size();i+=i&-i)counts[i]+=n;}
    void rebuild() {
        counts.assign(blocks.size()+1,0);
        for(size_t i=0;i<blocks.size();++i)counts[i+1]=blocks[i].size();
        for(size_t i=1;i<counts.size();++i){size_t parent=i+(i&-i);if(parent<counts.size())counts[parent]+=counts[i];}
    }
public:
    void clear() {for(auto& b:blocks){b.clear();spare.push_back(std::move(b));}blocks.clear();counts.clear();total=0;}
    void assignSorted(const std::vector<Key>& keys) {
        clear();
        for(size_t i=0;i<keys.size();i+=capacity/2){Block b=fresh();b.assign(keys.begin()+i,keys.begin()+std::min(keys.size(),i+capacity/2));blocks.push_back(std::move(b));}
        total=keys.size();rebuild();
    }
    size_t insertRankBounded(const Key& key,size_t limit) {
        if(total==limit && blocks.back().back()<key)return SIZE_MAX;
        size_t rank=insertRank(key);
        if(total>limit){auto& b=blocks.back();b.pop_back();--total;
            if(b.empty()){spare.push_back(std::move(b));blocks.pop_back();rebuild();}
            else countAdd(blocks.size()-1,size_t(-1));
        }
        return rank;
    }
    size_t insertRank(const Key& key) {
        if(blocks.empty()){blocks.push_back(fresh());rebuild();}
        size_t lo=0,hi=blocks.size();
        if(total)while(lo<hi){size_t mid=(lo+hi)/2;if(blocks[mid].back()<key)lo=mid+1;else hi=mid;}
        size_t i=lo==blocks.size()?lo-1:lo;
        auto& b=blocks[i];auto it=std::lower_bound(b.begin(),b.end(),key);
        size_t rank=it-b.begin();for(size_t j=i;j;j-=j&-j)rank+=counts[j];
        b.insert(it,key);++total;
        if(b.size()>capacity){Block right=fresh();right.assign(b.begin()+capacity/2,b.end());b.resize(capacity/2);blocks.insert(blocks.begin()+i+1,std::move(right));rebuild();}
        else countAdd(i,1);
        return rank;
    }
};
