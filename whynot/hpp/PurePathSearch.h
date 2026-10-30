#pragma once

#include "hnswlib/hnswlib.h"
#include "hnswlib/visited_list_pool.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

// Reuse tagged visitation and parent buffers across serial provenance searches.
template <typename dist_t>
class PurePathSearcher : public hnswlib::HierarchicalNSW<dist_t> {
    using HierarchicalNSW = hnswlib::HierarchicalNSW<dist_t>;
    using tableint = hnswlib::tableint;
    using linklistsizeint = hnswlib::linklistsizeint;
    using labeltype = hnswlib::labeltype;
    using vl_type = hnswlib::vl_type;

public:
    enum class SearchMode {
        Single,
        BidirectionalSerial
    };

    struct PurePathResult {
        size_t query_id = 0;
        labeltype target_id = 0;
        labeltype meet_id = 0;
        std::vector<labeltype> path_external_ids;
        size_t dist_computations = 0;  
    };

    PurePathSearcher(hnswlib::SpaceInterface<dist_t> *space, size_t max_elements,
                     size_t M = 16, size_t ef_construction = 200,
                     size_t random_seed = 100, bool allow_replace_deleted = false)
                : HierarchicalNSW(space, max_elements, M, ef_construction, random_seed, allow_replace_deleted),
                    visited_array_(),
                    closed_array_() {
                resetSearchBuffers(max_elements);
        }

    PurePathResult run(size_t query_id, const void *query_data, labeltype target_id, SearchMode mode) const {
        dist_comp_counter_ = 0;   
        PurePathResult r;
        switch (mode) {
            case SearchMode::Single:
                r = runSingle(query_id, query_data, target_id);
                break;
            case SearchMode::BidirectionalSerial:
                r = runBidirectionalSerial(query_id, query_data, target_id);
                break;
            default:
                throw std::runtime_error("unsupported pure path search mode");
        }
        r.dist_computations = dist_comp_counter_;
        return r;
    }

    PurePathResult runSingle(size_t query_id, const void *query_data, labeltype target_id) const {
        const tableint target_internal = getInternalIdOrThrow(target_id);
        const tableint eq = descendToLayer0Entry(query_data);

        const size_t n = static_cast<size_t>(this->cur_element_count.load());
        ensureSearchBuffers(n);

        const vl_type search_tag = acquireVisitedTag();
        
        const vl_type ptag = acquireParentTagA();
        tableint *parent = parent_a_.get();
        vl_type *pvalid = pvalid_a_.get();

        setVisited(eq, search_tag);
        parent[eq] = eq; pvalid[eq] = ptag;
        if (eq == target_internal) {
            return buildTrivialResult(query_id, target_id, eq);
        }

        std::priority_queue<NodeDist, std::vector<NodeDist>, MinDistanceOnly> frontier;
        frontier.push(NodeDist(computeDistanceToQuery(query_data, eq), eq));

        bool found = false;
        while (!frontier.empty()) {
            NodeDist nd = frontier.top();
            frontier.pop();
            tableint u = nd.second;
            if (isClosed(u, search_tag)) {
                continue;
            }

            setClosed(u, search_tag);
            linklistsizeint *ll = this->get_linklist0(u);
            size_t sz = this->getListCount(ll);
            int *data = reinterpret_cast<int *>(ll);
            for (size_t j = 1; j <= sz; ++j) {
                tableint v = static_cast<tableint>(data[j]);
                if (isVisited(v, search_tag)) {
                    continue;
                }

                setVisited(v, search_tag);
                parent[v] = u; pvalid[v] = ptag;

                if (v == target_internal) {
                    found = true;
                    break;
                }

                frontier.push(NodeDist(computeDistanceToQuery(query_data, v), v));
            }

            if (found) {
                break;
            }
        }

        if (!found) {
            throw std::runtime_error("target is in index but unreachable from layer-0 entry");
        }

        PurePathResult out;
        out.query_id = query_id;
        out.target_id = target_id;
        out.meet_id = this->getExternalLabel(target_internal);

        std::vector<tableint> path_internal = buildPathEqToTarget(eq, target_internal, parent, pvalid, ptag);
        appendExternalLabels(path_internal, out.path_external_ids);
        return out;
    }

    PurePathResult runBidirectionalSerial(size_t query_id, const void *query_data, labeltype target_id) const {
        const tableint target_internal = getInternalIdOrThrow(target_id);
        const tableint eq = descendToLayer0Entry(query_data);
        if (eq == target_internal) {
            return buildTrivialResult(query_id, target_id, eq);
        }

        const size_t n = static_cast<size_t>(this->cur_element_count.load());
        ensureSearchBuffers(n);

        const vl_type forward_visited_tag = acquireVisitedTag();
        const vl_type backward_visited_tag = acquireVisitedTag();
        const vl_type forward_closed_tag = acquireClosedTag();
        const vl_type backward_closed_tag = acquireClosedTag();

        std::priority_queue<NodeDist, std::vector<NodeDist>, MinDistanceOnly> forward_frontier;
        std::priority_queue<NodeDist, std::vector<NodeDist>, MinDistanceOnly> backward_frontier;
        
        const vl_type fptag = acquireParentTagA();
        const vl_type bptag = acquireParentTagB();
        tableint *forward_parent = parent_a_.get();
        tableint *backward_parent = parent_b_.get();
        vl_type *fpvalid = pvalid_a_.get();
        vl_type *bpvalid = pvalid_b_.get();

        setVisited(eq, forward_visited_tag);
        forward_parent[eq] = eq; fpvalid[eq] = fptag;
        forward_frontier.push(NodeDist(computeDistanceToQuery(query_data, eq), eq));

        setVisited(target_internal, backward_visited_tag);
        backward_parent[target_internal] = target_internal; bpvalid[target_internal] = bptag;
        backward_frontier.push(NodeDist(computeDistanceToQuery(query_data, target_internal), target_internal));

        const tableint invalid = std::numeric_limits<tableint>::max();
        tableint meet = invalid;

        while (!forward_frontier.empty() && !backward_frontier.empty()) {
            tableint forward_u = std::numeric_limits<tableint>::max();
            const bool has_f = popOpenNodeSerial(forward_frontier, forward_closed_tag, forward_u);
            if (has_f) {
                tableint u = forward_u;
                setClosed(u, forward_closed_tag);

                if (isVisited(u, backward_visited_tag)) {
                    meet = u;
                    break;
                }

                linklistsizeint *ll = this->get_linklist0(u);
                size_t sz = this->getListCount(ll);
                int *data = reinterpret_cast<int *>(ll);
                for (size_t j = 1; j <= sz; ++j) {
                    tableint v = static_cast<tableint>(data[j]);
                    if (isVisited(v, forward_visited_tag)) {
                        continue;
                    }

                    setVisited(v, forward_visited_tag);
                    forward_parent[v] = u; fpvalid[v] = fptag;

                    if (isClosed(v, backward_closed_tag)) {
                        meet = v;
                        break;
                    }

                    forward_frontier.push(NodeDist(computeDistanceToQuery(query_data, v), v));
                }
                if (meet != invalid) {
                    break;
                }
            }

            tableint backward_u = std::numeric_limits<tableint>::max();
            const bool has_b = popOpenNodeSerial(backward_frontier, backward_closed_tag, backward_u);
            if (has_b) {
                tableint u = backward_u;
                setClosed(u, backward_closed_tag);

                if (isVisited(u, forward_visited_tag)) {
                    meet = u;
                    break;
                }

                const std::vector<tableint> &in_neigh = incoming_neighbors_[u];
                for (size_t j = 0; j < in_neigh.size(); ++j) {
                    tableint v = in_neigh[j];
                    if (isVisited(v, backward_visited_tag)) {
                        continue;
                    }

                    setVisited(v, backward_visited_tag);
                    backward_parent[v] = u; bpvalid[v] = bptag;

                    if (isClosed(v, forward_closed_tag)) {
                        meet = v;
                        break;
                    }

                    backward_frontier.push(NodeDist(computeDistanceToQuery(query_data, v), v));
                }
                if (meet != invalid) {
                    break;
                }
            }

            if (!has_f && !has_b) {
                break;
            }
        }

        if (meet == invalid) {
            throw std::runtime_error("bidirectional serial failed to meet");
        }

        return buildMergedResult(query_id, target_id, target_internal, eq, meet,
                                 forward_parent, fpvalid, fptag,
                                 backward_parent, bpvalid, bptag);
    }

    void buildIncomingNeighbors() const {
        const size_t n = static_cast<size_t>(this->cur_element_count.load());
        incoming_neighbors_.assign(n, std::vector<tableint>());

        for (tableint u = 0; u < static_cast<tableint>(n); ++u) {
            linklistsizeint *ll = this->get_linklist0(u);
            size_t sz = this->getListCount(ll);
            int *data = reinterpret_cast<int *>(ll);
            for (size_t j = 1; j <= sz; ++j) {
                tableint v = static_cast<tableint>(data[j]);
                incoming_neighbors_[v].push_back(u);
            }
        }
    }

    std::vector<char> computeLayer0Reachable() const {
        const size_t n = static_cast<size_t>(this->cur_element_count.load());
        std::vector<char> reachable(n, 0);
        if (n == 0) return reachable;

        std::vector<tableint> queue;
        queue.reserve(n);
        tableint start = this->enterpoint_node_;
        reachable[start] = 1;
        queue.push_back(start);

        for (size_t head = 0; head < queue.size(); ++head) {
            tableint u = queue[head];
            linklistsizeint *ll = this->get_linklist0(u);
            size_t sz = this->getListCount(ll);
            int *data = reinterpret_cast<int *>(ll);
            for (size_t j = 1; j <= sz; ++j) {
                tableint v = static_cast<tableint>(data[j]);
                if (!reachable[v]) {
                    reachable[v] = 1;
                    queue.push_back(v);
                }
            }
        }
        return reachable;
    }

    bool isLabelReachable(labeltype label, const std::vector<char>& reachable) const {
        std::unique_lock<std::mutex> lock(this->label_lookup_lock);
        typename std::unordered_map<labeltype, tableint>::const_iterator it =
            this->label_lookup_.find(label);
        if (it == this->label_lookup_.end()) {
            return false;
        }
        return it->second < reachable.size() && reachable[it->second] != 0;
    }

    bool isReachableFromQuery(const void *query_data, labeltype target_id) const {
        const size_t n = static_cast<size_t>(this->cur_element_count.load());
        if (incoming_neighbors_.size() != n) {
            buildIncomingNeighbors();
        }

        tableint target_internal;
        {
            std::unique_lock<std::mutex> lock(this->label_lookup_lock);
            typename std::unordered_map<labeltype, tableint>::const_iterator it =
                this->label_lookup_.find(target_id);
            if (it == this->label_lookup_.end()) {
                return false;
            }
            target_internal = it->second;
        }

        const tableint eq = descendToLayer0Entry(query_data);
        if (eq == target_internal) {
            return true;
        }

        ensureSearchBuffers(n);
        const vl_type forward_visited_tag = acquireVisitedTag();
        const vl_type backward_visited_tag = acquireVisitedTag();
        const vl_type forward_closed_tag = acquireClosedTag();
        const vl_type backward_closed_tag = acquireClosedTag();

        std::priority_queue<NodeDist, std::vector<NodeDist>, MinDistanceOnly> forward_frontier;
        std::priority_queue<NodeDist, std::vector<NodeDist>, MinDistanceOnly> backward_frontier;

        setVisited(eq, forward_visited_tag);
        forward_frontier.push(NodeDist(computeDistanceToQuery(query_data, eq), eq));

        setVisited(target_internal, backward_visited_tag);
        backward_frontier.push(NodeDist(computeDistanceToQuery(query_data, target_internal), target_internal));

        while (!forward_frontier.empty() && !backward_frontier.empty()) {
            tableint forward_u = std::numeric_limits<tableint>::max();
            const bool has_f = popOpenNodeSerial(forward_frontier, forward_closed_tag, forward_u);
            if (has_f) {
                tableint u = forward_u;
                setClosed(u, forward_closed_tag);

                if (isVisited(u, backward_visited_tag)) {
                    return true;
                }

                linklistsizeint *ll = this->get_linklist0(u);
                size_t sz = this->getListCount(ll);
                int *data = reinterpret_cast<int *>(ll);
                for (size_t j = 1; j <= sz; ++j) {
                    tableint v = static_cast<tableint>(data[j]);
                    if (isVisited(v, forward_visited_tag)) {
                        continue;
                    }
                    setVisited(v, forward_visited_tag);
                    if (isClosed(v, backward_closed_tag)) {
                        return true;
                    }
                    forward_frontier.push(NodeDist(computeDistanceToQuery(query_data, v), v));
                }
            }

            tableint backward_u = std::numeric_limits<tableint>::max();
            const bool has_b = popOpenNodeSerial(backward_frontier, backward_closed_tag, backward_u);
            if (has_b) {
                tableint u = backward_u;
                setClosed(u, backward_closed_tag);

                if (isVisited(u, forward_visited_tag)) {
                    return true;
                }

                const std::vector<tableint> &in_neigh = incoming_neighbors_[u];
                for (size_t j = 0; j < in_neigh.size(); ++j) {
                    tableint v = in_neigh[j];
                    if (isVisited(v, backward_visited_tag)) {
                        continue;
                    }
                    setVisited(v, backward_visited_tag);
                    if (isClosed(v, forward_closed_tag)) {
                        return true;
                    }
                    backward_frontier.push(NodeDist(computeDistanceToQuery(query_data, v), v));
                }
            }

            if (!has_f && !has_b) {
                break;
            }
        }

        return false;
    }

    struct OmegaSnapshot {
        labeltype omega = 0;
        std::vector<tableint> nbrs;
    };

    bool internalOf(labeltype label, tableint &out) {
        std::unique_lock<std::mutex> lock(this->label_lookup_lock);
        auto it = this->label_lookup_.find(label);
        if (it == this->label_lookup_.end()) return false;
        out = it->second;
        return true;
    }

    OmegaSnapshot snapshotOmega(labeltype omega_id) {
        tableint om;
        OmegaSnapshot snap;
        if (!internalOf(omega_id, om)) return snap;
        std::unique_lock<std::mutex> lock(this->link_list_locks_[om]);
        linklistsizeint *ll = this->get_linklist0(om);
        size_t sz = this->getListCount(ll);
        tableint *data = reinterpret_cast<tableint *>(ll + 1);
        snap.omega = omega_id;
        snap.nbrs.assign(data, data + sz);
        return snap;
    }

    // Apply graph edits to both outgoing adjacency and the reverse-search index.
    bool applyRepair(labeltype bridge_u, labeltype removed_v, labeltype omega_id, int pass) {
        tableint u, om;
        if (!internalOf(bridge_u, u) || !internalOf(omega_id, om)) return false;
        if (pass == 1) {
            std::unique_lock<std::mutex> lock(this->link_list_locks_[u]);
            linklistsizeint *ll = this->get_linklist0(u);
            size_t sz = this->getListCount(ll);
            size_t cap = static_cast<size_t>(this->maxM0_);
            if (sz >= cap) return false;
            tableint *data = reinterpret_cast<tableint *>(ll + 1);
            data[sz] = om;
            this->setListCount(ll, static_cast<unsigned short>(sz + 1));
            updateIncoming(u, om, true);
            return true;
        } else {
            tableint v;
            if (!internalOf(removed_v, v)) return false;
            auto om_ll = this->get_linklist0(om);
            auto om_n = this->getListCount(om_ll);
            auto om_ids = reinterpret_cast<tableint*>(om_ll + 1);
            bool route_exists = std::find(om_ids, om_ids + om_n, v) != om_ids + om_n;
            if (!route_exists && om_n >= this->maxM0_) return false;
            bool replaced = false;
            {
                std::unique_lock<std::mutex> lock(this->link_list_locks_[u]);
                linklistsizeint *ll = this->get_linklist0(u);
                size_t sz = this->getListCount(ll);
                tableint *data = reinterpret_cast<tableint *>(ll + 1);
                for (size_t j = 0; j < sz; ++j) {
                    if (data[j] == v) { data[j] = om; replaced = true; break; }
                }
            }
            if (!replaced) return false;
            updateIncoming(u, v, false);
            updateIncoming(u, om, true);
            {
                std::unique_lock<std::mutex> lock(this->link_list_locks_[om]);
                linklistsizeint *ll = this->get_linklist0(om);
                size_t sz = this->getListCount(ll);
                size_t cap = static_cast<size_t>(this->maxM0_);
                if (!route_exists && sz < cap) {
                    tableint *data = reinterpret_cast<tableint *>(ll + 1);
                    data[sz] = v;
                    this->setListCount(ll, static_cast<unsigned short>(sz + 1));
                    updateIncoming(om, v, true);
                }
            }
            return true;
        }
    }

    // Restore outgoing and reverse adjacency before the next independent point.
    void rollbackRepair(labeltype bridge_u, labeltype removed_v, labeltype omega_id,
                        int pass, const OmegaSnapshot &snap) {
        tableint u, om;
        if (internalOf(bridge_u, u) && internalOf(omega_id, om)) {
            if (pass == 1) {
                std::unique_lock<std::mutex> lock(this->link_list_locks_[u]);
                linklistsizeint *ll = this->get_linklist0(u);
                size_t sz = this->getListCount(ll);
                tableint *data = reinterpret_cast<tableint *>(ll + 1);
                if (sz > 0 && data[sz - 1] == om) {
                    this->setListCount(ll, static_cast<unsigned short>(sz - 1));
                    updateIncoming(u, om, false);
                }
            } else {
                tableint v;
                if (internalOf(removed_v, v)) {
                    std::unique_lock<std::mutex> lock(this->link_list_locks_[u]);
                    linklistsizeint *ll = this->get_linklist0(u);
                    size_t sz = this->getListCount(ll);
                    tableint *data = reinterpret_cast<tableint *>(ll + 1);
                    for (size_t j = 0; j < sz; ++j) {
                        if (data[j] == om) {
                            data[j] = v;
                            updateIncoming(u, om, false);
                            updateIncoming(u, v, true);
                            break;
                        }
                    }
                }
            }
        }
        if (internalOf(omega_id, om)) {
            std::unique_lock<std::mutex> lock(this->link_list_locks_[om]);
            linklistsizeint *ll = this->get_linklist0(om);
            tableint *data = reinterpret_cast<tableint *>(ll + 1);
            for (size_t j = 0; j < this->getListCount(ll); ++j)
                if (std::find(snap.nbrs.begin(), snap.nbrs.end(), data[j]) == snap.nbrs.end())
                    updateIncoming(om, data[j], false);
            for (size_t j = 0; j < snap.nbrs.size(); ++j) data[j] = snap.nbrs[j];
            this->setListCount(ll, static_cast<unsigned short>(snap.nbrs.size()));
        }
    }

private:
    void updateIncoming(tableint from, tableint to, bool add) const {
        if (incoming_neighbors_.size() != this->cur_element_count.load()) return;
        auto& v = incoming_neighbors_[to];
        auto it = std::lower_bound(v.begin(), v.end(), from);
        if (add) { if (it == v.end() || *it != from) v.insert(it, from); }
        else if (it != v.end() && *it == from) v.erase(it);
    }
    using NodeDist = std::pair<dist_t, tableint>;

    struct MinDistanceOnly {
        bool operator()(const NodeDist &a, const NodeDist &b) const {
            return a > b;
        }
    };

    mutable std::unique_ptr<vl_type[]> visited_array_;
    mutable std::unique_ptr<vl_type[]> closed_array_;
    mutable size_t search_buffer_capacity_ = 0;
    mutable vl_type visited_array_tag_ = 0;
    mutable vl_type closed_array_tag_ = 0;
    mutable std::mutex search_buffer_lock_;
    mutable std::vector<std::vector<tableint> > incoming_neighbors_;
    mutable size_t dist_comp_counter_ = 0;  

    mutable std::unique_ptr<tableint[]> parent_a_;
    mutable std::unique_ptr<tableint[]> parent_b_;
    mutable std::unique_ptr<vl_type[]> pvalid_a_;
    mutable std::unique_ptr<vl_type[]> pvalid_b_;
    mutable vl_type pvalid_tag_a_ = 0;
    mutable vl_type pvalid_tag_b_ = 0;

    tableint getInternalIdOrThrow(labeltype label) const {
        std::unique_lock<std::mutex> lock(this->label_lookup_lock);
        typename std::unordered_map<labeltype, tableint>::const_iterator it = this->label_lookup_.find(label);
        if (it == this->label_lookup_.end()) {
            throw std::runtime_error("label not found");
        }
        return it->second;
    }

    tableint descendToLayer0Entry(const void *query_data) const {
        if (this->cur_element_count.load() == 0) {
            throw std::runtime_error("index is empty");
        }

        tableint curr_obj = this->enterpoint_node_;
        dist_t cur_dist = computeDistanceToQuery(query_data, curr_obj);

        for (int level = this->maxlevel_; level > 0; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data = reinterpret_cast<unsigned int *>(this->get_linklist(curr_obj, level));
                int size = this->getListCount(data);
                tableint *datal = reinterpret_cast<tableint *>(data + 1);

                for (int i = 0; i < size; ++i) {
                    tableint cand = datal[i];
                    dist_t d = computeDistanceToQuery(query_data, cand);
                    if (d < cur_dist) {
                        cur_dist = d;
                        curr_obj = cand;
                        changed = true;
                    }
                }
            }
        }

        return curr_obj;
    }

    dist_t computeDistanceToQuery(const void *query_data, tableint node) const {
        ++dist_comp_counter_;
        return this->fstdistfunc_(query_data, this->getDataByInternalId(node), this->dist_func_param_);
    }

    void ensureSearchBuffers(size_t n) const {
        if (search_buffer_capacity_ >= n) {
            return;
        }

        std::lock_guard<std::mutex> lock(search_buffer_lock_);
        if (search_buffer_capacity_ >= n) {
            return;
        }

        std::unique_ptr<vl_type[]> new_visited(new vl_type[n]);
        std::unique_ptr<vl_type[]> new_closed(new vl_type[n]);
        for (size_t i = 0; i < n; ++i) {
            new_visited[i] = 0;
            new_closed[i] = 0;
        }
        
        std::unique_ptr<tableint[]> new_parent_a(new tableint[n]);
        std::unique_ptr<tableint[]> new_parent_b(new tableint[n]);
        std::unique_ptr<vl_type[]> new_pvalid_a(new vl_type[n]);
        std::unique_ptr<vl_type[]> new_pvalid_b(new vl_type[n]);
        for (size_t i = 0; i < n; ++i) {
            new_pvalid_a[i] = 0;
            new_pvalid_b[i] = 0;
        }

        visited_array_ = std::move(new_visited);
        closed_array_ = std::move(new_closed);
        parent_a_ = std::move(new_parent_a);
        parent_b_ = std::move(new_parent_b);
        pvalid_a_ = std::move(new_pvalid_a);
        pvalid_b_ = std::move(new_pvalid_b);
        search_buffer_capacity_ = n;
    }

    inline vl_type acquireVisitedTag() const {
        return acquireTag(visited_array_.get(), search_buffer_capacity_, visited_array_tag_);
    }

    inline vl_type acquireClosedTag() const {
        return acquireTag(closed_array_.get(), search_buffer_capacity_, closed_array_tag_);
    }

    inline vl_type acquireParentTagA() const {
        return acquireTag(pvalid_a_.get(), search_buffer_capacity_, pvalid_tag_a_);
    }

    inline vl_type acquireParentTagB() const {
        return acquireTag(pvalid_b_.get(), search_buffer_capacity_, pvalid_tag_b_);
    }

    inline void resetSearchBuffers(size_t n) const {
        ensureSearchBuffers(n);
        for (size_t i = 0; i < search_buffer_capacity_; ++i) {
            visited_array_[i] = 0;
            closed_array_[i] = 0;
        }
        visited_array_tag_ = 0;
        closed_array_tag_ = 0;
    }

    inline vl_type acquireTag(vl_type *array, size_t capacity, vl_type &tag) const {
        ++tag;
        if (tag == 0) {
            std::lock_guard<std::mutex> lock(search_buffer_lock_);
            for (size_t i = 0; i < capacity; ++i) {
                array[i] = 0;
            }
            ++tag;
        }
        return tag;
    }

    inline bool isVisited(tableint node, vl_type tag) const {
        return visited_array_[node] == tag;
    }

    inline void setVisited(tableint node, vl_type tag) const {
        visited_array_[node] = tag;
    }

    inline bool isClosed(tableint node, vl_type tag) const {
        return closed_array_[node] == tag;
    }

    inline void setClosed(tableint node, vl_type tag) const {
        closed_array_[node] = tag;
    }

    std::vector<tableint> buildPathEqToTarget(tableint eq, tableint target,
                                              const tableint *parent,
                                              const vl_type *pvalid, vl_type ptag) const {
        std::vector<tableint> reverse_path;

        tableint cur = target;
        while (pvalid[cur] == ptag) {
            reverse_path.push_back(cur);
            if (cur == eq) {
                break;
            }
            cur = parent[cur];
        }

        if (reverse_path.empty() || reverse_path.back() != eq) {
            throw std::runtime_error("target path reconstruction failed");
        }

        return std::vector<tableint>(reverse_path.rbegin(), reverse_path.rend());
    }

    std::vector<tableint> buildPathNodeToRoot(tableint node, tableint root,
                                              const tableint *parent,
                                              const vl_type *pvalid, vl_type ptag) const {
        std::vector<tableint> path;

        tableint cur = node;
        while (pvalid[cur] == ptag) {
            path.push_back(cur);
            if (cur == root) {
                break;
            }
            cur = parent[cur];
        }

        if (path.empty() || path.back() != root) {
            throw std::runtime_error("root path reconstruction failed");
        }
        return path;
    }

    void appendExternalLabels(const std::vector<tableint> &path_internal,
                              std::vector<labeltype> &out) const {
        out.reserve(path_internal.size());
        for (size_t i = 0; i < path_internal.size(); ++i) {
            out.push_back(this->getExternalLabel(path_internal[i]));
        }
    }

    PurePathResult buildTrivialResult(size_t query_id, labeltype target_id, tableint meet) const {
        PurePathResult out;
        out.query_id = query_id;
        out.target_id = target_id;
        out.meet_id = this->getExternalLabel(meet);
        out.path_external_ids.push_back(this->getExternalLabel(meet));
        return out;
    }

    PurePathResult buildMergedResult(size_t query_id, labeltype target_id, tableint target_internal,
                                     tableint eq, tableint meet,
                                     const tableint *forward_parent, const vl_type *fpvalid, vl_type fptag,
                                     const tableint *backward_parent, const vl_type *bpvalid, vl_type bptag) const {
        std::vector<tableint> f_meet_to_eq = buildPathNodeToRoot(meet, eq, forward_parent, fpvalid, fptag);
        std::vector<tableint> b_meet_to_w = buildPathNodeToRoot(meet, target_internal, backward_parent, bpvalid, bptag);

        std::vector<tableint> forward_path(f_meet_to_eq.rbegin(), f_meet_to_eq.rend());
        std::vector<tableint> backward_path = b_meet_to_w;

        std::vector<tableint> merged = forward_path;
        for (size_t i = 1; i < backward_path.size(); ++i) {
            merged.push_back(backward_path[i]);
        }

        PurePathResult out;
        out.query_id = query_id;
        out.target_id = target_id;
        out.meet_id = this->getExternalLabel(meet);
        appendExternalLabels(merged, out.path_external_ids);
        return out;
    }

    inline bool popOpenNodeSerial(std::priority_queue<NodeDist, std::vector<NodeDist>, MinDistanceOnly> &frontier,
                           vl_type closed_tag,
                           tableint &out_node) const {
        while (!frontier.empty()) {
            out_node = frontier.top().second;
            frontier.pop();
            return true;
        }
        return false;
    }

};
