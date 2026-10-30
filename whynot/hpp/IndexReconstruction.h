#pragma once

#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/hnswalg.h"
#include "IncrementalCalculation.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace whynot {

struct SearchResultWithOrder {
    hnswlib::labeltype vertex_id;
    float              distance;
    size_t             insert_order; 
};

struct EdgeInfo {
    hnswlib::labeltype bridge_u;   
    hnswlib::labeltype omega_id;
    hnswlib::labeltype removed_v;  
    bool               is_slot_fill; 
};

struct ReconstructResult {
    bool     success;
    size_t   pass;   
    EdgeInfo edge;

    ReconstructResult() : success(false), pass(0), edge(EdgeInfo{0, 0, 0, false}) {}
    ReconstructResult(bool s, size_t p, EdgeInfo e) : success(s), pass(p), edge(e) {}
};

enum class Strategy {
    Base,       
    WV_Linear,  
    WV_Pareto,  
    DD,         
    WVD_Linear, 
    WVD_Pareto  
};

// The paper strategy prioritizes Pareto candidates and preserves displaced routes.
template <typename dist_t>
class IndexReconstructionSearcher : public IncrementalCalculation<dist_t> {
    using Base            = IncrementalCalculation<dist_t>;
    using tableint        = hnswlib::tableint;
    using linklistsizeint = hnswlib::linklistsizeint;
    using labeltype       = hnswlib::labeltype;

public:
    IndexReconstructionSearcher(hnswlib::SpaceInterface<dist_t> *space,
                                size_t max_elements,
                                size_t M               = 16,
                                size_t ef_construction = 200,
                                size_t random_seed     = 100,
                                bool   allow_replace_deleted = false)
        : Base(space, max_elements, M, ef_construction, random_seed,
               allow_replace_deleted) {}

    std::vector<SearchResultWithOrder>
    searchWithInsertOrder(const void *query_data, size_t ef, size_t k) const {
        
        tableint curr = this->enterpoint_node_;
        dist_t   cur_dist =
            this->fstdistfunc_(query_data, this->getDataByInternalId(curr),
                               this->dist_func_param_);
        for (int level = this->maxlevel_; level > 0; --level) {
            bool changed = true;
            while (changed) {
                changed              = false;
                unsigned int *data   = reinterpret_cast<unsigned int *>(
                    this->get_linklist(curr, level));
                int          size    = this->getListCount(data);
                tableint    *datal   = reinterpret_cast<tableint *>(data + 1);
                for (int i = 0; i < size; ++i) {
                    tableint cand = datal[i];
                    dist_t   d =
                        this->fstdistfunc_(query_data,
                                           this->getDataByInternalId(cand),
                                           this->dist_func_param_);
                    if (d < cur_dist) {
                        cur_dist = d;
                        curr     = cand;
                        changed  = true;
                    }
                }
            }
        }
        return searchBaseLayer0WithInsertOrder(curr, query_data, ef, k);
    }

    ReconstructResult reconstruct(
        const void                            *q_data,
        labeltype                              omega_id,
        const std::vector<SearchResultWithOrder> &S_q,
        size_t                                 efs        = 100,
        Strategy                               strategy   = Strategy::WVD_Pareto,
        float alpha  = 0.2f, float beta_  = 0.2f, float gamma_ = 0.2f,
        float delta_ = 0.2f, float zeta_  = 0.2f,
        float eta    = 0.5f,
        const std::string &direction_mode  = "cosine")
    {
        tableint omega_internal = getInternalIdOrThrow(omega_id);
        const void *omega_data  = this->getDataByInternalId(omega_internal);

        std::vector<SearchResultWithOrder> S_omega =
            searchBaseLayer0WithInsertOrder(omega_internal, omega_data, efs, S_q.size());

        std::vector<CandidateEntry> L = buildCandidateList(
            q_data, omega_data, omega_id, omega_internal,
            S_q, S_omega,
            strategy,
            alpha, beta_, gamma_, delta_, zeta_);

        if (L.empty()) {
            return ReconstructResult{false, 0, EdgeInfo{}};
        }

        for (const CandidateEntry &ce : L) {
            tableint u_internal = getInternalIdOrThrow(ce.vertex_id);
            std::unique_lock<std::mutex> lock(this->link_list_locks_[u_internal]);
            linklistsizeint *ll = this->get_linklist0(u_internal);
            size_t           sz = this->getListCount(ll);
            size_t           cap = static_cast<size_t>(this->maxM0_);
            if (sz < cap) {
                tableint *data = reinterpret_cast<tableint *>(ll + 1);
                data[sz]       = omega_internal;
                this->setListCount(ll, static_cast<unsigned short>(sz + 1));
                EdgeInfo ei;
                ei.bridge_u    = ce.vertex_id;
                ei.omega_id    = omega_id;
                ei.removed_v   = 0;
                ei.is_slot_fill = true;
                return ReconstructResult{true, 1, ei};
            }
        }

        for (const CandidateEntry &ce : L) {
            tableint u_internal = getInternalIdOrThrow(ce.vertex_id);
            std::unique_lock<std::mutex> lock(this->link_list_locks_[u_internal]);
            linklistsizeint *ll  = this->get_linklist0(u_internal);
            size_t           sz  = this->getListCount(ll);
            tableint        *nbrs = reinterpret_cast<tableint *>(ll + 1);

            std::vector<tableint> C(nbrs, nbrs + sz);

            tableint v_far =
                selectVfar(u_internal, C, omega_internal, strategy, eta,
                           direction_mode);

            // Check route-preservation capacity before mutating either adjacency list.
            auto omega_ll = this->get_linklist0(omega_internal);
            auto omega_n = this->getListCount(omega_ll);
            auto omega_ids = reinterpret_cast<tableint*>(omega_ll + 1);
            bool route_exists = std::find(omega_ids, omega_ids + omega_n, v_far) != omega_ids + omega_n;
            if (!route_exists && omega_n >= this->maxM0_)
                throw std::runtime_error("repair requires an extra omega adjacency slot; index left unchanged");

            for (size_t j = 0; j < sz; ++j) {
                if (nbrs[j] == v_far) {
                    nbrs[j] = omega_internal;
                    break;
                }
            }

            {
                std::unique_lock<std::mutex> lock_omega(
                    this->link_list_locks_[omega_internal]);
                linklistsizeint *ll_omega = this->get_linklist0(omega_internal);
                size_t           sz_omega = this->getListCount(ll_omega);
                size_t           cap_omega = static_cast<size_t>(this->maxM0_);
                if (!route_exists && sz_omega < cap_omega) {
                    tableint *data_omega =
                        reinterpret_cast<tableint *>(ll_omega + 1);
                    data_omega[sz_omega] = v_far;
                    this->setListCount(ll_omega,
                                       static_cast<unsigned short>(sz_omega + 1));
                }
                
            }

            EdgeInfo ei;
            ei.bridge_u     = ce.vertex_id;
            ei.omega_id     = omega_id;
            ei.removed_v    = this->getExternalLabel(v_far);
            ei.is_slot_fill = false;
            return ReconstructResult{true, 2, ei};
        }

        return ReconstructResult{false, 0, EdgeInfo{}};
    }

private:

    struct CandidateEntry {
        labeltype vertex_id;
        float     score;         
        size_t    insert_order;  
    };

    std::vector<SearchResultWithOrder>
    searchBaseLayer0WithInsertOrder(tableint ep_id, const void *data_point,
                                    size_t ef, size_t k) const {
        hnswlib::VisitedList *vl =
            this->visited_list_pool_->getFreeVisitedList();
        hnswlib::vl_type *visited_array    = vl->mass;
        hnswlib::vl_type  visited_array_tag = vl->curV;

        using Pair = std::pair<dist_t, tableint>;
        struct MaxCmp {
            bool operator()(const Pair &a, const Pair &b) const {
                return a < b;
            }
        };
        
        struct MinCmp {
            bool operator()(const Pair &a, const Pair &b) const {
                return a > b;
            }
        };
        std::priority_queue<Pair, std::vector<Pair>, MaxCmp> top_candidates;
        std::priority_queue<Pair, std::vector<Pair>, MinCmp> candidate_set;

        std::unordered_map<tableint, size_t> w_insert_order;
        size_t order_counter = 0;

        dist_t lowerBound;
        if (!this->isMarkedDeleted(ep_id)) {
            dist_t dist =
                this->fstdistfunc_(data_point, this->getDataByInternalId(ep_id),
                                   this->dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(dist, ep_id);
            w_insert_order[ep_id] = ++order_counter;
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            Pair curr_pair = candidate_set.top();
            if (!top_candidates.empty() && top_candidates.top() < curr_pair && top_candidates.size() == ef) {
                break;
            }
            candidate_set.pop();

            tableint curr_id = curr_pair.second;
            linklistsizeint *ll =
                reinterpret_cast<linklistsizeint *>(this->get_linklist0(curr_id));
            size_t    sz   = this->getListCount(ll);
            int      *data = reinterpret_cast<int *>(ll);

            for (size_t j = 1; j <= sz; ++j) {
                tableint cand = static_cast<tableint>(data[j]);
                if (visited_array[cand] == visited_array_tag) continue;
                visited_array[cand] = visited_array_tag;
                w_insert_order[cand] = ++order_counter;

                dist_t dist =
                    this->fstdistfunc_(data_point,
                                       this->getDataByInternalId(cand),
                                       this->dist_func_param_);
                if (top_candidates.size() < ef || Pair(dist, cand) < top_candidates.top()) {
                    candidate_set.emplace(dist, cand);
                    if (!this->isMarkedDeleted(cand)) {
                        top_candidates.emplace(dist, cand);
                        if (top_candidates.size() > ef) {

                            top_candidates.pop();
                        }
                        if (!top_candidates.empty()) {
                            lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }
        }

        this->visited_list_pool_->releaseVisitedList(vl);

        std::vector<SearchResultWithOrder> result;
        result.reserve(top_candidates.size());
        while (!top_candidates.empty()) {
            Pair p = top_candidates.top();
            top_candidates.pop();
            labeltype ext = this->getExternalLabel(p.second);
            size_t    ord = 0;
            auto      it  = w_insert_order.find(p.second);
            if (it != w_insert_order.end()) ord = it->second;
            result.push_back(SearchResultWithOrder{ext, static_cast<float>(p.first), ord});
        }
        
        std::reverse(result.begin(), result.end());

        if (result.size() > k) result.resize(k);
        return result;
    }

    std::vector<CandidateEntry> buildCandidateList(
        const void                            *q_data,
        const void                            *omega_data,
        labeltype                              omega_id,
        tableint                               omega_internal,
        const std::vector<SearchResultWithOrder> &S_q,
        const std::vector<SearchResultWithOrder> &S_omega,
        Strategy                               strategy,
        float alpha, float beta_, float gamma_, float delta_, float zeta_) const
    {
        
        std::unordered_set<labeltype> set_q, set_omega;
        std::unordered_map<labeltype, size_t> order_q, order_omega;
        for (const auto &r : S_q) {
            set_q.insert(r.vertex_id);
            order_q[r.vertex_id] = r.insert_order;
        }
        for (const auto &r : S_omega) {
            set_omega.insert(r.vertex_id);
            order_omega[r.vertex_id] = r.insert_order;
        }

        std::vector<labeltype> U;
        for (const auto &r : S_q) {
            if (set_omega.count(r.vertex_id) && r.vertex_id != omega_id) {
                U.push_back(r.vertex_id);
            }
        }

        bool use_U = !U.empty();

        std::vector<labeltype> working;
        bool use_order_q = true; 
        if (use_U) {
            working = U;
        } else {
            for (const auto &r : S_q) {
                if (r.vertex_id != omega_id) {
                    working.push_back(r.vertex_id);
                }
            }
        }

        if (working.empty()) return {};

        bool use_wv = (strategy == Strategy::WV_Linear  ||
                       strategy == Strategy::WV_Pareto   ||
                       strategy == Strategy::WVD_Linear  ||
                       strategy == Strategy::WVD_Pareto);
        bool use_pareto = (strategy == Strategy::WV_Pareto ||
                           strategy == Strategy::WVD_Pareto);

        if (!use_wv) {
            
            std::vector<CandidateEntry> L;
            L.reserve(working.size());
            for (labeltype vid : working) {
                size_t ord = use_order_q ? order_q[vid] : order_omega[vid];
                L.push_back(CandidateEntry{vid, 0.0f, ord});
            }
            std::sort(L.begin(), L.end(),
                      [](const CandidateEntry &a, const CandidateEntry &b) {
                          return a.insert_order < b.insert_order;
                      });
            return L;
        }

        std::vector<float> dist_to_q, dist_to_omega;
        dist_to_q.reserve(working.size());
        dist_to_omega.reserve(working.size());
        for (labeltype vid : working) {
            tableint iid = getInternalIdOrThrow(vid);
            const void *vd = this->getDataByInternalId(iid);
            dist_to_q.push_back(static_cast<float>(
                this->fstdistfunc_(q_data, vd, this->dist_func_param_)));
            dist_to_omega.push_back(static_cast<float>(
                this->fstdistfunc_(omega_data, vd, this->dist_func_param_)));
        }

        float max_dq = *std::max_element(dist_to_q.begin(), dist_to_q.end());
        float min_dq = *std::min_element(dist_to_q.begin(), dist_to_q.end());
        float max_do = *std::max_element(dist_to_omega.begin(), dist_to_omega.end());
        float min_do = *std::min_element(dist_to_omega.begin(), dist_to_omega.end());

        float range_dq = max_dq - min_dq;
        float range_do = max_do - min_do;

        size_t S_ref_size = use_order_q ? S_q.size() : S_omega.size();

        std::vector<CandidateEntry> L;
        L.reserve(working.size());

        for (size_t i = 0; i < working.size(); ++i) {
            labeltype vid = working[i];
            tableint  iid = getInternalIdOrThrow(vid);

            float W_A = (range_dq > 0.0f)
                            ? (max_dq - dist_to_q[i]) / range_dq
                            : 1.0f;

            linklistsizeint *ll_u = this->get_linklist0(iid);
            size_t           sz_u = this->getListCount(ll_u);
            float            W_B  = static_cast<float>(this->maxM0_ - sz_u) /
                                    static_cast<float>(this->maxM0_);

            float W_C = (range_do > 0.0f)
                            ? (max_do - dist_to_omega[i]) / range_do
                            : 1.0f;

            size_t ord = use_order_q ? order_q[vid] : order_omega[vid];
            float W_D = -static_cast<float>(ord); 

            float W_E = computeNeighborhoodQuality(iid);

            float score = alpha  * W_A + beta_  * W_B + gamma_ * W_C +
                          delta_ * W_D + zeta_  * W_E;

            L.push_back(CandidateEntry{vid, score, ord});
        }

        if (!use_pareto) {
            
            std::sort(L.begin(), L.end(),
                      [](const CandidateEntry &a, const CandidateEntry &b) {
                          return a.score > b.score;
                      });
            return L;
        }

        struct FiveScores {
            float W_A, W_B, W_C, W_D, W_E;
        };

        std::vector<FiveScores> scores5(L.size());
        for (size_t i = 0; i < working.size(); ++i) {
            labeltype vid = working[i];
            tableint  iid = getInternalIdOrThrow(vid);

            float WA = (range_dq > 0.0f)
                           ? (max_dq - dist_to_q[i]) / range_dq
                           : 1.0f;
            linklistsizeint *ll_u = this->get_linklist0(iid);
            size_t           sz_u = this->getListCount(ll_u);
            float            WB   = static_cast<float>(this->maxM0_ - sz_u) /
                                    static_cast<float>(this->maxM0_);
            float WC = (range_do > 0.0f)
                           ? (max_do - dist_to_omega[i]) / range_do
                           : 1.0f;
            size_t ord = use_order_q ? order_q[vid] : order_omega[vid];
            float WD = -static_cast<float>(ord); 
            float WE = computeNeighborhoodQuality(iid);
            scores5[i] = FiveScores{WA, WB, WC, WD, WE};
        }

        // A candidate dominates another only if no criterion is worse.
        auto dominates = [&](size_t a, size_t b) -> bool {
            const FiveScores &sa = scores5[a];
            const FiveScores &sb = scores5[b];
            bool strictly_better = false;
            auto check = [&](float va, float vb) -> bool {
                if (va < vb) return false;
                if (va > vb) strictly_better = true;
                return true;
            };
            if (!check(sa.W_A, sb.W_A)) return false;
            if (!check(sa.W_B, sb.W_B)) return false;
            if (!check(sa.W_C, sb.W_C)) return false;
            if (!check(sa.W_D, sb.W_D)) return false;
            if (!check(sa.W_E, sb.W_E)) return false;
            return strictly_better;
        };

        size_t n = L.size();
        std::vector<bool> on_frontier(n, true);
        for (size_t i = 0; i < n; ++i) {
            if (!on_frontier[i]) continue;
            for (size_t j = 0; j < n; ++j) {
                if (i == j || !on_frontier[j]) continue;
                if (dominates(j, i)) {
                    on_frontier[i] = false;
                    break;
                }
            }
        }

        std::vector<CandidateEntry> out;
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (on_frontier[i]) out.push_back(L[i]);
        }
        for (size_t i = 0; i < n; ++i) {
            if (!on_frontier[i]) out.push_back(L[i]);
        }
        
        auto frontier_end = out.begin();
        size_t frontier_count = 0;
        for (size_t i = 0; i < n; ++i) {
            if (on_frontier[i]) ++frontier_count;
        }
        std::sort(out.begin(), out.begin() + frontier_count,
                  [](const CandidateEntry &a, const CandidateEntry &b) {
                      return a.insert_order < b.insert_order;
                  });
        std::sort(out.begin() + frontier_count, out.end(),
                  [](const CandidateEntry &a, const CandidateEntry &b) {
                      return a.insert_order < b.insert_order;
                  });
        return out;
    }

    tableint selectVfar(tableint u_internal,
                        const std::vector<tableint> &C,
                        tableint omega_internal,
                        Strategy strategy,
                        float eta,
                        const std::string &direction_mode) const
    {
        bool use_dd = (strategy == Strategy::DD        ||
                       strategy == Strategy::WVD_Linear ||
                       strategy == Strategy::WVD_Pareto);

        if (!use_dd) {
            return selectVfarByDistanceSum(u_internal, C);
        }

        if (direction_mode == "voronoi") {
            return selectVfarByDirectionDiversityVoronoi(u_internal, C, eta);
        }
        return selectVfarByDirectionDiversityCosine(u_internal, C, eta);
    }

    tableint selectVfarByDistanceSum(tableint u_internal,
                                     const std::vector<tableint> &C) const {
        size_t sz = C.size();
        std::vector<float> dist_sum(sz, 0.0f);
        std::vector<float> dist_to_u(sz, 0.0f);

        for (size_t i = 0; i < sz; ++i) {
            dist_to_u[i] = static_cast<float>(
                this->fstdistfunc_(this->getDataByInternalId(u_internal),
                                   this->getDataByInternalId(C[i]),
                                   this->dist_func_param_));
            for (size_t j = 0; j < sz; ++j) {
                if (i == j) continue;
                dist_sum[i] += static_cast<float>(
                    this->fstdistfunc_(this->getDataByInternalId(C[i]),
                                       this->getDataByInternalId(C[j]),
                                       this->dist_func_param_));
            }
        }

        size_t best = 0;
        for (size_t i = 1; i < sz; ++i) {
            if (dist_sum[i] < dist_sum[best] ||
                (dist_sum[i] == dist_sum[best] && dist_to_u[i] > dist_to_u[best])) {
                best = i;
            }
        }
        return C[best];
    }

    tableint selectVfarByDirectionDiversityCosine(tableint u_internal,
                                                   const std::vector<tableint> &C,
                                                   float eta) const {
        size_t sz  = C.size();
        size_t dim = *reinterpret_cast<const size_t *>(this->dist_func_param_);
        const float *u_vec =
            reinterpret_cast<const float *>(this->getDataByInternalId(u_internal));

        std::vector<std::vector<float>> dir(sz, std::vector<float>(dim, 0.0f));
        for (size_t i = 0; i < sz; ++i) {
            const float *v_vec =
                reinterpret_cast<const float *>(this->getDataByInternalId(C[i]));
            float norm = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                float diff = v_vec[d] - u_vec[d];
                dir[i][d]  = diff;
                norm       += diff * diff;
            }
            norm = std::sqrt(norm);
            if (norm > 0.0f) {
                for (size_t d = 0; d < dim; ++d) dir[i][d] /= norm;
            }
        }

        std::vector<float> dist_sum(sz, 0.0f);
        std::vector<float> dist_to_u(sz, 0.0f);
        for (size_t i = 0; i < sz; ++i) {
            dist_to_u[i] = static_cast<float>(
                this->fstdistfunc_(this->getDataByInternalId(u_internal),
                                   this->getDataByInternalId(C[i]),
                                   this->dist_func_param_));
            for (size_t j = 0; j < sz; ++j) {
                if (i == j) continue;
                dist_sum[i] += static_cast<float>(
                    this->fstdistfunc_(this->getDataByInternalId(C[i]),
                                       this->getDataByInternalId(C[j]),
                                       this->dist_func_param_));
            }
        }

        std::vector<float> concentration(sz, 0.0f);
        if (sz > 1) {
            for (size_t i = 0; i < sz; ++i) {
                float sum_pos = 0.0f;
                for (size_t j = 0; j < sz; ++j) {
                    if (i == j) continue;
                    float dot = 0.0f;
                    for (size_t d = 0; d < dim; ++d) {
                        dot += dir[i][d] * dir[j][d];
                    }
                    if (dot > 0.0f) sum_pos += dot;
                }
                concentration[i] = sum_pos / static_cast<float>(sz - 1);
            }
        }

        size_t best = 0;
        float  best_score = dist_sum[0] * (1.0f - eta * concentration[0]);
        for (size_t i = 1; i < sz; ++i) {
            float s = dist_sum[i] * (1.0f - eta * concentration[i]);
            if (s < best_score ||
                (s == best_score && dist_to_u[i] > dist_to_u[best])) {
                best       = i;
                best_score = s;
            }
        }
        return C[best];
    }

    tableint selectVfarByDirectionDiversityVoronoi(tableint u_internal,
                                                    const std::vector<tableint> &C,
                                                    float eta,
                                                    int p = 12) const {
        size_t sz  = C.size();
        size_t dim = *reinterpret_cast<const size_t *>(this->dist_func_param_);
        const float *u_vec =
            reinterpret_cast<const float *>(this->getDataByInternalId(u_internal));

        std::vector<int> sector(sz, 0);
        for (size_t i = 0; i < sz; ++i) {
            const float *v_vec =
                reinterpret_cast<const float *>(this->getDataByInternalId(C[i]));
            float dx = (dim >= 1) ? (v_vec[0] - u_vec[0]) : 0.0f;
            float dy = (dim >= 2) ? (v_vec[1] - u_vec[1]) : 0.0f;
            
            float angle = std::atan2(dy, dx); 
            if (angle < 0.0f) angle += 2.0f * 3.14159265358979f;
            sector[i] = static_cast<int>(angle / (2.0f * 3.14159265358979f) * p) % p;
        }

        std::vector<int> sector_count(p, 0);
        for (size_t i = 0; i < sz; ++i) {
            sector_count[sector[i]]++;
        }

        std::vector<float> dist_sum(sz, 0.0f);
        std::vector<float> dist_to_u(sz, 0.0f);
        for (size_t i = 0; i < sz; ++i) {
            dist_to_u[i] = static_cast<float>(
                this->fstdistfunc_(this->getDataByInternalId(u_internal),
                                   this->getDataByInternalId(C[i]),
                                   this->dist_func_param_));
            for (size_t j = 0; j < sz; ++j) {
                if (i == j) continue;
                dist_sum[i] += static_cast<float>(
                    this->fstdistfunc_(this->getDataByInternalId(C[i]),
                                       this->getDataByInternalId(C[j]),
                                       this->dist_func_param_));
            }
        }

        size_t best       = 0;
        float  best_score = dist_sum[0] *
                            (1.0f - eta * sector_count[sector[0]] / static_cast<float>(sz));
        for (size_t i = 1; i < sz; ++i) {
            float cv = sector_count[sector[i]] / static_cast<float>(sz);
            float s  = dist_sum[i] * (1.0f - eta * cv);
            if (s < best_score ||
                (s == best_score && dist_to_u[i] > dist_to_u[best])) {
                best       = i;
                best_score = s;
            }
        }
        return C[best];
    }

    float computeNeighborhoodQuality(tableint u_internal) const {
        linklistsizeint *ll = this->get_linklist0(u_internal);
        size_t           sz = this->getListCount(ll);
        if (sz == 0) return 1.0f;

        tableint *nbrs = reinterpret_cast<tableint *>(ll + 1);
        std::vector<float> dists(sz);
        float mean = 0.0f;
        for (size_t i = 0; i < sz; ++i) {
            dists[i] = static_cast<float>(
                this->fstdistfunc_(this->getDataByInternalId(u_internal),
                                   this->getDataByInternalId(nbrs[i]),
                                   this->dist_func_param_));
            mean += dists[i];
        }
        mean /= static_cast<float>(sz);

        float var = 0.0f;
        for (size_t i = 0; i < sz; ++i) {
            float diff = dists[i] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(sz);
        return 1.0f / (1.0f + var);
    }

    tableint getInternalIdOrThrow(labeltype label) const {
        std::unique_lock<std::mutex> lock(this->label_lookup_lock);
        auto it = this->label_lookup_.find(label);
        if (it == this->label_lookup_.end()) {
            throw std::runtime_error("label not found in index");
        }
        return it->second;
    }
};

} 
