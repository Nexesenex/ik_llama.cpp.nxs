#pragma once

#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace llama_moe_cache_policy_detail {

enum class kind { lru, classlru };

inline bool valid(enum llama_moe_cache_policy policy) {
    return policy == LLAMA_MOE_CACHE_POLICY_LRU || policy == LLAMA_MOE_CACHE_POLICY_CLASSLRU;
}

// One device group's expert-slot bank.  The legacy blocking path owns slots
// through plan(); the asynchronous hybrid executor reserves and completes
// fills through the same state so that pins, generations and demand admission
// observe one consistent lifecycle.
struct state {
    static constexpr int n_class = 8;
    static constexpr float half_life = 128.0f;

    struct fill { int32_t expert; int32_t slot; };
    enum class slot_state : uint8_t { FREE, COPYING, READY };

    int32_t n_expert = 0;
    int32_t n_slots = 0;
    // Number of layers belonging to this cache group; a budget that can hold
    // every layer's experts admits misses immediately.
    int32_t n_group_layers = 0;
    kind policy = kind::lru;
    std::vector<int32_t *> slot_map;
    std::vector<int32_t> key_of;
    std::vector<uint64_t> slot_generation;
    std::vector<uint32_t> pin_count;
    std::vector<slot_state> slot_states;
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    std::vector<uint32_t> seen;
    std::vector<int32_t> uniq;
    // Consecutive observations of a missing (layer, expert) key.  Evicting a
    // resident expert for a one-off route is destructive at low hit rates, so
    // fills are demand-driven instead of admitted on every miss.
    std::vector<uint8_t> miss_demand;
    uint32_t seen_gen = 0;

    int32_t head = -1;
    int32_t tail = -1;
    std::vector<int8_t> list_of;
    std::array<int32_t, n_class> class_head;
    std::array<int32_t, n_class> class_tail;
    std::vector<float> crf;
    std::vector<int64_t> last_use;
    std::vector<int64_t> clock;

    // n_group_layers is the number of layers served by this group.  The
    // default keeps the four-argument form used by the policy tests and only
    // affects demand admission, which is inert for the blocking planner.
    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots, kind policy,
            int32_t n_group_layers = 0) {
        GGML_ASSERT(n_layer >= 0 && n_expert > 0 && n_slots >= 0);
        this->n_expert = n_expert;
        this->n_slots = n_slots;
        this->n_group_layers = n_group_layers;
        this->policy = policy;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        slot_generation.assign(n_slots, 0);
        pin_count.assign(n_slots, 0);
        slot_states.assign(n_slots, slot_state::FREE);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        seen.assign(n_expert, 0);
        miss_demand.assign((size_t) n_layer * n_expert, 0);
        if (policy == kind::lru) {
            head = n_slots ? 0 : -1;
            tail = n_slots - 1;
            return;
        }
        list_of.assign(n_slots, 0);
        class_head.fill(-1);
        class_tail.fill(-1);
        if (n_slots > 0) {
            class_head[0] = 0;
            class_tail[0] = n_slots - 1;
        }
        crf.assign((size_t) n_layer*n_expert, 0.0f);
        last_use.assign((size_t) n_layer*n_expert, -1);
        clock.assign(n_layer, 0);
    }

    // Freezing turns every slot reserved by the blocking path into a READY
    // source for the hybrid executor.  Slots without a resident key stay FREE.
    void freeze() {
        for (int32_t s = 0; s < n_slots; ++s) {
            slot_states[s] = key_of[s] >= 0 ? slot_state::READY : slot_state::FREE;
        }
    }

    struct miss_observation {
        uint8_t demand = 0;
        bool incremented = false;
    };

    miss_observation observe_miss(int32_t key) {
        miss_observation observation;
        if (key < 0 || (size_t) key >= miss_demand.size()) return observation;
        uint8_t & demand = miss_demand[key];
        observation.incremented = demand < std::numeric_limits<uint8_t>::max();
        if (observation.incremented) ++demand;
        observation.demand = demand;
        return observation;
    }

    void reset_miss_demand(int32_t key) {
        if (key >= 0 && (size_t) key < miss_demand.size()) miss_demand[key] = 0;
    }

    // Demand admission.  A first miss is admitted when the group can hold all
    // of its experts anyway, otherwise it must be observed twice before a free
    // slot is spent.  Replacing a resident expert requires eight observations,
    // which keeps one-off routes from thrashing the bank.
    uint8_t admission_threshold(bool has_free_slot) const {
        const bool full_group_resident = (size_t) n_slots >=
            (size_t) n_group_layers * (size_t) n_expert;
        return has_free_slot ? (full_group_resident ? 1 : 2) : 8;
    }

    bool admit_miss(int32_t key, bool has_free_slot) {
        const uint8_t threshold = admission_threshold(has_free_slot);
        return observe_miss(key).demand >= threshold;
    }

    void touch_lru(int32_t s) {
        if (s == tail) return;
        if (prev[s] >= 0) next[prev[s]] = next[s]; else head = next[s];
        prev[next[s]] = prev[s];
        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }

    void move_class(int32_t s, int c) {
        const int from = list_of[s];
        if (from == c && s == class_tail[from]) return;
        if (from >= 0) {
            if (prev[s] >= 0) next[prev[s]] = next[s]; else class_head[from] = next[s];
            if (next[s] >= 0) prev[next[s]] = prev[s]; else class_tail[from] = prev[s];
        }
        prev[s] = class_tail[c];
        next[s] = -1;
        if (class_tail[c] >= 0) next[class_tail[c]] = s; else class_head[c] = s;
        class_tail[c] = s;
        list_of[s] = (int8_t) c;
    }

    // Recency update for one slot under either policy.  ClassLRU moves the
    // slot to the class described by its resident key; use() must have been
    // called for that key at least once (plan() and finish_fill() do so).
    void touch(int32_t s) {
        if (s < 0 || s >= n_slots) return;
        if (policy == kind::lru) {
            touch_lru(s);
            return;
        }
        if (key_of[s] < 0) {
            move_class(s, 0);
            return;
        }
        move_class(s, class_of(key_of[s]));
    }

    void unlink(int32_t s) {
        if (s < 0 || s >= n_slots) return;
        if (policy == kind::lru) {
            if (head != s && prev[s] < 0 && next[s] < 0) return;
            if (prev[s] >= 0) next[prev[s]] = next[s]; else head = next[s];
            if (next[s] >= 0) prev[next[s]] = prev[s]; else tail = prev[s];
            prev[s] = next[s] = -1;
            return;
        }
        const int from = list_of[s];
        if (from < 0) return;
        if (prev[s] >= 0) next[prev[s]] = next[s]; else class_head[from] = next[s];
        if (next[s] >= 0) prev[next[s]] = prev[s]; else class_tail[from] = prev[s];
        prev[s] = next[s] = -1;
        list_of[s] = -1;
    }

    void link_tail(int32_t s) {
        if (s < 0 || s >= n_slots) return;
        if (policy == kind::lru) {
            prev[s] = tail;
            next[s] = -1;
            if (tail >= 0) next[tail] = s; else head = s;
            tail = s;
            return;
        }
        if (list_of[s] >= 0) unlink(s);
        if (key_of[s] < 0) {
            move_class(s, 0);
            return;
        }
        move_class(s, class_of(key_of[s]));
    }

    uint64_t advance_generation(int32_t s) {
        // Assignment and pin epochs reject stale or ABA releases for a reused slot.
        if (++slot_generation[s] == 0) ++slot_generation[s];
        return slot_generation[s];
    }

    int32_t make_key(int32_t il, int32_t expert) const { return il*n_expert + expert; }
    int32_t key_layer(int32_t key) const { return key/n_expert; }
    int32_t key_expert(int32_t key) const { return key%n_expert; }

    int class_of(int32_t key) const {
        GGML_ASSERT(std::isfinite(crf[key]) && crf[key] >= 1.0f);
        return std::min(n_class - 1, (int) std::log2(crf[key]));
    }

    float decayed_crf(int32_t il, int32_t key) const {
        return crf[key]*std::exp2(-(clock[il] - last_use[key])/half_life);
    }

    void use(int32_t il, int32_t key) {
        crf[key] = decayed_crf(il, key) + 1.0f;
        last_use[key] = clock[il];
        GGML_ASSERT(std::isfinite(crf[key]) && crf[key] >= 1.0f);
    }

    void advance_clock(int32_t il, int64_t n_tokens) {
        if (policy != kind::classlru || il < 0 || (size_t) il >= clock.size() || n_tokens <= 0) return;
        clock[il] += n_tokens;
    }

    // Account for one routed expert.  Both resident and missing keys enter the
    // frequency model; a resident hit is also promoted to its new class.
    void observe_route(int32_t il, int32_t expert, int32_t slot) {
        if (policy != kind::classlru) return;
        const int32_t key = make_key(il, expert);
        use(il, key);
        if (slot >= 0) touch(slot);
    }

    // A pinned slot is READY, still maps to its owner and has no other pin.
    bool can_pin(int32_t il, int32_t expert) const {
        if (il < 0 || (size_t) il >= slot_map.size() || !slot_map[il] ||
                expert < 0 || expert >= n_expert) return false;
        const int32_t s = slot_map[il][expert];
        return s >= 0 && s < n_slots && slot_states[s] == slot_state::READY &&
            key_of[s] == il*n_expert + expert && slot_generation[s] != 0 && pin_count[s] == 0;
    }

    bool pin(int32_t il, int32_t expert, uint64_t & generation) {
        if (!can_pin(il, expert)) return false;
        const int32_t s = slot_map[il][expert];
        generation = advance_generation(s);
        ++pin_count[s];
        touch(s);
        return true;
    }

    // Release validation rejects a token whose slot has been reassigned, which
    // is how an ABA reuse of the same slot stays safe.
    bool can_release(int32_t il, int32_t expert, int32_t s, uint64_t generation) const {
        if (il < 0 || (size_t) il >= slot_map.size() || !slot_map[il] ||
                expert < 0 || expert >= n_expert || s < 0 || s >= n_slots) return false;
        return slot_states[s] == slot_state::READY && key_of[s] == il*n_expert + expert &&
            slot_generation[s] == generation && pin_count[s] == 1 && slot_map[il][expert] == s;
    }

    void release(int32_t s) {
        if (s >= 0 && s < n_slots && pin_count[s] > 0) --pin_count[s];
    }

    bool is_selected(int32_t il, int32_t key) const {
        return key >= 0 && key_layer(key) == il && seen[key_expert(key)] == seen_gen;
    }

    int32_t candidate(int c, int32_t il, bool honor_selection) const {
        for (int32_t s = class_head[c]; s >= 0; s = next[s]) {
            if (pin_count[s] != 0 || slot_states[s] == slot_state::COPYING) continue;
            if (honor_selection && is_selected(il, key_of[s])) continue;
            return s;
        }
        return -1;
    }

    int32_t victim(int32_t il, bool honor_selection) const {
        int32_t best = -1;
        double best_score = -1.0;
        for (int c = 0; c < n_class; ++c) {
            const int32_t s = candidate(c, il, honor_selection);
            if (s < 0) continue;
            const int32_t key = key_of[s];
            if (key < 0) return s;
            const int64_t age = clock[key_layer(key)] - last_use[key];
            const double score = std::ldexp((double) age, -c);
            if (score > best_score) {
                best = s;
                best_score = score;
            }
        }
        return best;
    }

    // Eviction candidate for the asynchronous executor.  Pinned and COPYING
    // slots never qualify, and routing selection is not consulted because the
    // executor pins every resident hit before it enqueues a fill.
    int32_t eviction_candidate(int32_t il, bool honor_selection = true) const {
        if (policy == kind::lru) {
            for (int32_t s = head; s >= 0; s = next[s]) {
                if (slot_states[s] != slot_state::READY || key_of[s] < 0 || pin_count[s] != 0) continue;
                return s;
            }
            return -1;
        }
        return victim(il, honor_selection);
    }

    bool plan_lru(int32_t il, const int32_t * ids, size_t n_ids,
            std::vector<fill> & fills, size_t & n_hit, size_t & n_eviction) {
        fills.clear();
        n_hit = 0;
        n_eviction = 0;
        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) return false;
        int32_t * slots = slot_map[il];
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch_lru(slots[e]);
                ++n_hit;
            }
        }
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) continue;
            const int32_t s = head;
            if (key_of[s] >= 0) {
                const int32_t old = key_of[s];
                slot_map[old / n_expert][old % n_expert] = -1;
                ++n_eviction;
            }
            key_of[s] = il*n_expert + e;
            advance_generation(s);
            slot_states[s] = slot_state::READY;
            slots[e] = s;
            touch_lru(s);
            fills.push_back({ e, s });
        }
        return true;
    }

    bool plan_classlru(int32_t il, const int32_t * ids, size_t n_ids, int64_t n_tokens,
            std::vector<fill> & fills, size_t & n_hit, size_t & n_eviction) {
        fills.clear();
        n_hit = n_eviction = 0;
        if (il < 0 || (size_t) il >= slot_map.size() || !slot_map[il] || !ids ||
                n_tokens <= 0 || n_ids == 0 || (uint64_t) n_tokens > n_ids ||
                n_ids % (size_t) n_tokens != 0) return false;
        for (size_t i = 0; i < n_ids; ++i) {
            if (ids[i] < 0 || ids[i] >= n_expert) return false;
        }
        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) return false;
        int32_t * slots = slot_map[il];
        for (int32_t e : uniq) {
            const int32_t key = make_key(il, e);
            use(il, key);
            if (slots[e] >= 0) {
                move_class(slots[e], class_of(key));
                ++n_hit;
            }
        }
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) continue;
            const int32_t s = victim(il, true);
            GGML_ASSERT(s >= 0);
            if (key_of[s] >= 0) {
                const int32_t old = key_of[s];
                slot_map[key_layer(old)][key_expert(old)] = -1;
                ++n_eviction;
            }
            const int32_t key = make_key(il, e);
            key_of[s] = key;
            advance_generation(s);
            slot_states[s] = slot_state::READY;
            slots[e] = s;
            move_class(s, class_of(key));
            fills.push_back({e, s});
        }
        advance_clock(il, n_tokens);
        return true;
    }

    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int64_t n_tokens,
            std::vector<fill> & fills, size_t & n_hit, size_t & n_eviction) {
        if (policy == kind::lru) return plan_lru(il, ids, n_ids, fills, n_hit, n_eviction);
        return plan_classlru(il, ids, n_ids, n_tokens, fills, n_hit, n_eviction);
    }
};

inline bool recover_selected_tokens(const ggml_tensor * flattened, int64_t n_expert_used,
        int64_t n_expert, int64_t & n_tokens) {
    if (!flattened || flattened->op != GGML_OP_RESHAPE || !flattened->src[0]) return false;
    const ggml_tensor * source = flattened->src[0];
    if (n_expert_used <= 0 || n_expert <= 0 || n_expert_used > n_expert ||
            source->ne[0] <= 0 || source->ne[0] > n_expert || source->ne[1] <= 0 ||
            source->ne[2] != 1 || source->ne[3] != 1 || flattened->ne[0] != ggml_nelements(source) ||
            flattened->ne[1] != 1 || flattened->ne[2] != 1 || flattened->ne[3] != 1 ||
            ggml_nelements(flattened) != ggml_nelements(source)) return false;
    n_tokens = source->ne[1];
    return true;
}

inline bool slot_map_eligible(int64_t n_tokens, int64_t n_expert_used,
        int64_t slot_entries, int64_t n_slots, int64_t max_batch) {
    if (n_tokens <= 0 || n_tokens > max_batch || n_expert_used <= 0 ||
            slot_entries < 0 || n_slots < 0 || n_tokens > INT64_MAX/n_expert_used) return false;
    return std::min<int64_t>(n_tokens*n_expert_used, slot_entries) <= n_slots;
}

} // namespace llama_moe_cache_policy_detail
