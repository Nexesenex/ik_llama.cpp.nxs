#pragma once

#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace llama_moe_cache_detail {

// One device group's expert-slot bank.  The legacy blocking path owns slots
// through plan(); the asynchronous hybrid executor reserves and completes
// fills through the same state so that pins, generations and demand admission
// observe one consistent lifecycle.
struct moe_cache_lru {
    struct fill { int32_t expert; int32_t slot; };
    enum class slot_state : uint8_t { FREE, COPYING, READY };

    int32_t n_expert = 0;
    int32_t n_slots = 0;
    // Number of layers belonging to this cache group; a budget that can hold
    // every layer's experts admits misses immediately.
    int32_t n_group_layers = 0;
    std::vector<int32_t *> slot_map;
    std::vector<int32_t> key_of;
    std::vector<uint64_t> slot_generation;
    std::vector<uint32_t> pin_count;
    std::vector<slot_state> state;
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

    void init(int32_t n_layer, int32_t n_group_layers, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots = n_slots;
        this->n_group_layers = n_group_layers;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        slot_generation.assign(n_slots, 0);
        pin_count.assign(n_slots, 0);
        state.assign(n_slots, slot_state::FREE);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head = 0;
        tail = n_slots - 1;
        seen.assign(n_expert, 0);
        miss_demand.assign((size_t) n_layer * n_expert, 0);
    }

    // Freezing turns every slot reserved by the blocking path into a READY
    // source for the hybrid executor.  Slots without a resident key stay FREE.
    void freeze() {
        for (int32_t s = 0; s < n_slots; ++s) {
            state[s] = key_of[s] >= 0 ? slot_state::READY : slot_state::FREE;
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

    void touch(int32_t s) {
        if (s == tail) return;
        if (prev[s] >= 0) next[prev[s]] = next[s]; else head = next[s];
        prev[next[s]] = prev[s];
        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }

    void unlink(int32_t s) {
        if (head != s && prev[s] < 0 && next[s] < 0) return;
        if (prev[s] >= 0) next[prev[s]] = next[s]; else head = next[s];
        if (next[s] >= 0) prev[next[s]] = prev[s]; else tail = prev[s];
        prev[s] = next[s] = -1;
    }

    void link_tail(int32_t s) {
        prev[s] = tail;
        next[s] = -1;
        if (tail >= 0) next[tail] = s; else head = s;
        tail = s;
    }

    uint64_t advance_generation(int32_t s) {
        // Assignment and pin epochs reject stale or ABA releases for a reused slot.
        if (++slot_generation[s] == 0) ++slot_generation[s];
        return slot_generation[s];
    }

    // A pinned slot is READY, still maps to its owner and has no other pin.
    bool can_pin(int32_t il, int32_t expert) const {
        if (il < 0 || (size_t) il >= slot_map.size() || !slot_map[il] ||
                expert < 0 || expert >= n_expert) return false;
        const int32_t s = slot_map[il][expert];
        return s >= 0 && s < n_slots && state[s] == slot_state::READY &&
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
        return state[s] == slot_state::READY && key_of[s] == il*n_expert + expert &&
            slot_generation[s] == generation && pin_count[s] == 1 && slot_map[il][expert] == s;
    }

    void release(int32_t s) {
        if (s >= 0 && s < n_slots && pin_count[s] > 0) --pin_count[s];
    }

    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit, size_t & n_eviction) {
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
                touch(slots[e]);
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
            slots[e] = s;
            touch(s);
            fills.push_back({ e, s });
        }
        return true;
    }
};

// The blocking (legacy) cache may only be consulted when the asynchronous
// hybrid executor does not own the layers.  In async mode the scheduler copy
// callback is not installed, so a layer rejected by the executor must run the
// stock FFN graph instead of reading slot maps that only that callback fills.
inline bool legacy_lookup_allowed(bool has_cache, bool async_mode, bool has_lora) {
    return has_cache && !async_mode && !has_lora;
}

} // namespace llama_moe_cache_detail
