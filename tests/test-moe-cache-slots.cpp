#include "llama-moe-cache-lru.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

using llama_moe_cache_detail::moe_cache_lru;

namespace {

struct fixture {
    moe_cache_lru lru;
    std::vector<std::vector<int32_t>> maps;

    fixture(int32_t n_layers, int32_t n_group_layers, int32_t n_expert, int32_t n_slots) {
        lru.init(n_layers, n_expert, n_slots, llama_moe_cache_policy_detail::kind::lru, n_group_layers);
        maps.assign(n_layers, std::vector<int32_t>((size_t) n_expert, -1));
        for (size_t il = 0; il < maps.size(); ++il) lru.slot_map[il] = maps[il].data();
    }

    bool plan(int32_t il, const std::vector<int32_t> & ids, std::vector<moe_cache_lru::fill> & fills,
            size_t & hits, size_t & evictions) {
        return lru.plan(il, ids.data(), ids.size(), 1, fills, hits, evictions);
    }
};

bool fail(const char * what) {
    std::fprintf(stderr, "MoE cache slot manager: %s\n", what);
    return false;
}

// plan() coalesces a token's duplicate experts, assigns free slots in LRU
// order and reports only unique resident experts as hits.
bool test_planning() {
    fixture f(2, 2, 4, 4);
    std::vector<moe_cache_lru::fill> fills;
    size_t hits = 0, evictions = 0;
    if (!f.plan(0, { 0, 1, 1 }, fills, hits, evictions)) return fail("initial plan rejected");
    if (fills.size() != 2 || hits != 0 || evictions != 0) return fail("initial plan accounting");
    if (fills[0].expert != 0 || fills[0].slot != 0 ||
            fills[1].expert != 1 || fills[1].slot != 1) return fail("initial plan slots");
    if (f.maps[0][0] != 0 || f.maps[0][1] != 1 || f.lru.key_of[0] != 0 || f.lru.key_of[1] != 1) {
        return fail("initial slot map");
    }
    if (f.lru.slot_generation[0] != 1 || f.lru.slot_generation[1] != 1) {
        return fail("assignment generation");
    }

    f.lru.freeze();
    if (!f.plan(0, { 0, 1, 1 }, fills, hits, evictions)) return fail("resident plan rejected");
    if (!fills.empty() || hits != 2 || evictions != 0) return fail("resident plan accounting");
    return true;
}

// A route that needs more unique experts than the group has slots must be
// rejected without mutating the bank.
bool test_oversized_route() {
    fixture f(2, 2, 8, 3);
    std::vector<moe_cache_lru::fill> fills;
    size_t hits = 0, evictions = 0;
    if (!f.plan(0, { 0, 1, 2 }, fills, hits, evictions)) return fail("seed plan rejected");
    f.lru.freeze();
    if (f.plan(0, { 0, 1, 2, 3 }, fills, hits, evictions)) return fail("oversized route accepted");
    if (!fills.empty() || hits != 0 || evictions != 0) return fail("oversized route accounting");
    if (f.maps[0][3] != -1) return fail("oversized route mutated the slot map");
    return true;
}

// Touching a hit moves it to the MRU end, so the next fill evicts the least
// recently used resident expert.
bool test_eviction_order() {
    fixture f(4, 4, 8, 3);
    std::vector<moe_cache_lru::fill> fills;
    size_t hits = 0, evictions = 0;
    if (!f.plan(0, { 0, 1, 2 }, fills, hits, evictions)) return fail("eviction seed rejected");
    f.lru.freeze();
    if (!f.plan(0, { 0 }, fills, hits, evictions) || hits != 1) return fail("eviction touch failed");
    if (!f.plan(0, { 3 }, fills, hits, evictions)) return fail("eviction fill rejected");
    if (fills.size() != 1 || fills[0].expert != 3 || fills[0].slot != 1 || evictions != 1) {
        return fail("eviction victim is not the LRU slot");
    }
    if (f.maps[0][1] != -1 || f.maps[0][3] != 1 || f.lru.key_of[1] != 3 ||
            f.lru.slot_generation[1] != 2) return fail("eviction slot map/generation");

    f.lru.freeze();
    if (!f.plan(0, { 0, 2, 3 }, fills, hits, evictions) || hits != 3) {
        return fail("resident set after eviction");
    }
    return true;
}

// Pin tokens advance the slot generation, and releasing a stale token after
// the slot was reassigned (ABA) must be rejected.
bool test_pins_and_aba() {
    fixture f(1, 1, 2, 1);
    std::vector<moe_cache_lru::fill> fills;
    size_t hits = 0, evictions = 0;
    if (!f.plan(0, { 0 }, fills, hits, evictions)) return fail("pin seed rejected");
    f.lru.freeze();
    if (!f.lru.can_pin(0, 0)) return fail("resident expert is not pinnable");

    uint64_t first = 0;
    if (!f.lru.pin(0, 0, first) || first != 2) return fail("pin generation");
    if (f.lru.pin_count[0] != 1) return fail("pin count not incremented");
    if (f.lru.can_pin(0, 0)) return fail("already pinned slot accepted a pin");
    uint64_t duplicate = 0;
    if (f.lru.pin(0, 0, duplicate)) return fail("double pin accepted");
    if (!f.lru.can_release(0, 0, 0, first)) return fail("valid release rejected");
    f.lru.release(0);
    if (f.lru.pin_count[0] != 0) return fail("pin count not decremented");
    if (f.lru.can_release(0, 0, 0, first)) return fail("release without a pin accepted");

    if (!f.plan(0, { 1 }, fills, hits, evictions) || evictions != 1) {
        return fail("ABA reassignment failed");
    }
    if (f.lru.key_of[0] != 1 || f.lru.slot_generation[0] != 3) {
        return fail("ABA reassignment generation");
    }
    f.lru.freeze();
    uint64_t second = 0;
    if (!f.lru.pin(0, 1, second) || second != 4) return fail("reused slot pin generation");
    if (f.lru.can_release(0, 0, 0, first)) return fail("stale ABA token accepted");
    if (!f.lru.can_release(0, 1, 0, second)) return fail("reused slot token rejected");
    f.lru.release(0);
    if (f.lru.can_release(0, 1, 0, second)) return fail("double release accepted");
    return true;
}

// Generation zero is reserved so a fresh slot can never satisfy a stale token,
// and incrementing the maximum wraps back to one instead of zero.
bool test_generation_wrap() {
    fixture f(1, 1, 1, 1);
    f.lru.slot_generation[0] = std::numeric_limits<uint64_t>::max();
    if (f.lru.advance_generation(0) != 1) return fail("generation wrap did not skip zero");
    if (f.lru.advance_generation(0) != 2) return fail("generation did not resume after wrap");
    return true;
}

// Repeated eviction/reuse cycles must keep the LRU links coherent.
bool test_wraparound_reuse() {
    fixture f(2, 2, 4, 2);
    std::vector<moe_cache_lru::fill> fills;
    size_t hits = 0, evictions = 0;
    if (!f.plan(0, { 0, 1, 1 }, fills, hits, evictions) || fills.size() != 2) {
        return fail("wrap-around seed rejected");
    }
    f.lru.freeze();
    for (int32_t i = 0; i < 300; ++i) {
        const int32_t expert = 2 + i % 2;
        if (!f.plan(0, { 0, expert, expert }, fills, hits, evictions)) {
            return fail("wrap-around plan rejected");
        }
        if (hits != 1 || fills.size() != 1 || evictions != 1 || fills[0].expert != expert) {
            return fail("wrap-around accounting");
        }
    }
    if (f.maps[0][0] != 0) return fail("stable expert was evicted during wrap-around");
    if (f.lru.slot_generation[1] != 301) return fail("wrap-around generation drift");
    return true;
}

// Admission is demand-driven: immediate when the whole group fits, twice for a
// free slot in a partial group, and eight times when a resident expert must be
// replaced.  A successful commit resets the counter.
bool test_demand_admission() {
    fixture full_group(2, 2, 2, 4);
    if (full_group.lru.admission_threshold(true) != 1) return fail("full-group threshold");
    if (full_group.lru.admission_threshold(false) != 8) return fail("full-group eviction threshold");
    if (!full_group.lru.admit_miss(0, true)) return fail("full-group first miss was not admitted");

    fixture partial_group(4, 4, 2, 3);
    if (partial_group.lru.admission_threshold(true) != 2) return fail("partial-group threshold");
    if (partial_group.lru.admit_miss(0, true)) return fail("partial-group first miss was admitted");
    if (!partial_group.lru.admit_miss(0, true)) return fail("partial-group second miss was not admitted");
    partial_group.lru.reset_miss_demand(0);
    if (partial_group.lru.admit_miss(0, true)) return fail("demand reset did not clear the counter");
    if (!partial_group.lru.admit_miss(0, true)) return fail("demand did not rebuild after reset");

    if (partial_group.lru.admission_threshold(false) != 8) return fail("eviction threshold");
    for (int32_t i = 0; i < 7; ++i) {
        if (partial_group.lru.admit_miss(1, false)) return fail("eviction admitted too early");
    }
    if (!partial_group.lru.admit_miss(1, false)) return fail("eviction admission did not mature");

    const auto saturated = [&]() {
        moe_cache_lru::miss_observation observation;
        for (int32_t i = 0; i < 300; ++i) observation = partial_group.lru.observe_miss(2);
        return observation;
    }();
    if (saturated.demand != std::numeric_limits<uint8_t>::max() || saturated.incremented) {
        return fail("demand counter saturation");
    }
    if (partial_group.lru.observe_miss(-1).demand != 0 ||
            partial_group.lru.observe_miss(1000000).demand != 0) {
        return fail("out-of-range demand observation");
    }
    return true;
}

// The legacy blocking lookup must stay disabled in async mode and while LoRA
// adapters are active, and an in-flight (COPYING) slot must not be pinnable.
bool test_async_guard() {
    using llama_moe_cache_detail::legacy_lookup_allowed;
    if (!legacy_lookup_allowed(true, false, false)) return fail("legacy lookup disabled unexpectedly");
    if (legacy_lookup_allowed(false, false, false)) return fail("legacy lookup without a cache");
    if (legacy_lookup_allowed(true, true, false)) return fail("legacy lookup enabled in async mode");
    if (legacy_lookup_allowed(true, false, true)) return fail("legacy lookup enabled with LoRA");

    fixture f(1, 1, 1, 1);
    std::vector<moe_cache_lru::fill> fills;
    size_t hits = 0, evictions = 0;
    if (!f.plan(0, { 0 }, fills, hits, evictions)) return fail("async guard seed rejected");
    f.lru.freeze();
    f.lru.slot_states[0] = moe_cache_lru::slot_state::COPYING;
    uint64_t generation = 0;
    if (f.lru.can_pin(0, 0) || f.lru.pin(0, 0, generation)) return fail("in-flight fill was pinned");
    f.lru.slot_states[0] = moe_cache_lru::slot_state::FREE;
    if (f.lru.can_pin(0, 0)) return fail("cancelled fill stayed pinnable");
    return true;
}

} // namespace

int main() {
    bool ok = true;
    ok &= test_planning();
    ok &= test_oversized_route();
    ok &= test_eviction_order();
    ok &= test_pins_and_aba();
    ok &= test_generation_wrap();
    ok &= test_wraparound_reuse();
    ok &= test_demand_admission();
    ok &= test_async_guard();
    if (ok) std::puts("MoE cache slot manager passed");
    return ok ? 0 : 1;
}
