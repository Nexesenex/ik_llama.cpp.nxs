#include "llama-moe-cache-share.h"
#include "llama-moe-cache-policy.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static bool check(const char * name, const std::vector<float> & cumulative,
        const std::vector<bool> & eligible, const std::vector<double> & expected) {
    const auto actual = llama_moe_cache_detail::device_shares(cumulative, eligible);
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "%s: share count mismatch\n", name);
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > 1e-9) {
            std::fprintf(stderr, "%s: share %zu was %.12f, expected %.12f\n",
                    name, i, actual[i], expected[i]);
            return false;
        }
    }
    return true;
}

static bool check_cache_policy() {
    using llama_moe_cache_policy_detail::kind;
    using llama_moe_cache_policy_detail::state;
    state default_lru, explicit_lru;
    default_lru.init(2, 8, 3, kind::lru);
    explicit_lru.init(2, 8, 3, kind::lru);
    std::vector<int32_t> maps_a(16, -1), maps_b(16, -1);
    default_lru.slot_map[0] = maps_a.data();
    explicit_lru.slot_map[0] = maps_b.data();
    const std::vector<std::vector<int32_t>> routes = {{1, 1, 2}, {2, 3}, {0, 3}, {4, 5}};
    for (const auto & route : routes) {
        std::vector<state::fill> a, b;
        size_t ah = 0, ae = 0, bh = 0, be = 0;
        if (!default_lru.plan(0, route.data(), route.size(), 1, a, ah, ae) ||
                !explicit_lru.plan(0, route.data(), route.size(), 1, b, bh, be) ||
                ah != bh || ae != be || a.size() != b.size() || maps_a != maps_b ||
                default_lru.key_of != explicit_lru.key_of) { std::fprintf(stderr, "LRU parity\n"); return false; }
        const size_t occupied = std::count_if(default_lru.key_of.begin(), default_lru.key_of.end(),
                [](int32_t key) { return key >= 0; });
        if (default_lru.filled_real_slots != occupied || explicit_lru.filled_real_slots != occupied) {
            std::fprintf(stderr, "LRU occupancy\n"); return false;
        }
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].expert != b[i].expert || a[i].slot != b[i].slot) { std::fprintf(stderr, "LRU fills\n"); return false; }
        }
    }

    state classlru;
    classlru.init(2, 8, 3, kind::classlru);
    std::vector<int32_t> layer0(8, -1), layer1(8, -1);
    classlru.slot_map[0] = layer0.data();
    classlru.slot_map[1] = layer1.data();
    // SER sentinels (-1) are skipped before they can consume a slot.
    state ser_lru;
    ser_lru.init(1, 8, 2, kind::lru);
    std::vector<int32_t> ser_map(8, -1);
    ser_lru.slot_map[0] = ser_map.data();
    std::vector<state::fill> ser_fills;
    size_t ser_hits = 0, ser_evictions = 0;
    const std::vector<int32_t> ser_route = {-1, -1, 6, -1};
    if (!ser_lru.plan(0, ser_route.data(), ser_route.size(), 1, ser_fills, ser_hits, ser_evictions) ||
            ser_fills.size() != 1 || ser_fills[0].expert != 6 || ser_hits != 0 ||
            ser_evictions != 0 || ser_lru.filled_real_slots != 1 || ser_map[6] < 0) {
        std::fprintf(stderr, "LRU SER route\n"); return false;
    }
    ser_fills.clear();
    const std::vector<int32_t> ser_dropped = {-1, -1};
    if (!ser_lru.plan(0, ser_dropped.data(), ser_dropped.size(), 1, ser_fills, ser_hits, ser_evictions) ||
            !ser_fills.empty() || ser_hits != 0 || ser_evictions != 0 ||
            ser_lru.filled_real_slots != 1 || ser_map[6] < 0) {
        std::fprintf(stderr, "LRU SER dropped-only route\n"); return false;
    }
    state ser_class;
    ser_class.init(1, 8, 2, kind::classlru);
    std::vector<int32_t> ser_class_map(8, -1);
    ser_class.slot_map[0] = ser_class_map.data();
    if (!ser_class.plan(0, ser_route.data(), ser_route.size(), 1, ser_fills, ser_hits, ser_evictions) ||
            ser_fills.size() != 1 || ser_fills[0].expert != 6 || ser_hits != 0 ||
            ser_evictions != 0 || ser_class.filled_real_slots != 1 || ser_class_map[6] < 0) {
        std::fprintf(stderr, "ClassLRU SER route\n"); return false;
    }
    ser_fills.clear();
    if (!ser_class.plan(0, ser_dropped.data(), ser_dropped.size(), 1, ser_fills, ser_hits, ser_evictions) ||
            !ser_fills.empty() || ser_hits != 0 || ser_evictions != 0 ||
            ser_class.filled_real_slots != 1 || ser_class_map[6] < 0) {
        std::fprintf(stderr, "ClassLRU SER dropped-only route\n"); return false;
    }
    auto plan = [&](int layer, std::vector<int32_t> ids, int tokens, size_t want_hits) {
        std::vector<state::fill> fills;
        size_t hits = 0, evictions = 0;
        return classlru.plan(layer, ids.data(), ids.size(), tokens, fills, hits, evictions) &&
                hits == want_hits && classlru.clock[layer] >= tokens;
    };
    if (!plan(0, {0, 1, 2}, 1, 0) || !plan(0, {0, 3}, 1, 1) || layer0[0] < 0 ||
            !plan(1, {0, 0}, 2, 0) || layer1[0] < 0 || classlru.key_of[layer1[0]] != 8 ||
            (layer0[0] >= 0 && classlru.key_of[layer0[0]] != 0)) { std::fprintf(stderr, "ClassLRU routes/maps\n"); return false; }
    if (classlru.filled_real_slots != 3) { std::fprintf(stderr, "ClassLRU occupancy\n"); return false; }
    std::vector<int32_t> before_key = classlru.key_of;
    std::vector<int32_t> before0 = layer0;
    std::vector<state::fill> fills;
    size_t hits = 0, evictions = 0;
    const int32_t too_many[] = {4, 5, 6, 7};
    if (classlru.plan(0, too_many, 4, 1, fills, hits, evictions) ||
            before_key != classlru.key_of || before0 != layer0) { std::fprintf(stderr, "failed-plan mutation\n"); return false; }
    if (classlru.crf[0] < 1.0f || classlru.clock[0] != 2 || classlru.clock[1] != 2) { std::fprintf(stderr, "ClassLRU clock/score\n"); return false; }
    state empty;
    empty.init(1, 8, 0, kind::classlru);
    if (empty.filled_real_slots != 0) { std::fprintf(stderr, "empty occupancy\n"); return false; }
    empty.slot_map[0] = layer0.data();
    if (empty.plan(0, too_many, 4, 1, fills, hits, evictions)) { std::fprintf(stderr, "empty cache accepted fill\n"); return false; }
    return default_lru.crf.empty() && default_lru.last_use.empty() &&
            default_lru.clock.empty() && classlru.crf.size() == 16;
}

static bool check_cpu_share_plan() {
    using namespace llama_moe_cache_policy_detail;
    constexpr int n_experts = 8;
    constexpr int real_rows = 3;
    constexpr int zero_rows = 2;
    state cache;
    cache.init(1, n_experts, real_rows, kind::lru);
    std::vector<int32_t> slots(n_experts, -1);
    cache.slot_map[0] = slots.data();
    state::cpu_share_plan plan;
    const std::vector<int32_t> seed = {0, 1, 5};
    if (!cache.plan_with_cpu_share(0, seed.data(), seed.size(), 1, seed.size(), 0.5, zero_rows, plan) ||
            !plan.cpu.empty() || cache.filled_real_slots != real_rows) {
        std::fprintf(stderr, "underfull share plan\n"); return false;
    }

    const std::vector<int32_t> routes = {2, 2, 3, 4};
    if (!cache.plan_with_cpu_share(0, routes.data(), routes.size(), 2, 2, 1.0, zero_rows, plan)) {
        std::fprintf(stderr, "mixed CPU-share plan rejected\n"); return false;
    }
    bool cpu3 = false, cpu4 = false;
    for (const auto & candidate : plan.cpu) {
        cpu3 |= candidate.expert == 3;
        cpu4 |= candidate.expert == 4;
    }
    if (plan.cpu.size() != 2 || !cpu3 || !cpu4 || plan.fills.size() != 1 || plan.fills[0].expert != 2 ||
            plan.fills[0].slot < 0 || plan.fills[0].slot >= real_rows ||
            slots[2] != plan.fills[0].slot || cache.key_of[plan.fills[0].slot] != cache.make_key(0, 2) ||
            plan.cpu_map[2] != -1 || plan.gpu_map[2] != plan.fills[0].slot ||
            plan.gpu_map[3] < real_rows || plan.gpu_map[3] >= real_rows + zero_rows ||
            plan.gpu_map[4] < real_rows || plan.gpu_map[4] >= real_rows + zero_rows ||
            plan.gpu_map[3] == plan.gpu_map[4] ||
            plan.cpu_map[3] != 3 || plan.cpu_map[4] != 4 ||
            cache.filled_real_slots != real_rows) {
        std::fprintf(stderr, "mixed CPU-share route/zero-slot assignment\n"); return false;
    }

    state zero, ordinary;
    constexpr int parity_rows = 2;
    zero.init(1, n_experts, parity_rows, kind::classlru);
    ordinary.init(1, n_experts, parity_rows, kind::classlru);
    std::vector<int32_t> zero_slots(n_experts, -1), ordinary_slots(n_experts, -1);
    zero.slot_map[0] = zero_slots.data(); ordinary.slot_map[0] = ordinary_slots.data();
    const std::vector<int32_t> route = {0, 1};
    std::vector<state::fill> fills;
    size_t hits = 0, evictions = 0;
    if (!ordinary.plan(0, route.data(), route.size(), 1, fills, hits, evictions) ||
            !zero.plan_with_cpu_share(0, route.data(), route.size(), 1, route.size(), 0.0, 0, plan) ||
            zero_slots != ordinary_slots || zero.key_of != ordinary.key_of ||
            zero.filled_real_slots != ordinary.filled_real_slots || !plan.cpu.empty() ||
            plan.n_hit != hits || plan.n_eviction != evictions || plan.fills.size() != fills.size()) {
        std::fprintf(stderr, "share-zero ClassLRU parity\n"); return false;
    }
    // SER (-1 ids) must be skipped by the share planner: dropped experts never
    // occupy rows, consume CPU-share staging slots, or touch the maps.
    state ser;
    ser.init(1, n_experts, real_rows, kind::lru);
    std::vector<int32_t> ser_slots(n_experts, -1);
    ser.slot_map[0] = ser_slots.data();
    const std::vector<int32_t> ser_seed = {-1, 0, 3, 5};
    if (!ser.plan_with_cpu_share(0, ser_seed.data(), ser_seed.size(), 1, ser_seed.size(), 0.0, 0, plan) ||
            plan.fills.size() != 3 || !plan.cpu.empty() || ser.filled_real_slots != real_rows) {
        std::fprintf(stderr, "SER underfull share plan\n"); return false;
    }
    const std::vector<int32_t> ser_mixed = {-1, 2, -1, 4};
    if (!ser.plan_with_cpu_share(0, ser_mixed.data(), ser_mixed.size(), 1, ser_mixed.size(),
            1.0, zero_rows, plan) || !plan.fills.empty() || plan.cpu.size() != 2 ||
            ser.filled_real_slots != real_rows) {
        std::fprintf(stderr, "SER mixed share plan\n"); return false;
    }
    bool ser_cpu2 = false, ser_cpu4 = false;
    for (const auto & candidate : plan.cpu) {
        ser_cpu2 |= candidate.expert == 2;
        ser_cpu4 |= candidate.expert == 4;
    }
    if (!ser_cpu2 || !ser_cpu4 ||
            plan.gpu_map[2] < real_rows || plan.gpu_map[2] >= real_rows + zero_rows ||
            plan.gpu_map[4] < real_rows || plan.gpu_map[4] >= real_rows + zero_rows ||
            plan.gpu_map[2] == plan.gpu_map[4] ||
            plan.cpu_map[2] != 2 || plan.cpu_map[4] != 4) {
        std::fprintf(stderr, "SER mixed share maps\n"); return false;
    }
    return true;
}

// The async executor implements only the plain LRU bank and never uses
// CPU-share staging slots, so these options must normalize to defaults.
static bool check_async_options() {
    using llama_moe_cache_policy_detail::normalize_async_cache_options;
    enum llama_moe_cache_policy policy = LLAMA_MOE_CACHE_POLICY_CLASSLRU;
    float share = 0.5f;
    normalize_async_cache_options(policy, share);
    if (policy != LLAMA_MOE_CACHE_POLICY_LRU || share != 0.0f) {
        std::fprintf(stderr, "async policy/share were not normalized\n");
        return false;
    }
    policy = LLAMA_MOE_CACHE_POLICY_LRU;
    share = 0.0f;
    normalize_async_cache_options(policy, share);
    if (policy != LLAMA_MOE_CACHE_POLICY_LRU || share != 0.0f) {
        std::fprintf(stderr, "async normalization changed the defaults\n");
        return false;
    }
    double share64 = 0.25;
    policy = LLAMA_MOE_CACHE_POLICY_CLASSLRU;
    normalize_async_cache_options(policy, share64);
    if (policy != LLAMA_MOE_CACHE_POLICY_LRU || share64 != 0.0) {
        std::fprintf(stderr, "async normalization missed the double share\n");
        return false;
    }
    policy = (enum llama_moe_cache_policy) 7;
    share = 0.5f;
    normalize_async_cache_options(policy, share);
    if (policy != (enum llama_moe_cache_policy) 7 || share != 0.0f) {
        std::fprintf(stderr, "async normalization masked an invalid policy\n");
        return false;
    }
    return true;
}

static bool check_layers(const char * name, size_t layers,
        const std::vector<double> & weights, const std::vector<size_t> & capacities,
        const std::vector<size_t> & expected) {
    const auto actual = llama_moe_cache_detail::layer_allocations(layers, weights, capacities);
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "%s: layer allocation count mismatch\n", name);
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "%s: layer allocation %zu was %zu, expected %zu\n",
                    name, i, actual[i], expected[i]);
            return false;
        }
    }
    return true;
}

int main() {
    llama_backend_init();
    bool ok = true;
    if (!check_cache_policy()) { std::fprintf(stderr, "cache policy regression failed\n"); ok = false; }
    if (!check_cpu_share_plan()) { std::fprintf(stderr, "CPU-share plan regression failed\n"); ok = false; }
    if (!check_async_options()) { std::fprintf(stderr, "async option regression failed\n"); ok = false; }
    if (llama_context_default_params().moe_cache_policy != LLAMA_MOE_CACHE_POLICY_LRU) {
        std::fprintf(stderr, "default cache policy is not LRU\n"); ok = false;
    }
    if (llama_context_default_params().moe_cache_cpu_share != 0.0f) {
        std::fprintf(stderr, "default CPU share is not zero\n"); ok = false;
    }
    if (llama_moe_cache_policy_detail::valid((enum llama_moe_cache_policy) 7) ||
            llama_init_from_model(nullptr, [] { auto p = llama_context_default_params();
                p.moe_cache_policy = (enum llama_moe_cache_policy) 7; return p; }()) != nullptr) {
        std::fprintf(stderr, "invalid API cache policy was accepted\n"); ok = false;
    }
    for (float share : {NAN, INFINITY, -0.01f, 1.01f}) {
        if (llama_init_from_model(nullptr, [share] { auto p = llama_context_default_params();
                p.moe_cache_cpu_share = share; return p; }()) != nullptr) {
            std::fprintf(stderr, "invalid API CPU share was accepted\n"); ok = false;
        }
    }
    ok &= check("equal", {0.5f, 1.0f}, {true, true}, {0.5, 0.5});
    ok &= check("asymmetric", {0.25f, 1.0f}, {true, true}, {0.25, 0.75});
    ok &= check("single device", {1.0f}, {true}, {1.0});
    ok &= check("zero fallback", {0.0f, 0.0f}, {true, true}, {0.5, 0.5});
    ok &= check("ineligible device", {0.25f, 1.0f}, {false, true}, {0.0, 1.0});
    ok &= check("fallback with ineligible device", {0.0f, 0.0f}, {true, false}, {1.0, 0.0});

    ok &= check_layers("layers equal", 4, {0.5, 0.5}, {8, 8}, {2, 2});
    ok &= check_layers("layers asymmetric", 8, {0.25, 0.75}, {10, 10}, {2, 6});
    ok &= check_layers("layers largest remainder", 3, {0.5, 0.5}, {10, 10}, {2, 1});
    ok &= check_layers("layers spill past capacity", 5, {0.9, 0.1}, {1, 10}, {1, 4});
    ok &= check_layers("layers capped by total", 100, {1.0, 1.0}, {3, 4}, {3, 4});
    ok &= check_layers("layers skip zero weight", 3, {0.0, 1.0}, {5, 5}, {0, 3});
    ok &= check_layers("layers equal fallback", 3, {0.0, 0.0}, {2, 2}, {2, 1});
    ok &= check_layers("layers spread by -ts", 72,
            {14592.0/62808.0, 23040.0/62808.0, 24576.0/62808.0}, {72, 72, 72}, {17, 27, 28});
    llama_backend_free();
    return ok ? 0 : 1;
}
