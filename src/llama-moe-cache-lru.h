#pragma once

#include "llama-moe-cache-policy.h"

namespace llama_moe_cache_detail {

// Compatibility alias: the slot lifecycle and the replacement policy share a
// single state object so that the blocking planner and the asynchronous
// executor never disagree about pins, generations or slot ownership.
using moe_cache_lru = llama_moe_cache_policy_detail::state;

// The blocking (legacy) cache may only be consulted when the asynchronous
// hybrid executor does not own the layers.  In async mode the scheduler copy
// callback is not installed, so a layer rejected by the executor must run the
// stock FFN graph instead of reading slot maps that only that callback fills.
inline bool legacy_lookup_allowed(bool has_cache, bool async_mode, bool has_lora) {
    return has_cache && !async_mode && !has_lora;
}

} // namespace llama_moe_cache_detail
