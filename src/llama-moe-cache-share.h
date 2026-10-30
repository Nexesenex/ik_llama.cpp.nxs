#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace llama_moe_cache_detail {

inline std::vector<double> device_shares(const std::vector<float> & cumulative,
        const std::vector<bool> & eligible) {
    if (cumulative.size() != eligible.size()) {
        throw std::invalid_argument("MoE cache split metadata has inconsistent device counts");
    }

    std::vector<double> shares(cumulative.size(), 0.0);
    double previous = 0.0;
    for (std::size_t i = 0; i < cumulative.size(); ++i) {
        const double point = std::isfinite(cumulative[i]) ? std::max(0.0, double(cumulative[i])) : previous;
        shares[i] = std::max(0.0, point - previous);
        previous = std::max(previous, point);
    }

    double total = 0.0;
    std::size_t eligible_count = 0;
    for (std::size_t i = 0; i < shares.size(); ++i) {
        if (eligible[i]) {
            total += shares[i];
            ++eligible_count;
        } else {
            shares[i] = 0.0;
        }
    }

    if (eligible_count == 0) {
        return shares;
    }
    if (!std::isfinite(total) || total <= 0.0) {
        const double equal_share = 1.0 / eligible_count;
        for (std::size_t i = 0; i < shares.size(); ++i) {
            shares[i] = eligible[i] ? equal_share : 0.0;
        }
        return shares;
    }

    for (std::size_t i = 0; i < shares.size(); ++i) {
        shares[i] = eligible[i] ? shares[i] / total : 0.0;
    }
    return shares;
}

// Distribute a whole-layer cache budget over groups proportionally to their
// weights, never exceeding each group's capacity. Ties go to the lowest index.
inline std::vector<std::size_t> layer_allocations(std::size_t layers,
        const std::vector<double> & weights, const std::vector<std::size_t> & capacities) {
    if (weights.size() != capacities.size()) {
        throw std::invalid_argument("MoE cache layer allocation metadata has inconsistent group counts");
    }

    std::vector<std::size_t> result(weights.size(), 0);
    std::vector<double> effective(weights.size(), 0.0);
    double total_weight = 0.0;
    std::size_t total_capacity = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        if (capacities[i] == 0) {
            continue;
        }
        effective[i] = std::isfinite(weights[i]) ? std::max(0.0, weights[i]) : 0.0;
        total_weight += effective[i];
        total_capacity += capacities[i];
    }
    if (layers == 0 || total_capacity == 0) {
        return result;
    }

    const std::size_t target = std::min(layers, total_capacity);
    if (total_weight <= 0.0) {
        const std::size_t n_groups = std::count_if(capacities.begin(), capacities.end(),
                [](std::size_t capacity) { return capacity != 0; });
        for (std::size_t i = 0; i < effective.size(); ++i) {
            if (capacities[i]) effective[i] = 1.0 / (double) n_groups;
        }
        total_weight = 1.0;
    }

    std::vector<double> remainders(weights.size(), 0.0);
    std::size_t assigned = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        if (effective[i] <= 0.0) {
            continue;
        }
        const double exact = (double) target * effective[i] / total_weight;
        const std::size_t whole = std::min(capacities[i], (std::size_t) std::floor(exact));
        result[i] = whole;
        assigned += whole;
        remainders[i] = exact - std::floor(exact);
    }
    while (assigned < target) {
        std::size_t best = weights.size();
        double best_remainder = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < weights.size(); ++i) {
            if (effective[i] <= 0.0 || result[i] >= capacities[i]) {
                continue;
            }
            if (remainders[i] > best_remainder) {
                best = i;
                best_remainder = remainders[i];
            }
        }
        if (best == weights.size()) {
            break;
        }
        ++result[best];
        ++assigned;
        remainders[best] -= 1.0;
    }
    return result;
}

} // namespace llama_moe_cache_detail
