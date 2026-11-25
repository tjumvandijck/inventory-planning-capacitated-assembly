#include <iostream>
#include "dynaplex/dynaplexprovider.h"
#include <dynaplex/models/asml_assembly_nonstationary/mdp.h>
#include <dynaplex/retrievemdp.h>
#include "../../lib/models/include/dynaplex/retrievestate.h"
#include <numeric>

#include <boost/math/distributions/students_t.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>

using namespace DynaPlex;


int SaveCSV_3vec(vector<vector<vector<double>>> data, std::string filename, std::string folder_path) {
    std::string file_path = folder_path + filename;
    std::ofstream out_file(file_path);

    for (auto i : data) {
        for (auto j : i) {
            for (auto k : j) {
                out_file << k << ",";
            }
            out_file << std::endl;
        }
    }
    out_file.close();
    return 0;
}

int SaveCSV_2vec(vector<vector<double>> data, std::string filename, std::string folder_path) {
    std::string file_path = folder_path + filename;
    std::ofstream out_file(file_path);

    for (auto i : data) {
        for (auto j : i) {
            out_file << j << ",";
        }
        out_file << std::endl;
    }
    out_file.close();
    return 0;
}

int SaveCSV_vec(vector<double> data, std::string filename, std::string folder_path) {
    std::string file_path = folder_path + filename;
    std::ofstream out_file(file_path);

    for (auto i : data) {
        out_file << i << ",";
    }
    out_file << std::endl;

    out_file.close();
    return 0;
}


struct PairedPercentResult {
    std::size_t n;
    // Positive values mean the candidate has higher cost than the benchmark.
    std::vector<double> percentage_differences;
    double mean_percent_cost_increase;
    double standard_error;
    double ci_lower;
    double ci_upper;
    double t_statistic;
    double two_sided_p_value;
};

// Positive values mean that candidate_cost is higher than benchmark_cost.
static double PercentageCostDifference(
    const double candidate_cost,
    const double benchmark_cost) {
    if (!std::isfinite(benchmark_cost) || !std::isfinite(candidate_cost) ||
        benchmark_cost < 0.0) {
        throw std::invalid_argument(
            "All costs must be finite and benchmark trajectory costs must be non-negative.");
    }
    if (benchmark_cost == 0.0) {
        if (candidate_cost != 0.0) {
            return 100;
            throw std::invalid_argument(
                "A percentage difference is undefined when benchmark cost is zero and candidate cost is not zero.");
        }
        return 0.0;
    }
    return 100.0 * (candidate_cost - benchmark_cost) / benchmark_cost;
}

PairedPercentResult PairedPercentDifferenceTest(
    std::span<const double> percentage_differences,
    double confidence_level = 0.95) {
    if (percentage_differences.size() < 2) {
        throw std::invalid_argument("At least two paired trajectories are required.");
    }
    if (!(confidence_level > 0.0 && confidence_level < 1.0)) {
        throw std::invalid_argument("confidence_level must lie in (0, 1).");
    }

    std::size_t n = 0;
    double mean = 0.0;
    double sum_squared_deviations = 0.0;
    std::vector<double> saved_differences;
    saved_differences.reserve(percentage_differences.size());
    for (const double difference : percentage_differences) {
        if (!std::isfinite(difference)) {
            throw std::invalid_argument("Percentage differences must be finite.");
        }
        saved_differences.push_back(difference);
        ++n;
        const double delta = difference - mean;
        mean += delta / static_cast<double>(n);
        sum_squared_deviations += delta * (difference - mean);
    }

    const double standard_error = std::sqrt(
        sum_squared_deviations / static_cast<double>(n - 1)) /
        std::sqrt(static_cast<double>(n));
    const boost::math::students_t_distribution<double> t_dist(
        static_cast<double>(n - 1));
    const double alpha = 1.0 - confidence_level;
    const double critical_value = boost::math::quantile(
        boost::math::complement(t_dist, alpha / 2.0));
    const double t_statistic = standard_error > 0.0 ? mean / standard_error : 0.0;
    const double p_value = standard_error > 0.0
        ? 2.0 * boost::math::cdf(
            boost::math::complement(t_dist, std::abs(t_statistic)))
        : (mean == 0.0 ? 1.0 : 0.0);

    return {
        n,
        std::move(saved_differences),
        mean,
        standard_error,
        mean - critical_value * standard_error,
        mean + critical_value * standard_error,
        t_statistic,
        p_value
    };
}

// candidate_costs[r] and benchmark_costs[r] must come from the same trajectory r.
PairedPercentResult PairedPercentCostTest(
    std::span<const double> candidate_costs,
    std::span<const double> benchmark_costs,
    double confidence_level = 0.95)
{
    if (candidate_costs.size() != benchmark_costs.size()) {
        throw std::invalid_argument("Paired samples must have equal lengths.");
    }
    if (candidate_costs.size() < 2) {
        throw std::invalid_argument("At least two paired trajectories are required.");
    }
    if (!(confidence_level > 0.0 && confidence_level < 1.0)) {
        throw std::invalid_argument("confidence_level must lie in (0, 1).");
    }

    // Welford's algorithm for the mean and sample variance of percentage changes.
    std::size_t n = 0;
    double mean = 0.0;
    double sum_squared_deviations = 0.0;
    std::vector<double> percentage_differences;
    percentage_differences.reserve(candidate_costs.size());

    for (std::size_t r = 0; r < candidate_costs.size(); ++r) {
        const double benchmark = benchmark_costs[r];
        const double candidate = candidate_costs[r];

        const double d = PercentageCostDifference(candidate, benchmark);
        percentage_differences.push_back(d);
        ++n;
        const double delta = d - mean;
        mean += delta / static_cast<double>(n);
        sum_squared_deviations += delta * (d - mean);
    }

    const double sample_sd =
        std::sqrt(sum_squared_deviations / static_cast<double>(n - 1));
    const double standard_error = sample_sd / std::sqrt(static_cast<double>(n));

    const double df = static_cast<double>(n - 1);
    const boost::math::students_t_distribution<double> t_dist(df);
    const double alpha = 1.0 - confidence_level;
    const double critical_value =
        boost::math::quantile(boost::math::complement(t_dist, alpha / 2.0));

    const double t_statistic =
        standard_error > 0.0 ? mean / standard_error : 0.0;
    const double p_value =
        standard_error > 0.0
        ? 2.0 * boost::math::cdf(
            boost::math::complement(t_dist, std::abs(t_statistic)))
        : (mean == 0.0 ? 1.0 : 0.0);

    return {
        n,
        std::move(percentage_differences),
        mean,
        standard_error,
        mean - critical_value * standard_error,
        mean + critical_value * standard_error,
        t_statistic,
        p_value
    };
}

static std::string GetNeuralNetworkLocation(
    const DynaPlex::VarGroup& mdp_config,
    const DynaPlex::VarGroup& training_config) {
    std::string id;
    int64_t number_of_nodes;
    int64_t backlog_penalty;
    int64_t N;
    int64_t M;
    int64_t num_gens;
    bool assembly;
    bool stationary;
    bool feature_engineering;
    bool zeros_max_system_output;
    bool homogeneous_capacity;
    bool ATO;

    mdp_config.Get("id", id);
    mdp_config.Get("NumberofNodes", number_of_nodes);
    mdp_config.Get("p", backlog_penalty);
    mdp_config.Get("assembly", assembly);
    mdp_config.Get("stationary", stationary);
    mdp_config.GetOrDefault("featureEngineering", feature_engineering, false);
    mdp_config.GetOrDefault("zerosMaxSystemOutput", zeros_max_system_output, false);
    mdp_config.GetOrDefault("homogeneous_capacity", homogeneous_capacity, false);
    mdp_config.GetOrDefault("ATO", ATO, false);
    training_config.Get("N", N);
    training_config.Get("M", M);
    training_config.Get("num_gens", num_gens);

    return id +
        "_NN_" + std::to_string(number_of_nodes) +
        "_p" + std::to_string(backlog_penalty) +
        (assembly ? "_A" : "_S") +
        (stationary ? "_S" : "_NS") +
        "_N" + std::to_string(N) +
        "_M" + std::to_string(M) +
        "_G" + std::to_string(num_gens) +
        (feature_engineering ? "_FE" : "_NoFE") +
        (zeros_max_system_output ? "_DivMean" : "_lowN") + 
        (homogeneous_capacity ? "_eqC" : "") +
        (ATO ? "_ATO" : "");
}

struct MultiplePolicyTrajectoryResults {
    // Indexed [policy][trajectory]. These are always returned and preserve
    // the trajectory pairing needed for later paired tests.
    std::vector<std::vector<double>> policy_costs;

    // Post-warm-up wall-clock milliseconds, indexed [policy][trajectory].
    std::vector<std::vector<double>> trajectory_evaluation_milliseconds;

    // [policy][trajectory], relative to percentage_difference_benchmark_policy_index.
    // Positive values mean that this policy has higher cost than the benchmark.
    std::vector<std::vector<double>> percentage_differences;
    int64_t percentage_difference_benchmark_policy_index = -1;

    // The fields below are populated only when trackStatistics is true.
    // [policy][trajectory][vertical position][node]
    std::vector<std::vector<std::vector<std::vector<double>>>>
        avg_production_per_node_per_vertpos_per_trajectory_per_policy;
    // [policy][trajectory][vertical position], number of post-warm-up event
    // periods spent at each vertical position.
    std::vector<std::vector<std::vector<double>>>
        num_periods_per_vertpos_per_policy_per_traj;
    // [policy][trajectory][vertical position]. This distinguishes a vertical
    // position that was not reached from one that was reached with zero WIP.
    std::vector<std::vector<std::vector<bool>>>
        vertical_position_reached_per_trajectory_per_policy;
    // [policy][trajectory][node]
    std::vector<std::vector<std::vector<double>>>
        avg_inv_per_node_per_trajectory_per_policy;

    // Per-vertical-position means across trajectories that reached that
    // position. Unreached trajectories do not contribute zero observations.
    // [policy][vertical position][node]
    std::vector<std::vector<std::vector<double>>>
        avg_production_per_node_per_vertpos_per_policy;
    // [policy][node]
    std::vector<std::vector<double>> avg_inv_per_node_per_policy;
    // [policy][trajectory]
    std::vector<std::vector<double>>
        avg_perc_periods_with_backlog_per_policy_per_traj;
    // [policy][trajectory], total positive downstream backlog units over all
    // post-warm-up periods (sum of arriving - downstream_on_hand when positive).
    std::vector<std::vector<double>>
        total_backlog_units_per_policy_per_traj;
    // [policy][trajectory], number of event periods observed after warm-up.
    std::vector<std::vector<double>>
        num_periods_after_warmup_per_policy_per_traj;
    std::vector<std::vector<double>>
        avg_backlog_costs_per_policy_per_traj;
    std::vector<std::vector<double>>
        avg_inv_costs_per_policy_per_traj;
    // [policy], averaged across trajectories
    std::vector<double> avg_backlog_costs_per_policy;
    std::vector<double> avg_inv_costs_per_policy;
};

// Persist the raw, paired trajectory-level outputs.  The CSV row ordering is
// the same as the vector ordering documented in MultiplePolicyTrajectoryResults:
// policy first, then trajectory (and, where applicable, vertical position and
// node).  The metadata file records the dimensions needed to read flat rows.
static void SaveTrajectoryTestingMultiplePoliciesResults(
    const MultiplePolicyTrajectoryResults& results,
    bool trackStatistics,
    const std::string& filename_run,
    const std::string& output_directory,
    const bool feature_engineering) {
    if (results.policy_costs.empty()) {
        throw std::invalid_argument(
            "Cannot save trajectory results without policy costs.");
    }

    const std::filesystem::path output_path(output_directory);
    std::filesystem::create_directories(output_path);

    const auto output_file = [&output_path, &filename_run](const std::string& name) {
        return output_path / (name + "_" + filename_run + ".csv");
        };

    const auto write_2vec = [](const std::vector<std::vector<double>>& values,
        const std::filesystem::path& path) {
            std::ofstream out(path);
            if (!out) {
                throw std::runtime_error("Could not open output file: " + path.string());
            }
            for (const auto& policy_values : values) {
                for (std::size_t trajectory = 0;
                    trajectory < policy_values.size(); ++trajectory) {
                    if (trajectory > 0) {
                        out << ',';
                    }
                    out << policy_values[trajectory];
                }
                out << '\n';
            }
        };

    const auto write_1vec = [](const std::vector<double>& values,
        const std::filesystem::path& path) {
            std::ofstream out(path);
            if (!out) {
                throw std::runtime_error("Could not open output file: " + path.string());
            }
            for (std::size_t policy = 0; policy < values.size(); ++policy) {
                if (policy > 0) {
                    out << ',';
                }
                out << values[policy];
            }
            out << '\n';
        };

    const auto write_3vec = [](const std::vector<std::vector<std::vector<double>>>& values,
        const std::filesystem::path& path) {
            std::ofstream out(path);
            if (!out) {
                throw std::runtime_error("Could not open output file: " + path.string());
            }
            for (const auto& policy_values : values) {
                for (const auto& trajectory_values : policy_values) {
                    for (std::size_t node = 0; node < trajectory_values.size(); ++node) {
                        if (node > 0) {
                            out << ',';
                        }
                        out << trajectory_values[node];
                    }
                    out << '\n';
                }
            }
        };

    const auto write_4vec = [](const std::vector<std::vector<std::vector<std::vector<double>>>>& values,
        const std::filesystem::path& path) {
            std::ofstream out(path);
            if (!out) {
                throw std::runtime_error("Could not open output file: " + path.string());
            }
            for (const auto& policy_values : values) {
                for (const auto& trajectory_values : policy_values) {
                    for (const auto& vertical_position_values : trajectory_values) {
                        for (std::size_t node = 0;
                            node < vertical_position_values.size(); ++node) {
                            if (node > 0) {
                                out << ',';
                            }
                            out << vertical_position_values[node];
                        }
                        out << '\n';
                    }
                }
            }
        };

    // These results do not depend on diagnostic statistic tracking.
    write_2vec(results.policy_costs, output_file("policy_costs"));
    write_2vec(results.trajectory_evaluation_milliseconds,
        output_file("trajectory_evaluation_milliseconds"));
    if (results.percentage_differences.empty() ||
        results.percentage_difference_benchmark_policy_index < 0) {
        throw std::invalid_argument(
            "Cannot save trajectory results without percentage differences and their benchmark policy index.");
    }
    write_2vec(results.percentage_differences,
        output_file("percentage_differences"));

    const std::size_t num_policies = results.policy_costs.size();
    const std::size_t num_trajectories = results.policy_costs.front().size();
    std::size_t num_vertical_positions = 0;
    std::size_t num_nodes = 0;
    if (trackStatistics) {
        if (results.avg_production_per_node_per_vertpos_per_trajectory_per_policy.empty() ||
            results.num_periods_per_vertpos_per_policy_per_traj.empty() ||
            results.avg_inv_per_node_per_trajectory_per_policy.empty() ||
            results.avg_production_per_node_per_vertpos_per_policy.empty() ||
            results.avg_inv_per_node_per_policy.empty() ||
            results.avg_perc_periods_with_backlog_per_policy_per_traj.empty() ||
            results.total_backlog_units_per_policy_per_traj.empty() ||
            results.num_periods_after_warmup_per_policy_per_traj.empty() ||
            results.avg_backlog_costs_per_policy_per_traj.empty() ||
            results.avg_inv_costs_per_policy_per_traj.empty() ||
            results.avg_backlog_costs_per_policy.empty() ||
            results.avg_inv_costs_per_policy.empty()) {
            throw std::invalid_argument(
                "trackStatistics was requested, but the trajectory results do not contain statistics.");
        }

		std::string filename_avg_production = feature_engineering ? "avg_production_per_node_per_vertpos_per_trajectory_per_policy_FE" : "avg_production_per_node_per_vertpos_per_trajectory_per_policy_noFE";
		std::string filename_num_periods_per_vertpos = feature_engineering ? "num_periods_per_vertpos_per_policy_per_traj_FE" : "num_periods_per_vertpos_per_policy_per_traj_noFE";
		std::string filename_avg_inv_per_node = feature_engineering ? "avg_inv_per_node_per_trajectory_per_policy_FE" : "avg_inv_per_node_per_trajectory_per_policy_noFE";
		std::string filename_avg_perc_backlog = feature_engineering ? "avg_perc_periods_with_backlog_per_policy_per_traj_FE" : "avg_perc_periods_with_backlog_per_policy_per_traj_noFE";
		std::string filename_total_backlog_units = feature_engineering ? "total_backlog_units_per_policy_per_traj_FE" : "total_backlog_units_per_policy_per_traj_noFE";
		std::string filename_num_periods_after_warmup = feature_engineering ? "num_periods_after_warmup_per_policy_per_traj_FE" : "num_periods_after_warmup_per_policy_per_traj_noFE";
		std::string filename_avg_backlog = feature_engineering ? "avg_backlog_costs_per_policy_per_traj_FE" : "avg_backlog_costs_per_policy_per_traj_noFE";
		std::string filename_avg_inv = feature_engineering ? "avg_inv_costs_per_policy_per_traj_FE" : "avg_inv_costs_per_policy_per_traj_noFE";
		std::string filename_avg_production_per_policy = feature_engineering ? "avg_production_per_node_per_vertpos_per_policy_FE" : "avg_production_per_node_per_vertpos_per_policy_noFE";
		std::string filename_avg_inv_per_node_per_policy = feature_engineering ? "avg_inv_per_node_per_policy_FE" : "avg_inv_per_node_per_policy_noFE";
		std::string filename_avg_backlog_per_policy = feature_engineering ? "avg_backlog_costs_per_policy_FE" : "avg_backlog_costs_per_policy_noFE";
		std::string filename_avg_inv_costs_per_policy = feature_engineering ? "avg_inv_costs_per_policy_FE" : "avg_inv_costs_per_policy_noFE";

        // TestInventoryperNode writes this KPI with vertical position as the
        // outer dimension: [vertical position][policy][node]. Preserve that
        // CSV row order for compatibility with its plotting scripts.
        const std::size_t num_vertical_positions_for_export =
            results.avg_production_per_node_per_vertpos_per_policy.front().size();
        const std::size_t num_nodes_for_export =
            results.avg_production_per_node_per_vertpos_per_policy.front().front().size();
        std::vector<std::vector<std::vector<double>>>
            avg_production_per_vertpos_per_policy(
                num_vertical_positions_for_export,
                std::vector<std::vector<double>>(
                    num_policies, std::vector<double>(num_nodes_for_export, 0.0)));
        for (std::size_t p = 0; p < num_policies; ++p) {
            for (std::size_t v = 0; v < num_vertical_positions_for_export; ++v) {
                for (std::size_t n = 0; n < num_nodes_for_export; ++n) {
                    avg_production_per_vertpos_per_policy[v][p][n] =
                        results.avg_production_per_node_per_vertpos_per_policy[p][v][n];
                }
            }
        }

        write_4vec(results.avg_production_per_node_per_vertpos_per_trajectory_per_policy,
            output_file(filename_avg_production));
        write_3vec(results.num_periods_per_vertpos_per_policy_per_traj,
            output_file(filename_num_periods_per_vertpos));
        write_3vec(results.avg_inv_per_node_per_trajectory_per_policy,
            output_file(filename_avg_inv_per_node));
        write_3vec(avg_production_per_vertpos_per_policy,
            output_file(filename_avg_production_per_policy));
        write_2vec(results.avg_inv_per_node_per_policy,
            output_file(filename_avg_inv_per_node_per_policy));
        write_2vec(results.avg_perc_periods_with_backlog_per_policy_per_traj,
            output_file(filename_avg_perc_backlog));
        write_2vec(results.total_backlog_units_per_policy_per_traj,
            output_file(filename_total_backlog_units));
        write_2vec(results.num_periods_after_warmup_per_policy_per_traj,
            output_file(filename_num_periods_after_warmup));
        write_2vec(results.avg_backlog_costs_per_policy_per_traj,
            output_file(filename_avg_backlog));
        write_2vec(results.avg_inv_costs_per_policy_per_traj,
            output_file(filename_avg_inv));
        write_1vec(results.avg_backlog_costs_per_policy,
            output_file(filename_avg_backlog_per_policy));
        write_1vec(results.avg_inv_costs_per_policy,
            output_file(filename_avg_inv_costs_per_policy));

        num_vertical_positions =
            results.avg_production_per_node_per_vertpos_per_trajectory_per_policy
            .front().front().size();
        num_nodes =
            results.avg_inv_per_node_per_trajectory_per_policy.front().front().size();
    }
    std::ofstream metadata(output_file("trajectory_results_metadata"));
    if (!metadata) {
        throw std::runtime_error("Could not open trajectory-results metadata file.");
    }
    metadata << "num_policies,num_trajectories,benchmark_policy_index,num_vertical_positions,num_nodes,statistics_tracked\n"
        << num_policies << ',' << num_trajectories << ','
        << results.percentage_difference_benchmark_policy_index << ','
        << num_vertical_positions << ',' << num_nodes << ','
        << (trackStatistics ? 1 : 0) << '\n';
}

MultiplePolicyTrajectoryResults TrajectoryTestingMultiplePolicies(
    const DynaPlex::MDP& mdp,
    const std::vector<DynaPlex::Policy>& policies,
    int64_t numTrajectories,
    int64_t horizonLength,
    int64_t warmup_periods,
    bool trackStatistics,
    int64_t percentage_difference_benchmark_policy_index,
    const DynaPlex::Policy& warmup_policy = {}) {
    if (!mdp) {
        throw DynaPlex::Error("TrajectoryTestingMultiplePolicies: mdp should not be null.");
    }
    if (policies.empty()) {
        throw DynaPlex::Error("TrajectoryTestingMultiplePolicies: provide at least one policy.");
    }
    if (percentage_difference_benchmark_policy_index < 0 ||
        percentage_difference_benchmark_policy_index >=
        static_cast<int64_t>(policies.size())) {
        throw DynaPlex::Error(
            "TrajectoryTestingMultiplePolicies: percentage-difference benchmark policy index is out of range.");
    }
    if (numTrajectories < 2) {
        throw DynaPlex::Error("TrajectoryTestingMultiplePolicies: at least two trajectories are required.");
    }
    if (warmup_periods < 0 || horizonLength < 0) {
        throw DynaPlex::Error("TrajectoryTestingMultiplePolicies: horizons must be non-negative.");
    }

    MultiplePolicyTrajectoryResults results;

    // policy_costs[p][t] is the measurement-period cost of policy p on
    // trajectory t. All policies use the same event-stream seed for a given
    // t, which makes the later comparisons paired.
    results.policy_costs = std::vector<std::vector<double>>(
        policies.size(), std::vector<double>(numTrajectories));
    results.trajectory_evaluation_milliseconds = std::vector<std::vector<double>>(
        policies.size(), std::vector<double>(numTrajectories));
    results.percentage_differences = std::vector<std::vector<double>>(
        policies.size(), std::vector<double>(numTrajectories));
    results.percentage_difference_benchmark_policy_index =
        percentage_difference_benchmark_policy_index;

    double cum_traj_costs = 0;

    using underlying_mdp = DynaPlex::Models::asml_assembly_nonstationary::MDP;
    std::shared_ptr<const underlying_mdp> eval_mdp;
    underlying_mdp::State init_state;
    std::vector<std::vector<int64_t>> max_inventory_position_per_node_per_policy;
    std::vector<int64_t> max_downstream_backlog_per_policy;
    if (trackStatistics) {
        eval_mdp = DynaPlex::RetrieveMDP<underlying_mdp>(
            const_cast<DynaPlex::MDP&>(mdp));
        init_state = eval_mdp->GetInitialState();
        max_inventory_position_per_node_per_policy.assign(
            policies.size(), std::vector<int64_t>(eval_mdp->NumberofNodes, 0));
        max_downstream_backlog_per_policy.assign(policies.size(), 0);

        const int64_t num_vertical_positions =
            eval_mdp->maxTree - eval_mdp->startMeanDem + 1;
        results.avg_production_per_node_per_vertpos_per_trajectory_per_policy =
            std::vector<std::vector<std::vector<std::vector<double>>>>(
                policies.size(),
                std::vector<std::vector<std::vector<double>>>(
                    numTrajectories,
                    std::vector<std::vector<double>>(
                        num_vertical_positions,
                        std::vector<double>(eval_mdp->NumberofNodes, 0.0))));
        results.num_periods_per_vertpos_per_policy_per_traj =
            std::vector<std::vector<std::vector<double>>>(
                policies.size(),
                std::vector<std::vector<double>>(
                    numTrajectories,
                    std::vector<double>(num_vertical_positions, 0.0)));
        results.vertical_position_reached_per_trajectory_per_policy =
            std::vector<std::vector<std::vector<bool>>>(
                policies.size(),
                std::vector<std::vector<bool>>(
                    numTrajectories,
                    std::vector<bool>(num_vertical_positions, false)));
        results.avg_inv_per_node_per_trajectory_per_policy =
            std::vector<std::vector<std::vector<double>>>(
                policies.size(),
                std::vector<std::vector<double>>(
                    numTrajectories,
                    std::vector<double>(eval_mdp->NumberofNodes, 0.0)));
        results.avg_perc_periods_with_backlog_per_policy_per_traj =
            std::vector<std::vector<double>>(
                policies.size(), std::vector<double>(numTrajectories, 0.0));
        results.total_backlog_units_per_policy_per_traj =
            std::vector<std::vector<double>>(
                policies.size(), std::vector<double>(numTrajectories, 0.0));
        results.num_periods_after_warmup_per_policy_per_traj =
            std::vector<std::vector<double>>(
                policies.size(), std::vector<double>(numTrajectories, 0.0));
        results.avg_backlog_costs_per_policy_per_traj =
            std::vector<std::vector<double>>(
                policies.size(), std::vector<double>(numTrajectories, 0.0));
        results.avg_inv_costs_per_policy_per_traj =
            std::vector<std::vector<double>>(
                policies.size(), std::vector<double>(numTrajectories, 0.0));
    }

    constexpr int64_t rng_seed = 19;
    const int64_t startingIndex = 0;

    for (int64_t t = startingIndex; t < numTrajectories + startingIndex; ++t) {
        for (size_t p = 0; p < policies.size(); ++p) {
            const auto& policy = policies[p];
            if (!policy) {
                throw DynaPlex::Error("TrajectoryTestingMultiplePolicies: policy should not be null.");
            }
            const auto& policy_for_warmup = warmup_policy ? warmup_policy : policy;

            Trajectory trajectory{};
            trajectory.RNGProvider.SeedEventStreams(true, rng_seed, t);
            mdp->InitiateState({ &trajectory, 1 });
            auto& category = trajectory.Category;
            underlying_mdp::State* state = nullptr;
            if (trackStatistics) {
                state = &DynaPlex::RetrieveState<underlying_mdp::State>(
                    trajectory.GetState());
            }

            // This MDP is finite-horizon but explicitly supports a warm-up
            // phase. AdaptWarmUp prevents demand-tree movement and final-state
            // transitions while the state is being warmed up.
            mdp->AdaptWarmUp({ &trajectory, 1 });
            for (int64_t warmup_event = 0; warmup_event < warmup_periods; ++warmup_event) {
                while (category.IsAwaitAction()) {
                    policy_for_warmup->SetAction({ &trajectory, 1 });
                    //policy->SetAction({ &trajectory, 1 });

                    mdp->IncorporateAction({ &trajectory, 1 });
                }
                if (!category.IsAwaitEvent()) {
                    throw DynaPlex::Error(
                        "TrajectoryTestingMultiplePolicies: warm-up trajectory is neither awaiting an event nor an action.");
                }
                mdp->IncorporateEvent({ &trajectory, 1 });
            }
            mdp->AdaptWarmUp({ &trajectory, 1 });

            // Warm-up costs are excluded but the warmed-up inventory state is
            // retained. This MDP keeps the category both in Trajectory and in
            // State, so reset both to the initial AwaitAction category.
            trajectory.CumulativeReturn = 0.0;
            if (trackStatistics) {
                trajectory.Category = init_state.cat;
                state->cat = init_state.cat;
            }

            // To track statistics
            int64_t measurement_events = 0;
            int64_t periods_with_backlog = 0;
            std::vector<double> avg_inventory_per_node_per_trajectory;
            std::vector<std::vector<double>>
                avg_production_per_node_per_vertpos_per_trajectory;
            std::vector<int64_t> count_vertpos;
            double total_inventory_costs_per_trajectory = 0.0;
            double total_backlog_costs_per_trajectory = 0.0;
            double total_backlog_units_per_trajectory = 0.0;
            if (trackStatistics) {
                avg_inventory_per_node_per_trajectory.assign(
                    eval_mdp->NumberofNodes, 0.0);
                const int64_t num_vertical_positions =
                    eval_mdp->maxTree - eval_mdp->startMeanDem + 1;
                avg_production_per_node_per_vertpos_per_trajectory.assign(
                    num_vertical_positions,
                    std::vector<double>(eval_mdp->NumberofNodes, 0.0));
                count_vertpos.assign(num_vertical_positions, 0);
            }


            // Running the evaluation after warm-up
            const auto evaluation_start = std::chrono::steady_clock::now();
            while (!category.IsFinal()) {
                while (category.IsAwaitAction()) {
                    const int64_t decided_node = trackStatistics ? state->currentNode : -1;
                    policy->SetAction({ &trajectory, 1 });
                    mdp->IncorporateAction({ &trajectory, 1 });
                    if (trackStatistics) {
                        int64_t inventory_position = 0;
                        for (int64_t lead_time_position = 0;
                            lead_time_position <= eval_mdp->Leadtime_vec.at(decided_node);
                            ++lead_time_position) {
                            inventory_position += state->StateVectors.at(
                                eval_mdp->frontposition_vec.at(decided_node) +
                                lead_time_position);
                        }
                        max_inventory_position_per_node_per_policy[p][decided_node] =
                            std::max(
                                max_inventory_position_per_node_per_policy[p][decided_node],
                                inventory_position);
                    }
                }
                if (category.IsFinal()) {
                    break;
                }
                if (!category.IsAwaitEvent()) {
                    throw DynaPlex::Error(
                        "TrajectoryTestingMultiplePolicies: trajectory is neither final, awaiting an event, nor awaiting an action.");
                }
                if (horizonLength > 0 && measurement_events >= horizonLength) {
                    throw DynaPlex::Error(
                        "TrajectoryTestingMultiplePolicies: trajectory exceeded horizonLength before reaching its final state.");
                }

                // Record at the same event-level observation point as
                // TestInventoryperNode. Keep the vertical position from
                // before IncorporateEvent so its numerator and denominator
                // refer to the same demand-tree position.
                int64_t arriving = 0;
                int64_t vertical_position = 0;
                if (trackStatistics) {
                    vertical_position = state->verticalpos;
                    ++count_vertpos.at(vertical_position);

                    // Non-downstream nodes: on-hand inventory for the
                    // trajectory average, WIP for the vertical-position view.
                    for (int64_t n = 0; n < eval_mdp->NumberofNodes - 1; ++n) {
                        avg_inventory_per_node_per_trajectory.at(n) +=
                            state->StateVectors.at(eval_mdp->frontposition_vec.at(n));
                        avg_production_per_node_per_vertpos_per_trajectory
                            .at(vertical_position).at(n) +=
                            state->StateVectors.at(eval_mdp->backposition_vec.at(n));
                        total_inventory_costs_per_trajectory +=
                            state->StateVectors.at(eval_mdp->frontposition_vec.at(n)) *
                            eval_mdp->h_vec.at(n);
                    }
                    const int64_t downstream_node = eval_mdp->NumberofNodes - 1;

                    avg_production_per_node_per_vertpos_per_trajectory
                        .at(vertical_position).at(downstream_node) += state->StateVectors.at(
                            eval_mdp->backposition_vec.at(downstream_node));
                    arriving = state->StateVectors.at(
                        eval_mdp->frontposition_vec.at(downstream_node) + 1);
                }

                mdp->IncorporateEvent({ &trajectory, 1 });
                ++measurement_events;

                if (trackStatistics) {
                    const int64_t downstream_node = eval_mdp->NumberofNodes - 1;
                    const int64_t downstream_on_hand = state->StateVectors.at(
                        eval_mdp->frontposition_vec.at(downstream_node));


                    if (downstream_on_hand < arriving) {
                        const int64_t backlog_units = arriving - downstream_on_hand;
                        ++periods_with_backlog;
                        max_downstream_backlog_per_policy[p] = std::max(
                            max_downstream_backlog_per_policy[p], backlog_units);
                        total_backlog_units_per_trajectory += backlog_units;
                        total_backlog_costs_per_trajectory +=
                            eval_mdp->p * backlog_units;
                    }
                    else {
                        const int64_t downstream_inventory = std::max(
                            downstream_on_hand - arriving, int64_t{ 0 });
                        avg_inventory_per_node_per_trajectory.at(downstream_node) +=
                            downstream_inventory;
                        total_inventory_costs_per_trajectory +=
                            downstream_inventory * eval_mdp->h_vec.at(downstream_node);
                    }

     //               if (std::abs(
     //                   (total_inventory_costs_per_trajectory + total_backlog_costs_per_trajectory)
     //                   - trajectory.CumulativeReturn
     //               ) > 1e-6) {
     //                   throw DynaPlex::Error(
					//		"TrajectoryTestingMultiplePolicies: cumulative return does not match inventory and backlog costs.");
					//}
                }
            }

            results.policy_costs[p][t] = trajectory.CumulativeReturn;
            results.trajectory_evaluation_milliseconds[p][t] =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - evaluation_start).count();
            if (trackStatistics) {
                if (measurement_events == 0) {
                    throw DynaPlex::Error(
                        "TrajectoryTestingMultiplePolicies: measurement trajectory contains no events.");
                }
                for (int64_t n = 0; n < eval_mdp->NumberofNodes; ++n) {
                    results.avg_inv_per_node_per_trajectory_per_policy[p][t][n] =
                        avg_inventory_per_node_per_trajectory[n] /
                        static_cast<double>(measurement_events);
                }
                for (size_t v = 0; v < count_vertpos.size(); ++v) {
                    results.num_periods_per_vertpos_per_policy_per_traj[p][t][v] =
                        static_cast<double>(count_vertpos[v]);
                    if (count_vertpos[v] == 0) {
                        continue;
                    }
                    results.vertical_position_reached_per_trajectory_per_policy[p][t][v] = true;
                    for (int64_t n = 0; n < eval_mdp->NumberofNodes; ++n) {
                        results
                            .avg_production_per_node_per_vertpos_per_trajectory_per_policy[p][t][v][n] =
                            avg_production_per_node_per_vertpos_per_trajectory[v][n] /
                            static_cast<double>(count_vertpos[v]);
                    }
                }
                results.avg_perc_periods_with_backlog_per_policy_per_traj[p][t] =
                    static_cast<double>(periods_with_backlog) /
                    static_cast<double>(measurement_events);
                results.total_backlog_units_per_policy_per_traj[p][t] =
                    total_backlog_units_per_trajectory;
                results.num_periods_after_warmup_per_policy_per_traj[p][t] =
                    static_cast<double>(measurement_events);
                results.avg_inv_costs_per_policy_per_traj[p][t] =
                    total_inventory_costs_per_trajectory;
                results.avg_backlog_costs_per_policy_per_traj[p][t] =
                    total_backlog_costs_per_trajectory;
            }
        }

        // Compute all policy differences only after their costs for this
        // common-random-number trajectory have been recorded.
        const double benchmark_cost = results.policy_costs
            .at(percentage_difference_benchmark_policy_index).at(t);
        for (std::size_t p = 0; p < policies.size(); ++p) {
            results.percentage_differences[p][t] = PercentageCostDifference(
                results.policy_costs[p][t], benchmark_cost);
        }
    }

    if (trackStatistics) {
        const std::size_t num_policies = policies.size();
        const std::size_t num_trajectories = static_cast<std::size_t>(numTrajectories);
        const std::size_t num_vertical_positions =
            results.avg_production_per_node_per_vertpos_per_trajectory_per_policy
                .front().front().size();
        const std::size_t num_nodes = static_cast<std::size_t>(eval_mdp->NumberofNodes);

        results.avg_production_per_node_per_vertpos_per_policy.assign(
            num_policies,
            std::vector<std::vector<double>>(
                num_vertical_positions, std::vector<double>(num_nodes, 0.0)));
        results.avg_inv_per_node_per_policy.assign(
            num_policies, std::vector<double>(num_nodes, 0.0));
        results.avg_backlog_costs_per_policy.assign(num_policies, 0.0);
        results.avg_inv_costs_per_policy.assign(num_policies, 0.0);

        for (std::size_t p = 0; p < num_policies; ++p) {
            std::vector<std::size_t> trajectories_reaching_vertical_position(
                num_vertical_positions, 0);
            for (std::size_t t = 0; t < num_trajectories; ++t) {
                for (std::size_t n = 0; n < num_nodes; ++n) {
                    results.avg_inv_per_node_per_policy[p][n] +=
                        results.avg_inv_per_node_per_trajectory_per_policy[p][t][n];
                }
                results.avg_backlog_costs_per_policy[p] +=
                    results.avg_backlog_costs_per_policy_per_traj[p][t];
                results.avg_inv_costs_per_policy[p] +=
                    results.avg_inv_costs_per_policy_per_traj[p][t];
                for (std::size_t v = 0; v < num_vertical_positions; ++v) {
                    if (!results.vertical_position_reached_per_trajectory_per_policy[p][t][v]) {
                        continue;
                    }
                    ++trajectories_reaching_vertical_position[v];
                    for (std::size_t n = 0; n < num_nodes; ++n) {
                        results.avg_production_per_node_per_vertpos_per_policy[p][v][n] +=
                            results
                                .avg_production_per_node_per_vertpos_per_trajectory_per_policy[p][t][v][n];
                    }
                }
            }

            for (std::size_t n = 0; n < num_nodes; ++n) {
                results.avg_inv_per_node_per_policy[p][n] /=
                    static_cast<double>(num_trajectories);
            }
            results.avg_backlog_costs_per_policy[p] /=
                static_cast<double>(num_trajectories);
            results.avg_inv_costs_per_policy[p] /=
                static_cast<double>(num_trajectories);
            for (std::size_t v = 0; v < num_vertical_positions; ++v) {
                if (trajectories_reaching_vertical_position[v] == 0) {
                    continue;
                }
                for (std::size_t n = 0; n < num_nodes; ++n) {
                    results.avg_production_per_node_per_vertpos_per_policy[p][v][n] /=
                        static_cast<double>(trajectories_reaching_vertical_position[v]);
                }
            }
        }
    }

    if (trackStatistics) {
        std::cout << " ----- Maximum post-decision inventory positions and backlog -----"
            << std::endl;
        for (std::size_t p = 0; p < policies.size(); ++p) {
            std::cout << "Policy " << p
                << ": maximum downstream backlog = "
                << max_downstream_backlog_per_policy[p]
                << ", maximum inventory position per node = [";
            for (int64_t n = 0; n < eval_mdp->NumberofNodes; ++n) {
                if (n > 0) {
                    std::cout << ", ";
                }
                std::cout << max_inventory_position_per_node_per_policy[p][n];
            }
            std::cout << "]" << std::endl;
        }
    }

    return results;
}

// Uses the same evaluation configuration as PolicyComparer. For this
// finite-horizon MDP, max_periods_until_error (or, if absent,
// periods_per_trajectory) is a safety cap rather than a fixed horizon.
MultiplePolicyTrajectoryResults TrajectoryTestingMultiplePolicies(
    const DynaPlex::MDP& mdp,
    const std::vector<DynaPlex::Policy>& policies,
    const DynaPlex::VarGroup& test_config,
    bool trackStatistics,
    int64_t percentage_difference_benchmark_policy_index,
    const DynaPlex::Policy& warmup_policy = {}) {
    int64_t numTrajectories;
    int64_t warmupPeriods;
    int64_t horizonLength;

    test_config.Get("number_of_trajectories", numTrajectories);
    test_config.GetOrDefault("warmup_periods", warmupPeriods, 0);
    test_config.GetOrDefault("max_periods_until_error", horizonLength, 0);
    if (horizonLength == 0) {
        test_config.GetOrDefault("periods_per_trajectory", horizonLength, 0);
    }

    return TrajectoryTestingMultiplePolicies(
        mdp, policies, numTrajectories, horizonLength, warmupPeriods,
        trackStatistics, percentage_difference_benchmark_policy_index,
        warmup_policy);
}




void TestTrajectories_notParallel() {

    auto& dp = DynaPlexProvider::Get();

    DynaPlex::VarGroup config;
    std::string id = "asml_assembly_nonstationary";
    int64_t NumberofNodes = 10;
    int64_t p = 50;
    bool assembly = true;
    bool stationary = false;
    bool featureEng = false;
    bool zerosMaxSystemOutput = false;

    config.Add("id", id);
    config.Add("NumberofNodes", NumberofNodes);
    // Discount factor is now 1 for the entire trajectory, where the trajectory itself is length Geom(discount_factor ) distributed
    config.Add("discount_factor", 1);
    //config.Add("discount_factor_geom", 0.975);
    //config.Add("discount_factor_geom", 0.9444444);
    config.Add("p", p);
    config.Add("assembly", assembly);
    config.Add("stationary", stationary);
    config.Add("featureEngineering", featureEng);
    config.Add("zerosMaxSystemOutput", zerosMaxSystemOutput);



    DynaPlex::VarGroup test_config;
    test_config.Add("warmup_periods", 20);
    test_config.Add("number_of_trajectories", 2000);
    test_config.Add("periods_per_trajectory", 10000);

    DynaPlex::VarGroup policy_config;
    policy_config.Add("id", "balancing_base_stock");
    policy_config.Add("shortfall", true);




    DynaPlex::MDP mdp = dp.GetMDP(config);
    auto policy_BS = mdp->GetPolicy(policy_config);

    // For the neural network
    // Train network feature engineering
    DynaPlex::VarGroup nn_architecture{
    {"type","mlp"},
    {"hidden_layers",DynaPlex::VarGroup::Int64Vec{256,128,128,64}},
    };


    DynaPlex::VarGroup nn_training{
            {"mini_batch_size", 64},
            {"max_training_epochs", 1000},
            {"train_based_on_probs", false}
    };

    int64_t N = 75003;
    int64_t M = 600;
    int64_t num_gens = 1;
    DynaPlex::VarGroup dcl_config{
        //use paper hyperparameters everywhere. 
        {"N",N},                            //Number of samples
        {"num_gens",num_gens},              // Number of generations
        {"M",M},                            // Number of exogenous scenarios per state-action pair
        {"H", 100},                             // The horizon length, but will be geometrically distributed
        {"L", 20},                              // the warm-up periods per trajectory
        {"nn_architecture",nn_architecture},
        {"nn_training",nn_training},
        {"retrain_lastgen_only",false},
        {"delete_samples_after_training",true}
    };

    //only print on this node in case of multi-node program. 
    dp.System() << config.Dump() << std::endl;

    std::string NNodes_str1 = "_NN" + std::to_string(static_cast<int64_t>(NumberofNodes));
    std::string p_str1 = "_p" + std::to_string(p);
    std::string assembly_str1;
    if (assembly) {
        assembly_str1 = "_A";
    }
    else {
        assembly_str1 = "_S";
    }
    std::string stationary_str1;
    if (stationary) {
        stationary_str1 = "_S";
    }
    else {
        stationary_str1 = "_NS";
    }
    std::string N_str1 = "_N" + std::to_string(static_cast<int64_t>(N));
    std::string M_str1 = "_M" + std::to_string(M);
    std::string num_gens_str1 = "_G" + std::to_string(num_gens);
    std::string featureEng_st1r;
    std::string zerosMaxSystemOutput_str1;
    if (featureEng) {
        featureEng_st1r = "_FE";
    }
    else {
        featureEng_st1r = "_NoFE";
    }
    if (zerosMaxSystemOutput) {
        zerosMaxSystemOutput_str1 = "_DivMean";
    }
    else {
        zerosMaxSystemOutput_str1 = "_lowN";
    }
    std::string loc1 = id + NNodes_str1 + p_str1 + assembly_str1 + stationary_str1 + N_str1 + M_str1 + num_gens_str1 + featureEng_st1r + zerosMaxSystemOutput_str1;
    dp.System() << "Network id:  " << loc1 << std::endl;


    auto path1 = dp.System().filepath(loc1, "dcl_gen" + 1);
    auto policy_NN = dp.LoadPolicy(mdp, path1);

    std::vector<DynaPlex::Policy> policies;
    policies.push_back(policy_BS);
    policies.push_back(policy_NN);

    auto policy = policy_NN;

    using underlying_mdp = DynaPlex::Models::asml_assembly_nonstationary::MDP;
    auto eval_mdp = underlying_mdp(config);
    const int64_t WarmupLength = 20;                                 // Set the length of the warm-up period (10k is used for the benchmark study)
    const int64_t numTrajectories = 500;                          // Set the simulation length (100k is used for the benchmark study)
    double BS_tot_costs = 0.0;                                                                       // Initialize the costs for the benchmark
    vector<double> BS_result_vec(numTrajectories, 0.0);                                      // Initialize vector to store benchmark costs for each 1/100th of the simulation length
    const double t_val = 1.985;


    // Initialize simulation-specific variables
    RNG rng{ true, int64_t(42) };                                              // Create a Random Number Generator (RNG) and seed it with the value 1231234
    //RNG rng{ uint64_t(1231234) };                                                   
    auto init_BS_state = eval_mdp.GetInitialState();    // Initialize a state with the forced instanceID
    double BS_costs = 0.0;                                                                           // Initialize the costs for the benchmark

    int64_t rng_seed = 17;
    int64_t warmupLength = 0;

    vector<double> avg_inventory_per_node_vec(eval_mdp.NumberofNodes, 0.0);
    vector<double> avg_perc_periods_with_backlog(numTrajectories, 0.0);

    // for (auto policy : policies) {
    BS_result_vec = vector<double>(numTrajectories, 0.0);
    // Perform a simulation run
    for (int64_t i = 0; i < numTrajectories; i++) {
        // Initialize states and trajectories
        //std::vector<DynaPlex::Trajectory> BS_trajectory{};
        //BS_trajectory.reserve(1);
        //BS_trajectory.back().RNGProvider.SeedEventStreams(true, rng_seed, i);
        if (i % 20 == 0) {
            std::cout << i << std::endl;
        }


        Trajectory BS_trajectory{};
        BS_trajectory.RNGProvider.SeedEventStreams(true, rng_seed, i);
        mdp->InitiateState({ &BS_trajectory, 1 });
        auto& BS_cat = BS_trajectory.Category;
        auto& BS_state = DynaPlex::RetrieveState<underlying_mdp::State>(BS_trajectory.GetState());
        BS_state = init_BS_state;
        warmupLength = 0;

        vector<double> avg_inventory_per_node_per_traj_vec(eval_mdp.NumberofNodes, 0.0);

        // Perform warm-up of WarmupLength without vertical move or reaching final state
        mdp->AdaptWarmUp({ &BS_trajectory, 1 });
        while (warmupLength < WarmupLength) {
            // Execute action loop
            while (BS_cat.IsAwaitAction()) {
                policy->SetAction({ &BS_trajectory,1 });
                mdp->IncorporateAction({ &BS_trajectory,1 });
            }

            // Process event
            if (BS_cat.IsAwaitEvent()) {
                mdp->IncorporateEvent({ &BS_trajectory,1 });
                ++warmupLength;
            }
        }
        mdp->AdaptWarmUp({ &BS_trajectory, 1 });

        // Reset costs after the warm-up period is over
        BS_trajectory.CumulativeReturn = 0.0;
        BS_trajectory.Category = init_BS_state.cat;
        BS_state.cat = init_BS_state.cat;
        int64_t count = 0;
        int64_t periods_with_backlog = 0;
        while (!BS_cat.IsFinal()) {
            ++count;
            // Execute action loop
            while (BS_cat.IsAwaitAction()) {
                policy->SetAction({ &BS_trajectory,1 });
                mdp->IncorporateAction({ &BS_trajectory,1 });
            }

            // Process event
            if (BS_cat.IsAwaitEvent()) {

                //for (int64_t n = 0; n < eval_mdp.NumberofNodes; n++) {
                //    avg_inventory_per_node_per_traj_vec.at(n) += BS_state.StateVectors.at(eval_mdp.frontposition_vec.at(n));
                //}

                //int64_t arriving = BS_state.StateVectors.at(eval_mdp.frontposition_vec.at(eval_mdp.NumberofNodes - 1) + 1);

                mdp->IncorporateEvent({ &BS_trajectory,1 });


                //if (BS_state.StateVectors.at(eval_mdp.frontposition_vec.at(eval_mdp.NumberofNodes - 1)) < arriving) {
                //    ++periods_with_backlog;
                //}
            }
        }

        for (int64_t n = 0; n < eval_mdp.NumberofNodes; n++) {
            avg_inventory_per_node_vec.at(n) += avg_inventory_per_node_per_traj_vec.at(n) / count;
        }
        avg_perc_periods_with_backlog.at(i) = static_cast<double>(periods_with_backlog) / count;
        BS_result_vec[i] += BS_trajectory.CumulativeReturn;
    }
    double sum = std::accumulate(BS_result_vec.begin(), BS_result_vec.end(), 0.0);
    double mean = sum / BS_result_vec.size();
    std::cout << " Mean = " << mean << std::endl;
    // }
    vector<double> avg_inv_vec(eval_mdp.NumberofNodes, 0.0);
    for (int64_t n = 0; n < eval_mdp.NumberofNodes; n++) {
        avg_inv_vec.at(n) += avg_inventory_per_node_vec.at(n) / numTrajectories;
    }

    for (int64_t n = 0; n < eval_mdp.NumberofNodes; n++) {
        std::cout << " Avg inventory of node " << 10 - n << " is " << avg_inv_vec.at(n) << std::endl;
    }

    double total_perc_periods_backlog = 0;
    for (int64_t n = 0; n < numTrajectories; n++) {
        total_perc_periods_backlog += avg_perc_periods_with_backlog.at(n);
    }

    std::cout << " % Periods with backlog " << total_perc_periods_backlog / numTrajectories << std::endl;

    double avg_inv_value = 0.0;
    for (int64_t n = 0; n < eval_mdp.NumberofNodes; n++) {
        avg_inv_value += avg_inv_vec.at(n) * eval_mdp.h_vec.at(n);
    }
    std::cout << "Avg inv value " << avg_inv_value << std::endl;


    // Compute total costs
    double BS_tot_cost = 0;
    for (int64_t i = 0; i < numTrajectories; i++) {
        BS_tot_cost += BS_result_vec[i];
    }
    std::cout << "BS: Avg. exact costs: " << BS_tot_cost / numTrajectories;
    std::cout << ", after " << numTrajectories << " iterations" << std::endl;
    std::cout << std::endl;

}


static void TrainandTestPaperResults(
    const std::string folder_name,
    const DynaPlex::VarGroup& mdp_config,
    const DynaPlex::VarGroup& test_config,
    const DynaPlex::VarGroup& training_config,
    const vector<bool> featureEng_vec,
    bool trainNN,
    bool NN,
    bool exact,
    bool save,
    bool trackStatistics) {
    auto& dp = DynaPlexProvider::Get();

    // The feature-engineering flags are modified below while constructing
    // policies, so work on a local copy and leave main's configuration intact.
    DynaPlex::VarGroup config = mdp_config;
    std::string id;
    int64_t NumberofNodes;
    int64_t p;
    bool assembly;
    bool stationary;
    //bool configuredFeatureEngineering;
    bool configuredZerosMaxSystemOutput;
    config.Get("id", id);
    config.Get("NumberofNodes", NumberofNodes);
    config.Get("p", p);
    config.Get("assembly", assembly);
    config.Get("stationary", stationary);
    //config.GetOrDefault("featureEngineering", configuredFeatureEngineering, false);
    config.GetOrDefault("zerosMaxSystemOutput", configuredZerosMaxSystemOutput, false);

    int64_t N;
    int64_t M;
    int64_t num_gens;
    training_config.Get("N", N);
    training_config.Get("M", M);
    training_config.Get("num_gens", num_gens);

    const std::string kpi_output_path = dp.System().filepath(folder_name);
    std::cout << kpi_output_path << std::endl;



    DynaPlex::VarGroup policy_config_BS_TW;
    policy_config_BS_TW.Add("id", "balancing_base_stock");

    DynaPlex::VarGroup policy_config_BS_R;
    policy_config_BS_R.Add("id", "balancing_base_stock");
    policy_config_BS_R.Add("balancing_downstream", false);

    DynaPlex::VarGroup policy_config_BS_N;
    policy_config_BS_N.Add("id", "balancing_base_stock");
    policy_config_BS_N.Add("balancing_downstream", false);
    policy_config_BS_N.Add("balancing_upstream", false);

    DynaPlex::VarGroup policy_config_BS_TW_S;
    policy_config_BS_TW_S.Add("id", "balancing_base_stock");
    policy_config_BS_TW_S.Add("shortfall", true);

    DynaPlex::VarGroup policy_config_BS_R_S;
    policy_config_BS_R_S.Add("id", "balancing_base_stock");
    policy_config_BS_R_S.Add("shortfall", true);
    policy_config_BS_R_S.Add("balancing_upstream", true);
    policy_config_BS_R_S.Add("balancing_downstream", false);

    DynaPlex::VarGroup policy_config_BS_N_S;
    policy_config_BS_N_S.Add("id", "balancing_base_stock");
    policy_config_BS_N_S.Add("shortfall", true);
    policy_config_BS_N_S.Add("balancing_upstream", false);
    policy_config_BS_N_S.Add("balancing_downstream", false);


    // Train network feature engineering
    DynaPlex::VarGroup nn_architecture{
    {"type","mlp"},
    {"hidden_layers",DynaPlex::VarGroup::Int64Vec{256,128,128,64}},
    };


    DynaPlex::VarGroup nn_training{
            {"mini_batch_size", 64},
            {"max_training_epochs", 1000},
            {"train_based_on_probs", false}
    };

    DynaPlex::VarGroup dcl_config{
        //use paper hyperparameters everywhere. 
        {"N",N},                            //Number of samples
        {"num_gens",num_gens},              // Number of generations
        {"M",M},                            // Number of exogenous scenarios per state-action pair
        {"H", 100},                             // The horizon length, but will be geometrically distributed
        {"L", 20},                              // the warm-up periods per trajectory
        {"nn_architecture",nn_architecture},
        {"nn_training",nn_training},
        {"retrain_lastgen_only",false},
        {"delete_samples_after_training",true}
    };

    DynaPlex::VarGroup training_config1{
    {"N", 25000},
    {"M", 600},
    {"num_gens", 5}
    };

    DynaPlex::VarGroup dcl_config1{
        //use paper hyperparameters everywhere. 
        {"N",25000},                            //Number of samples
        {"num_gens",5},              // Number of generations
        {"M",600},                            // Number of exogenous scenarios per state-action pair
        {"H", 100},                             // The horizon length, but will be geometrically distributed
        {"L", 20},                              // the warm-up periods per trajectory
        {"nn_architecture",nn_architecture},
        {"nn_training",nn_training},
        {"retrain_lastgen_only",false},
        {"delete_samples_after_training",true}
    };

    // Training and evaluating
    for (bool featureEng : featureEng_vec) {
        config.Set("featureEngineering", featureEng);

        DynaPlex::MDP mdp = dp.GetMDP(config);
        auto policy_BS_TW = mdp->GetPolicy(policy_config_BS_TW);
        auto policy_BS_R = mdp->GetPolicy(policy_config_BS_R);
        auto policy_BS_N = mdp->GetPolicy(policy_config_BS_N);
        auto policy_BS_TW_S = mdp->GetPolicy(policy_config_BS_TW_S);
        auto policy_BS_R_S = mdp->GetPolicy(policy_config_BS_R_S);
        auto policy_BS_N_S = mdp->GetPolicy(policy_config_BS_N_S);

        const std::string loc = GetNeuralNetworkLocation(config, training_config);
        std::cout << loc << std::endl;

        DynaPlex::Policy policy_NN_best2;
        int64_t policy_lowest_costs2 = -1;


        if (trainNN) {
            //only print on this node in case of multi-node program. 
            dp.System() << config.Dump() << std::endl;

            const std::string loc1 = GetNeuralNetworkLocation(config, training_config1);

            auto dcl2 = dp.GetDCL(mdp, policy_BS_TW_S, dcl_config1, policy_BS_TW_S);

            //const std::string loc1 = GetNeuralNetworkLocation(
            //    config, auxiliary_training_config);
            //dp.System() << "Network id previous generation:  " << loc1 << std::endl;
            //auto path1 = dp.System().filepath(loc1, "dcl_gen" + 5);
            //auto nn_policy1 = dp.LoadPolicy(mdp, path1);



            // Call and train DCL with specified instance to solve
            auto dcl = dp.GetDCL(mdp, policy_BS_TW_S, dcl_config, policy_BS_TW_S);
            //auto dcl = dp.GetDCL(mdp, nn_policy1, dcl_config, nn_policy1);
            dcl.TrainPolicy();

            dp.System() << "Network id:  " << loc << std::endl;
            for (int64_t gen = 1; gen <= num_gens; gen++)
            {
                auto policy = dcl.GetPolicy(gen);
                auto path = dp.System().filepath(loc, "dcl_gen" + gen);
                dp.SavePolicy(policy, path);
            }



            //// The previous 5 generations, when that is a better policy
            //// Evaluate which generation performs best
            //std::vector<DynaPlex::Policy> policies2;
            //for (int64_t gen = 1; gen <= 5; gen++)
            //{
            //    auto path = dp.System().filepath(loc1, "dcl_gen" + gen);
            //    auto nn_policy = dp.LoadPolicy(mdp, path);
            //    policies2.push_back(nn_policy);
            //}
            //policies2.push_back(policy_BS_TW_S);

            //auto comparer_nn = dp.GetPolicyComparer(mdp, test_config);
            //auto comparison_nn = comparer_nn.Compare_withWarmup(policies2, num_gens);

            //double best_nn_cost = std::numeric_limits<double>::infinity();
            //int64_t count = 0;
            //for (auto& VarGroup : comparison_nn)
            //{
            //    double nn_cost;
            //    dp.System() << VarGroup.Dump() << std::endl;
            //    VarGroup.Get("mean", nn_cost);
            //    if (count != num_gens) {
            //        if (nn_cost < best_nn_cost) {
            //            best_nn_cost = nn_cost;
            //            policy_lowest_costs2 = count;
            //        }
            //    }
            //    count++;
            //}

            //std::cout << " Best Gen: " << policy_lowest_costs2 + 1 << std::endl;
            //policy_NN_best2 = policies2[policy_lowest_costs2];
        }


        int64_t policy_lowest_costs = -1;
        DynaPlex::Policy policy_NN_best;

        if (NN) {
            // Evaluate which generation performs best
            std::vector<DynaPlex::Policy> policies;
            for (int64_t gen = 1; gen <= num_gens; gen++)
            {
                auto path = dp.System().filepath(loc, "dcl_gen" + gen);
                auto nn_policy = dp.LoadPolicy(mdp, path);
                policies.push_back(nn_policy);
            }
            policies.push_back(policy_BS_TW_S);

            auto comparer_nn = dp.GetPolicyComparer(mdp, test_config);
            auto comparison_nn = comparer_nn.Compare_withWarmup(policies, num_gens);

            double best_nn_cost = std::numeric_limits<double>::infinity();
            int64_t count = 0;
            for (auto& VarGroup : comparison_nn)
            {
                double nn_cost;
                dp.System() << VarGroup.Dump() << std::endl;
                VarGroup.Get("mean", nn_cost);
                if (count != num_gens) {
                    if (nn_cost < best_nn_cost) {
                        best_nn_cost = nn_cost;
                        policy_lowest_costs = count;
                    }
                }
                count++;
            }

            std::cout << " Best Gen: " << policy_lowest_costs + 1 << std::endl;
            policy_NN_best = policies[policy_lowest_costs];
        }

        // Evaluation
        std::vector<DynaPlex::Policy> policies_evaluation;
        int64_t warm_up_policy = 0;
        auto comparer_eval = dp.GetPolicyComparer(mdp, test_config);

        if (exact) {
            DynaPlex::VarGroup exact_config = DynaPlex::VarGroup{ {"max_states",12000000}, {"silent", false } };
            // Running the evaluation after warm-up
            const auto optimal_policy_time_start = std::chrono::steady_clock::now();

            auto exactsolver = dp.GetExactSolver(mdp, exact_config);
            auto optimal_policy = exactsolver.GetOptimalPolicy(policy_BS_TW_S);

            const auto optimal_policy_time = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - optimal_policy_time_start).count();
            std::cout << "Optimal policy time is : " << optimal_policy_time << std::endl;

            policies_evaluation.push_back(optimal_policy);
            policies_evaluation.push_back(policy_BS_N);
            policies_evaluation.push_back(policy_BS_N_S);
            policies_evaluation.push_back(policy_BS_R);
            policies_evaluation.push_back(policy_BS_R_S);
            policies_evaluation.push_back(policy_BS_TW);
            policies_evaluation.push_back(policy_BS_TW_S);
            if (NN) {
                policies_evaluation.push_back(policy_NN_best);
                if (policy_lowest_costs2 != -1) {
                    policies_evaluation.push_back(policy_NN_best2);
                }
            }


            warm_up_policy = 0;

            auto comparison_opt = comparer_eval.Compare_withWarmup(policies_evaluation, warm_up_policy);
            for (auto& VarGroup : comparison_opt) {
                std::cout << VarGroup.Dump() << std::endl;
            }
        }
        else {
            std::cout << " ----- Compare all policies - starting from BS policy ------" << std::endl;
            policies_evaluation.push_back(policy_BS_N);
            policies_evaluation.push_back(policy_BS_N_S);
            policies_evaluation.push_back(policy_BS_R);
            policies_evaluation.push_back(policy_BS_R_S);
            policies_evaluation.push_back(policy_BS_TW);
            policies_evaluation.push_back(policy_BS_TW_S);
            if (NN) {
                policies_evaluation.push_back(policy_NN_best);
                }
            warm_up_policy = 5;

            auto comparison_end_BS = comparer_eval.Compare_withWarmup(policies_evaluation, warm_up_policy);
            for (auto& VarGroup : comparison_end_BS) {
                std::cout << VarGroup.Dump() << std::endl;
            }
        }

        // Run the custom paired evaluation once after policies_evaluation is
        // complete. Use the same warm-up policy as Compare_withWarmup.
        // Add {"track_statistics", true} to test_config when the per-node
        // trajectory KPIs should also be recorded and saved. Keep it false
        // for a timing-focused run, so milliseconds exclude bookkeeping.
        if (trackStatistics) {
            bool track_statistics = true;
            const auto trajectory_results = TrajectoryTestingMultiplePolicies(
                mdp,
                policies_evaluation,
                test_config,
                track_statistics,
                warm_up_policy,
                policies_evaluation.at(warm_up_policy));

            if (save) {
                SaveTrajectoryTestingMultiplePoliciesResults(
                    trajectory_results,
                    track_statistics,
                    loc,
                    kpi_output_path, featureEng);
            }

            std::cout << " ----- Post-warm-up trajectory evaluation -----" << std::endl;
            for (size_t policy_index = 0;
                policy_index < policies_evaluation.size(); ++policy_index) {
                const auto& evaluation_times =
                    trajectory_results.trajectory_evaluation_milliseconds.at(policy_index);
                const double mean_evaluation_time = std::accumulate(
                    evaluation_times.begin(), evaluation_times.end(), 0.0) /
                    static_cast<double>(evaluation_times.size());
                std::cout << "Policy " << policy_index
                    << ": mean evaluation time = "
                    << mean_evaluation_time << " ms per trajectory" << std::endl;

                const double mean = std::accumulate(
                    trajectory_results.policy_costs.at(policy_index).begin(), trajectory_results.policy_costs.at(policy_index).end(), 0.0
                ) / static_cast<double>(trajectory_results.policy_costs.at(policy_index).size());
                std::cout << "Policy " << policy_index
                    << ": mean cost = "
                    << mean << " per trajectory" << std::endl;
            }

            std::cout << " ----- Paired percentage cost changes relative to policy "
                << warm_up_policy << " -----" << std::endl;


            for (size_t policy_index = 0;
                policy_index < policies_evaluation.size(); ++policy_index) {
                if (policy_index == static_cast<size_t>(warm_up_policy)) {
                    continue;
                }

                const auto paired_cost_result = PairedPercentDifferenceTest(
                    std::span<const double>{
                    trajectory_results.percentage_differences.at(policy_index)});
                std::cout << "Policy " << policy_index
                    << ": mean cost increase = "
                    << paired_cost_result.mean_percent_cost_increase
                    << "% (95% CI: [" << paired_cost_result.ci_lower
                    << "%, " << paired_cost_result.ci_upper
                    << "%], p=" << paired_cost_result.two_sided_p_value
                    << ")" << std::endl;
            }
        }
    }

    
}


void ComparePolicies(bool exact, bool nn) {
    auto& dp = DynaPlexProvider::Get();


    DynaPlex::VarGroup config;
    std::string id = "asml_assembly_nonstationary";
    int64_t NumberofNodes = 3;
    int64_t p = 50;
    bool assembly = true;
    bool stationary = true;
    bool featureEng = true;
    bool zerosMaxSystemOutput = false;


    config.Add("id", id);
    config.Add("NumberofNodes", NumberofNodes);
    // Discount factor is now 1 for the entire trajectory, where the trajectory itself is length Geom(discount_factor ) distributed
    config.Add("discount_factor", 1);
    //config.Add("discount_factor_geom", 0.975);
    //config.Add("discount_factor_geom", 0.9444444);
    config.Add("p", p);
    config.Add("assembly", assembly);
    config.Add("stationary", stationary);
    config.Add("featureEngineering", featureEng);
    config.Add("zerosMaxSystemOutput", zerosMaxSystemOutput);

    DynaPlex::VarGroup test_config;
    test_config.Add("warmup_periods", 30);
    test_config.Add("number_of_trajectories", 1000);
    test_config.Add("periods_per_trajectory", 10000);

    DynaPlex::VarGroup policy_config;
    policy_config.Add("id", "balancing_base_stock");

    DynaPlex::VarGroup policy_config2;
    policy_config2.Add("id", "balancing_base_stock");
    policy_config2.Add("balancing_downstream", false);

    DynaPlex::VarGroup policy_config3;
    policy_config3.Add("id", "balancing_base_stock");
    policy_config3.Add("balancing_downstream", false);
    policy_config3.Add("balancing_upstream", false);

    DynaPlex::VarGroup policy_config4;
    policy_config4.Add("id", "balancing_base_stock");
    policy_config4.Add("shortfall", true);


    DynaPlex::VarGroup policy_config5;
    policy_config5.Add("id", "balancing_base_stock");
    policy_config5.Add("shortfall", true);
    policy_config5.Add("balancing_upstream", true);
    policy_config5.Add("balancing_downstream", false);

    DynaPlex::VarGroup policy_config6;
    policy_config6.Add("id", "balancing_base_stock");
    policy_config6.Add("shortfall", true);
    policy_config6.Add("balancing_upstream", false);
    policy_config6.Add("balancing_downstream", false);


    // Train network feature engineering
    DynaPlex::VarGroup nn_architecture{
    {"type","mlp"},
    {"hidden_layers",DynaPlex::VarGroup::Int64Vec{256,128,128,64}},
    };


    DynaPlex::VarGroup nn_training{
            {"mini_batch_size", 64},
            {"max_training_epochs", 1000},
            {"train_based_on_probs", false}
    };

    int64_t N = 75000;
    int64_t M = 600;
    int64_t num_gens = 2;
    DynaPlex::VarGroup dcl_config{
        //use paper hyperparameters everywhere. 
        {"N",N},                            //Number of samples
        {"num_gens",num_gens},              // Number of generations
        {"M",M},                            // Number of exogenous scenarios per state-action pair
        {"H", 100},                             // The horizon length, but will be geometrically distributed
        {"L", 20},                              // the warm-up periods per trajectory
        {"nn_architecture",nn_architecture},
        {"nn_training",nn_training},
        {"retrain_lastgen_only",false},
        {"delete_samples_after_training",true}
    };


    vector<int64_t> instance_vec = { 0 };
    vector<int64_t> p_vec = { 100 };
    vector<bool> stationary_vec = { false };
    for (int64_t instance : instance_vec) {
        config.Set("instance", instance);

        DynaPlex::MDP mdp = dp.GetMDP(config);
        auto policy_dt = mdp->GetPolicy(policy_config);
        auto policy_dt_R = mdp->GetPolicy(policy_config2);
        auto policy_no_bal = mdp->GetPolicy(policy_config3);
        auto policy_dt_S_CB = mdp->GetPolicy(policy_config4);
        auto policy_dt_S_R = mdp->GetPolicy(policy_config5);
        auto policy_dt_S_no_bal = mdp->GetPolicy(policy_config6);
        //auto policy_dt_no_bal_bottleneck = mdp->GetPolicy(policy_config7);
        //auto policy_dt_R_no_bal_bottleneck = mdp->GetPolicy(policy_config8);
        //auto policy_no_bal_no_bal_bottleneck = mdp->GetPolicy(policy_config9);
        //auto policy_dt_S_CB_no_bal_bottleneck = mdp->GetPolicy(policy_config10);
        //auto policy_dt_S_R_no_bal_bottleneck = mdp->GetPolicy(policy_config11);
        //auto policy_dt_S_no_bal_no_bal_bottleneck = mdp->GetPolicy(policy_config12);



        //only print on this node in case of multi-node program. 
        dp.System() << config.Dump() << std::endl;

        std::vector<DynaPlex::Policy> policies;
        policies.push_back(policy_dt);
        policies.push_back(policy_dt_R);
        policies.push_back(policy_no_bal);
        policies.push_back(policy_dt_S_CB);
        policies.push_back(policy_dt_S_R);
        policies.push_back(policy_dt_S_no_bal);
        //policies.push_back(policy_dt_no_bal_bottleneck);
        //policies.push_back(policy_dt_R_no_bal_bottleneck);
        //policies.push_back(policy_no_bal_no_bal_bottleneck);
        //policies.push_back(policy_dt_S_CB_no_bal_bottleneck);
        //policies.push_back(policy_dt_S_R_no_bal_bottleneck);
        //policies.push_back(policy_dt_S_no_bal_no_bal_bottleneck);



        auto comparer = dp.GetPolicyComparer(mdp, test_config);

        //auto comparison = comparer.Compare_withWarmup(policies, 3);
        //for (auto& VarGroup : comparison) {
        //    std::cout << VarGroup.Dump() << std::endl;
        if ((!exact) && (!nn)) {
            auto comparison = comparer.Compare_withWarmup(policies, 3, 3);
            for (auto& VarGroup : comparison) {
                std::cout << VarGroup.Dump() << std::endl;
            }
        }
        if ((exact) && (!nn)) {
            DynaPlex::VarGroup exact_config = DynaPlex::VarGroup{ {"max_states",12000000}, {"silent", false } };

            auto exactsolver = dp.GetExactSolver(mdp, exact_config);
            auto optimal_policy = exactsolver.GetOptimalPolicy(policy_dt);
            //double bs_pol_cost = exactsolver.ComputeCosts(false, policy_dt);
            // std::cout << "BS optimal costs:  " << bs_pol_cost << std::endl;

            std::vector<DynaPlex::Policy> policies_opt;
            policies_opt.push_back(optimal_policy);
            policies_opt.push_back(policy_dt_R);
            policies_opt.push_back(policy_dt);
            policies_opt.push_back(policy_dt_R);
            policies_opt.push_back(policy_no_bal);
            policies_opt.push_back(policy_dt_S_CB);
            policies_opt.push_back(policy_dt_S_R);
            policies_opt.push_back(policy_dt_S_no_bal);



            auto comparison_opt = comparer.Compare_withWarmup(policies_opt, 0);
            for (auto& VarGroup : comparison_opt) {
                std::cout << VarGroup.Dump() << std::endl;
            }
        }
        if (nn) {
            std::string NNodes_str1 = "_NN" + std::to_string(static_cast<int64_t>(NumberofNodes));
            std::string p_str1 = "_p" + std::to_string(p);
            std::string assembly_str1;
            if (assembly) {
                assembly_str1 = "_A";
            }
            else {
                assembly_str1 = "_S";
            }
            std::string stationary_str1;
            if (stationary) {
                stationary_str1 = "_S";
            }
            else {
                stationary_str1 = "_NS";
            }
            std::string N_str1 = "_N" + std::to_string(static_cast<int64_t>(N));
            std::string M_str1 = "_M" + std::to_string(M);
            std::string num_gens_str1 = "_G" + std::to_string(num_gens);
            std::string featureEng_st1r;
            std::string zerosMaxSystemOutput_str1;
            if (featureEng) {
                featureEng_st1r = "_FE";
            }
            else {
                featureEng_st1r = "_NoFE";
            }
            if (zerosMaxSystemOutput) {
                zerosMaxSystemOutput_str1 = "_DivMean";
            }
            else {
                zerosMaxSystemOutput_str1 = "_lowN";
            }
            std::string loc1 = id + NNodes_str1 + p_str1 + assembly_str1 + stationary_str1 + N_str1 + M_str1 + num_gens_str1 + featureEng_st1r + zerosMaxSystemOutput_str1;
            dp.System() << "Network id:  " << loc1 << std::endl;


            //auto path1 = dp.System().filepath(loc1, "dcl_gen" + 1);
            //auto policy_NN_best = dp.LoadPolicy(mdp, path1);

            auto path1 = dp.System().filepath(loc1, "dcl_gen" + 1);
            std::vector<DynaPlex::Policy> policiesNN;
            for (auto gen = 1; gen <= num_gens; gen++) {
                path1 = dp.System().filepath(loc1, "dcl_gen" + gen);
                policiesNN.push_back(dp.LoadPolicy(mdp, path1));
            }

            ////// Call and train DCL with specified instance to solve
            //auto dcl = dp.GetDCL(mdp, policy_dt_S_CB, dcl_config, policy_dt_S_CB);
            //////dcl.TrainPolicy();



            //auto policiesNN = dcl.GetPolicies();
            auto comparisonNN = comparer.Compare_withWarmup(policiesNN);

            double lowest_cost = 100000000;                 // Initialize lowest costs with very high costs
            int64_t policy_lowest_costs = -1;
            int64_t count = 0;
            double mean;
            for (auto& VarGroup : comparisonNN)
            {
                dp.System() << VarGroup.Dump() << std::endl;
                comparisonNN[count].Get("mean", mean);
                // Find the neural network with lowest costs, note the first policy is the benchmark policy
                if ((mean < lowest_cost) && (count != 0) && (count != 6)) {
                    policy_lowest_costs = count;
                }
                count++;
            }
            auto policy_NN_best = policiesNN[policy_lowest_costs];





            if (!exact) {
                std::cout << " ----- Compare with best NN generation policy ------" << std::endl;
                std::vector<DynaPlex::Policy> policies_withNN;
                policies_withNN.push_back(policy_dt);
                policies_withNN.push_back(policy_dt_R);
                policies_withNN.push_back(policy_no_bal);
                policies_withNN.push_back(policy_dt_S_CB);
                policies_withNN.push_back(policy_NN_best);
                policies_withNN.push_back(policy_dt_S_R);
                policies_withNN.push_back(policy_dt_S_no_bal);
                //auto comparison_end = comparer.Compare(policies_withNN, 3);
                auto comparison_end = comparer.Compare_withWarmup(policies_withNN, 3);
                for (auto& VarGroup : comparison_end) {
                    std::cout << VarGroup.Dump() << std::endl;
                }



                //std::cout << " ----- Compare with best NN generation policy with best benchmark ------" << std::endl;
                //std::vector<DynaPlex::Policy> policies_withNN_test;
                //policies_withNN_test.push_back(policy_dt_S_CB);
                //policies_withNN_test.push_back(policy_NN_best);
                //auto comparison_end_test = comparer.Compare(policies_withNN_test, 0);
                //auto comparison_end_test = comparer.Compare_withWarmup(policies_withNN_test, 1);
                //for (auto& VarGroup : comparison_end_test) {
                //    std::cout << VarGroup.Dump() << std::endl;
                //}

            }

            if (exact) {
                std::cout << " ----- Compute optimal policy ------" << std::endl;

                DynaPlex::VarGroup exact_config = DynaPlex::VarGroup{ {"max_states",12000000}, {"silent", false } };

                auto exactsolver = dp.GetExactSolver(mdp, exact_config);
                auto optimal_policy = exactsolver.GetOptimalPolicy(policy_dt);

                std::vector<DynaPlex::Policy> policies_all;
                policies_all.push_back(optimal_policy);
                policies_all.push_back(policy_NN_best);
                policies_all.push_back(policy_dt);
                policies_all.push_back(policy_dt_R);
                policies_all.push_back(policy_no_bal);
                policies_all.push_back(policy_dt_S_CB);
                policies_all.push_back(policy_dt_S_R);
                policies_all.push_back(policy_dt_S_no_bal);
                auto comparison_all = comparer.Compare_withWarmup(policies_all, 0);
                for (auto& VarGroup : comparison_all) {
                    std::cout << VarGroup.Dump() << std::endl;
                }
            }
        }
    }
}

/** Function to simulate our proposed shortfall distribution */
void ComputeShortfall() {
    auto& dp = DynaPlexProvider::Get();

    DynaPlex::VarGroup config;
    std::string id = "asml_assembly_nonstationary";
    int64_t NumberofNodes = 10;
    int64_t p = 50;
    bool assembly = true;
    bool stationary = false;

    config.Add("id", id);
    config.Add("NumberofNodes", NumberofNodes);
    // Discount factor is now 1 for the entire trajectory, where the trajectory itself is length Geom(discount_factor ) distributed
    config.Add("discount_factor", 1);
    //config.Add("discount_factor_geom", 0.975);
    //config.Add("discount_factor_geom", 0.9444444);
    config.Add("p", p);
    config.Add("assembly", assembly);
    config.Add("stationary", stationary);
    config.Add("featureEngineering", true);


    DynaPlex::VarGroup test_config;
    test_config.Add("warmup_periods", 30);
    test_config.Add("number_of_trajectories", 2000);
    test_config.Add("periods_per_trajectory", 10000);


    DynaPlex::VarGroup policy_config;
    policy_config.Add("id", "balancing_base_stock");
    policy_config.Add("shortfall", false);

    DynaPlex::MDP mdp = dp.GetMDP(config);
    auto policy = mdp->GetPolicy(policy_config);

    const int64_t WarmupLength = 20;                                 // Set the length of the warm-up period (10k is used for the benchmark study)
    const int64_t numTrajectories = 5000;                          // Set the simulation length (100k is used for the benchmark study)



    using underlying_mdp = DynaPlex::Models::asml_assembly_nonstationary::MDP;
    auto eval_mdp = underlying_mdp(config);

    auto init_BS_state = eval_mdp.GetInitialState();    // Initialize a state with the forced instanceID

    vector<vector<vector<int64_t>>> shortfall_results_3vec;                               // Initialize vector to store shortfall vector
    for (size_t p = 0; p <= eval_mdp.maxTree - eval_mdp.startMeanDem; p++) {
        shortfall_results_3vec.push_back({});
        for (int64_t n = 0; n < NumberofNodes; n++) {
            shortfall_results_3vec.at(p).push_back({});
            for (int64_t i = 0; i < 100; i++) {
                shortfall_results_3vec.at(p).at(n).push_back(0);
            }
        }
    }
    int64_t rng_seed = 17;
    int64_t warmupLength = 0;
    // Perform a simulation run
    for (int64_t i = 0; i < numTrajectories; i++) {
        // Initialize states and trajectories
        //std::vector<DynaPlex::Trajectory> BS_trajectory{};
        //BS_trajectory.reserve(1);
        //BS_trajectory.back().RNGProvider.SeedEventStreams(true, rng_seed, i);


        Trajectory BS_trajectory{};
        BS_trajectory.RNGProvider.SeedEventStreams(true, rng_seed, i);
        mdp->InitiateState({ &BS_trajectory, 1 });
        auto& BS_cat = BS_trajectory.Category;
        auto& BS_state = DynaPlex::RetrieveState<underlying_mdp::State>(BS_trajectory.GetState());
        BS_state = init_BS_state;
        warmupLength = 0;

        // Perform warm-up of WarmupLength without vertical move or reaching final state
        mdp->AdaptWarmUp({ &BS_trajectory, 1 });
        while (warmupLength < WarmupLength) {
            // Execute action loop
            while (BS_cat.IsAwaitAction()) {
                policy->SetAction({ &BS_trajectory,1 });
                mdp->IncorporateAction({ &BS_trajectory,1 });
            }

            // Process event
            if (BS_cat.IsAwaitEvent()) {
                mdp->IncorporateEvent({ &BS_trajectory,1 });
                ++warmupLength;
            }
        }
        mdp->AdaptWarmUp({ &BS_trajectory, 1 });


        // Reset costs after the warm-up period is over
        BS_trajectory.CumulativeReturn = 0.0;
        BS_trajectory.Category = init_BS_state.cat;
        BS_state.cat = init_BS_state.cat;
        while (!BS_cat.IsFinal()) {
            // Execute action loop
            while (BS_cat.IsAwaitAction()) {
                policy->SetAction({ &BS_trajectory,1 });
                mdp->IncorporateAction({ &BS_trajectory,1 });


                if (BS_state.currentNode != NumberofNodes - 1) {
                    ++shortfall_results_3vec.at(BS_state.verticalpos).at(BS_state.currentNode + 1).at(int64_t(eval_mdp.base_stock_levels[BS_state.verticalpos][BS_state.currentNode + 1]) - BS_state.echelonInv_vec.at(BS_state.currentNode + 1));
                }
                else {
                    ++shortfall_results_3vec.at(BS_state.verticalpos).at(0).at(int64_t(eval_mdp.base_stock_levels[BS_state.verticalpos][0]) - BS_state.echelonInv_vec.at(0));
                }


            }

            // Process event
            if (BS_cat.IsAwaitEvent()) {
                mdp->IncorporateEvent({ &BS_trajectory,1 });
            }
        }
    }
    std::string filename;

    std::string NNodes_str = "_NN" + std::to_string(static_cast<int64_t>(NumberofNodes));
    std::string p_str = "_p" + std::to_string(p);
    std::string assembly_str;
    if (assembly) {
        assembly_str = "_A";
    }
    else {
        assembly_str = "_S";
    }
    std::string stationary_str;
    if (stationary) {
        stationary_str = "_S";
    }
    else {
        stationary_str = "_NS2";
    }
    filename = id + NNodes_str + p_str + assembly_str + stationary_str + ".csv";
    std::cout << filename << std::endl;
    std::ofstream myFile;
    myFile.open(filename);
    for (auto i : shortfall_results_3vec) {
        for (auto j : i) {
            for (auto inv : j) {
                myFile << inv << ",";
            }
            myFile << std::endl;
        }
    }

    myFile.close();

}

void TestInventoryperNode(bool nn) {

    auto& dp = DynaPlexProvider::Get();

    DynaPlex::VarGroup config;
    std::string id = "asml_assembly_nonstationary";
    int64_t NumberofNodes = 10;
    int64_t p = 100;
    bool assembly = true;
    bool stationary = false;
    bool featureEng = true;
    bool zerosMaxSystemOutput = false;

    config.Add("id", id);
    config.Add("NumberofNodes", NumberofNodes);
    // Discount factor is now 1 for the entire trajectory, where the trajectory itself is length Geom(discount_factor ) distributed
    config.Add("discount_factor", 1);
    config.Add("p", p);
    config.Add("assembly", assembly);
    config.Add("stationary", stationary);
    config.Add("featureEngineering", featureEng);
    config.Add("zerosMaxSystemOutput", zerosMaxSystemOutput);

    DynaPlex::VarGroup test_config;
    test_config.Add("warmup_periods", 30);
    test_config.Add("number_of_trajectories", 5000);
    test_config.Add("periods_per_trajectory", 10000);

    DynaPlex::VarGroup policy_config;
    policy_config.Add("id", "balancing_base_stock");

    DynaPlex::VarGroup policy_config2;
    policy_config2.Add("id", "balancing_base_stock");
    policy_config2.Add("balancing_downstream", false);

    DynaPlex::VarGroup policy_config3;
    policy_config3.Add("id", "balancing_base_stock");
    policy_config3.Add("balancing_downstream", false);
    policy_config3.Add("balancing_upstream", false);

    DynaPlex::VarGroup policy_config4;
    policy_config4.Add("id", "balancing_base_stock");
    policy_config4.Add("shortfall", true);


    DynaPlex::VarGroup policy_config5;
    policy_config5.Add("id", "balancing_base_stock");
    policy_config5.Add("shortfall", true);
    policy_config5.Add("balancing_upstream", true);
    policy_config5.Add("balancing_downstream", false);

    DynaPlex::VarGroup policy_config6;
    policy_config6.Add("id", "balancing_base_stock");
    policy_config6.Add("shortfall", true);
    policy_config6.Add("balancing_upstream", false);
    policy_config6.Add("balancing_downstream", false);

    // Train network feature engineering
    DynaPlex::VarGroup nn_architecture{
        {"type","mlp"},
        {"hidden_layers",DynaPlex::VarGroup::Int64Vec{256,128,128,64}},
    };

    DynaPlex::VarGroup nn_training{
            {"mini_batch_size", 64},
            {"max_training_epochs", 1000},
            {"train_based_on_probs", false}
    };

    DynaPlex::MDP mdp = dp.GetMDP(config);

    auto policy_dt = mdp->GetPolicy(policy_config);
    auto policy_dt_R = mdp->GetPolicy(policy_config2);
    auto policy_no_bal = mdp->GetPolicy(policy_config3);
    auto policy_dt_S_CB = mdp->GetPolicy(policy_config4);
    auto policy_dt_S_R = mdp->GetPolicy(policy_config5);
    auto policy_dt_S_no_bal = mdp->GetPolicy(policy_config6);

    using underlying_mdp = DynaPlex::Models::asml_assembly_nonstationary::MDP;
    auto eval_mdp = underlying_mdp(config);
    const int64_t WarmupLength = 20;                                 // Set the length of the warm-up period (10k is used for the benchmark study)
    const int64_t numTrajectories = 1000;                          // Set the simulation length (100k is used for the benchmark study)
    double tot_costs = 0.0;                                                                       // Initialize the costs for the benchmark
    const double t_val = 1.985;


    // Initialize simulation-specific variables
    RNG rng{ true, int64_t(42) };                                              // Create a Random Number Generator (RNG) and seed it with the value 1231234
    //RNG rng{ uint64_t(1231234) };                                                   
    auto init_BS_state = eval_mdp.GetInitialState();    // Initialize a state with the forced instanceID
    double BS_costs = 0.0;                                                                           // Initialize the costs for the benchmark

    int64_t rng_seed = 17;
    int64_t warmupLength = 0;

    std::vector<DynaPlex::Policy> policies;
    policies.push_back(policy_dt);
    policies.push_back(policy_dt_R);
    policies.push_back(policy_no_bal);
    policies.push_back(policy_dt_S_CB);
    policies.push_back(policy_dt_S_R);
    policies.push_back(policy_dt_S_no_bal);


    int64_t N = 75003;
    int64_t M = 600;
    int64_t num_gens = 1;
    DynaPlex::VarGroup dcl_config{
        //use paper hyperparameters everywhere. 
        {"N",N},                            //Number of samples
        {"num_gens",num_gens},              // Number of generations
        {"M",M},                            // Number of exogenous scenarios per state-action pair
        {"H", 100},                             // The horizon length, but will be geometrically distributed
        {"L", 20},                              // the warm-up periods per trajectory
        {"nn_architecture",nn_architecture},
        {"nn_training",nn_training},
        {"retrain_lastgen_only",false},
        {"delete_samples_after_training",true}
    };

    if (nn) {
        std::string NNodes_str1 = "_NN" + std::to_string(static_cast<int64_t>(NumberofNodes));
        std::string p_str1 = "_p" + std::to_string(p);
        std::string assembly_str1;
        if (assembly) {
            assembly_str1 = "_A";
        }
        else {
            assembly_str1 = "_S";
        }
        std::string stationary_str1;
        if (stationary) {
            stationary_str1 = "_S";
        }
        else {
            stationary_str1 = "_NS";
        }
        std::string N_str1 = "_N" + std::to_string(static_cast<int64_t>(N));
        std::string M_str1 = "_M" + std::to_string(M);
        std::string num_gens_str1 = "_G" + std::to_string(num_gens);
        std::string featureEng_st1r;
        std::string zerosMaxSystemOutput_str1;
        if (featureEng) {
            featureEng_st1r = "_FE";
        }
        else {
            featureEng_st1r = "_NoFE";
        }
        if (zerosMaxSystemOutput) {
            zerosMaxSystemOutput_str1 = "_DivMean";
        }
        else {
            zerosMaxSystemOutput_str1 = "_lowN";
        }
        std::string loc1 = id + NNodes_str1 + p_str1 + assembly_str1 + stationary_str1 + N_str1 + M_str1 + num_gens_str1 + featureEng_st1r + zerosMaxSystemOutput_str1;
        dp.System() << "Network id:  " << loc1 << std::endl;


        auto path1 = dp.System().filepath(loc1, "dcl_gen" + 1);
        auto policy_NN_best = dp.LoadPolicy(mdp, path1);
        policies.push_back(policy_NN_best);
    }

    vector<vector<vector<vector<double>>>> avg_inv_per_node_per_vertpos_per_trajectory_per_policy_vec(policies.size(), vector<vector<vector<double>>>(numTrajectories, vector<vector<double>>(eval_mdp.maxTree - eval_mdp.startMeanDem + 1, vector<double>(eval_mdp.NumberofNodes, 0.0))));
    vector<vector<vector<double>>> avg_inv_per_node_per_trajectory_per_policy_vec(policies.size(), vector<vector<double>>(numTrajectories, vector<double>(eval_mdp.NumberofNodes, 0.0)));
    vector<vector<double>> avg_perc_periods_with_backlog_per_policy_per_traj(policies.size(), vector<double>(numTrajectories, 0.0));
    vector<vector<double>> avg_backlog_costs_per_policy_per_traj(policies.size(), vector<double>(numTrajectories, 0.0));
    vector<vector<double>> avg_inv_costs_per_policy_per_traj(policies.size(), vector<double>(numTrajectories, 0.0));
    vector<vector<double>> avg_tot_costs_per_policy_per_traj(policies.size(), vector<double>(numTrajectories, 0.0));
    vector<vector<double>> avg_tot_costs_per_policy_per_traj2(policies.size(), vector<double>(numTrajectories, 0.0));

    int64_t policy_count = 0;
    // Perform a simulation run
    for (auto& policy : policies) {
        std::cout << " Policy " << std::endl;
        for (int64_t i = 0; i < numTrajectories; i++) {
            // Initialize states and trajectories
            Trajectory BS_trajectory{};
            // To maintain CRN between the policies, per trajactory
            BS_trajectory.RNGProvider.SeedEventStreams(true, rng_seed, i);
            mdp->InitiateState({ &BS_trajectory, 1 });

            auto& BS_cat = BS_trajectory.Category;
            auto& BS_state = DynaPlex::RetrieveState<underlying_mdp::State>(BS_trajectory.GetState());
            BS_state = init_BS_state;
            warmupLength = 0;

            vector<double> avg_inventory_per_node_per_traj_vec(eval_mdp.NumberofNodes, 0.0);
            vector<vector<double>> avg_inventory_per_node_per_vertpos_per_traj_vec;
            for (auto i = 0; i < eval_mdp.maxTree - eval_mdp.startMeanDem + 1; i++) {
                avg_inventory_per_node_per_vertpos_per_traj_vec.push_back({});
                for (auto n = 0; n < eval_mdp.NumberofNodes; n++) {
                    avg_inventory_per_node_per_vertpos_per_traj_vec.at(i).push_back(0.0);
                }
            }
            double tot_inv_costs_per_traj = 0.0;
            double tot_backlog_costs_per_traj = 0.0;

            // Perform warm-up of WarmupLength without vertical move or reaching final state
            mdp->AdaptWarmUp({ &BS_trajectory, 1 });
            while (warmupLength < WarmupLength) {
                // Execute action loop
                while (BS_cat.IsAwaitAction()) {
                    policy_dt_S_CB->SetAction({ &BS_trajectory,1 });
                    mdp->IncorporateAction({ &BS_trajectory,1 });
                }
                // Process event
                if (BS_cat.IsAwaitEvent()) {
                    mdp->IncorporateEvent({ &BS_trajectory,1 });
                    ++warmupLength;
                }
            }
            mdp->AdaptWarmUp({ &BS_trajectory, 1 });

            // Reset costs after the warm-up period is over
            BS_trajectory.CumulativeReturn = 0.0;
            BS_trajectory.Category = init_BS_state.cat;
            BS_state.cat = init_BS_state.cat;
            int64_t count = 0;
            vector<int64_t> count_vertpos(eval_mdp.maxTree - eval_mdp.startMeanDem + 1, 0);
            int64_t periods_with_backlog = 0;
            while (!BS_cat.IsFinal()) {
                ++count;
                ++count_vertpos.at(BS_state.verticalpos);
                // Execute action loop
                while (BS_cat.IsAwaitAction()) {
                    policy->SetAction({ &BS_trajectory,1 });
                    mdp->IncorporateAction({ &BS_trajectory,1 });
                }

                // Process event
                if (BS_cat.IsAwaitEvent()) {

                    // For all nodes except the most downstream node
                    for (int64_t n = 0; n < eval_mdp.NumberofNodes - 1; n++) {
                        avg_inventory_per_node_per_traj_vec.at(n) += BS_state.StateVectors.at(eval_mdp.frontposition_vec.at(n));
                        avg_inventory_per_node_per_vertpos_per_traj_vec.at(BS_state.verticalpos).at(n) += BS_state.StateVectors.at(eval_mdp.backposition_vec.at(n));
                        tot_inv_costs_per_traj += BS_state.StateVectors.at(eval_mdp.frontposition_vec.at(n)) * eval_mdp.h_vec.at(n);
                    }

                    int64_t arriving = BS_state.StateVectors.at(eval_mdp.frontposition_vec.at(eval_mdp.NumberofNodes - 1) + 1);

                    mdp->IncorporateEvent({ &BS_trajectory,1 });

                    // For the most downstream node
                    // If backlog
                    if (BS_state.StateVectors.at(eval_mdp.frontposition_vec.at(eval_mdp.NumberofNodes - 1)) < arriving) {
                        ++periods_with_backlog;
                        avg_inventory_per_node_per_traj_vec.at(eval_mdp.NumberofNodes - 1) += 0;
                        avg_inventory_per_node_per_vertpos_per_traj_vec.at(BS_state.verticalpos).at(eval_mdp.NumberofNodes - 1) += BS_state.StateVectors.at(eval_mdp.backposition_vec.at(eval_mdp.NumberofNodes - 1));
                        tot_backlog_costs_per_traj += eval_mdp.p * (arriving - BS_state.StateVectors.at(eval_mdp.backposition_vec.at(eval_mdp.NumberofNodes - 1)));
                    }
                    else {
                        avg_inventory_per_node_per_traj_vec.at(eval_mdp.NumberofNodes - 1) += std::max(int(BS_state.StateVectors.at(eval_mdp.frontposition_vec.at(eval_mdp.NumberofNodes - 1)) - arriving), 0);
                        avg_inventory_per_node_per_vertpos_per_traj_vec.at(BS_state.verticalpos).at(eval_mdp.NumberofNodes - 1) += BS_state.StateVectors.at(eval_mdp.backposition_vec.at(eval_mdp.NumberofNodes - 1));
                        tot_inv_costs_per_traj += (BS_state.StateVectors.at(eval_mdp.frontposition_vec.at(eval_mdp.NumberofNodes - 1)) - arriving) * eval_mdp.h_vec.at(eval_mdp.NumberofNodes - 1);
                    }
                }
            }

            for (int64_t v = 0; v < eval_mdp.maxTree - eval_mdp.startMeanDem + 1; v++) {
                if (count_vertpos.at(v) > 0) {
                    for (int64_t n = 0; n < eval_mdp.NumberofNodes; n++) {
                        avg_inv_per_node_per_vertpos_per_trajectory_per_policy_vec.at(policy_count).at(i).at(v).at(n) += avg_inventory_per_node_per_vertpos_per_traj_vec.at(v).at(n) / count_vertpos.at(v);
                    }
                }
                else {
                    for (int64_t n = 0; n < eval_mdp.NumberofNodes; n++) {
                        avg_inv_per_node_per_vertpos_per_trajectory_per_policy_vec.at(policy_count).at(i).at(v).at(n) += 0;
                    }
                }
            }

            for (int64_t n = 0; n < eval_mdp.NumberofNodes; n++) {
                avg_inv_per_node_per_trajectory_per_policy_vec.at(policy_count).at(i).at(n) += avg_inventory_per_node_per_traj_vec.at(n) / count;
            }
            avg_perc_periods_with_backlog_per_policy_per_traj.at(policy_count).at(i) = static_cast<double>(periods_with_backlog) / count;
            avg_inv_costs_per_policy_per_traj.at(policy_count).at(i) = static_cast<double>(tot_inv_costs_per_traj);
            avg_backlog_costs_per_policy_per_traj.at(policy_count).at(i) = static_cast<double>(tot_backlog_costs_per_traj);
            avg_tot_costs_per_policy_per_traj.at(policy_count).at(i) = static_cast<double>(BS_trajectory.CumulativeReturn);
            avg_tot_costs_per_policy_per_traj2.at(policy_count).at(i) = static_cast<double>(BS_trajectory.CumulativeReturn);
        }
        ++policy_count;

    }

    std::cout << " averaging the results " << std::endl;
    // To compute the averages per policy
    vector<vector<vector<double>>> avg_inv_per_node_per_policy_per_vertpos_3vec(eval_mdp.maxTree - eval_mdp.startMeanDem + 1, vector<vector<double>>(policies.size(), vector<double>(eval_mdp.NumberofNodes, 0.0)));
    vector<vector<double>> avg_inv_per_node_per_policy_vec(policies.size(), vector<double>(eval_mdp.NumberofNodes, 0.0));
    vector<double> avg_perc_periods_with_backlog_per_policy(policies.size(), 0.0);
    vector<double> avg_inv_costs_per_policy(policies.size(), 0.0);
    vector<double> avg_backlog_costs_per_policy(policies.size(), 0.0);
    vector<double> avg_tot_costs_per_policy(policies.size(), 0.0);
    vector<double> avg_tot_costs_per_policy2(policies.size(), 0.0);

    for (int64_t p = 0; p < policies.size(); p++) {
        for (int64_t i = 0; i < numTrajectories; i++) {
            for (int64_t n = 0; n < eval_mdp.NumberofNodes; n++) {
                avg_inv_per_node_per_policy_vec.at(p).at(n) += avg_inv_per_node_per_trajectory_per_policy_vec.at(p).at(i).at(n);
            }
            for (int64_t v = 0; v < eval_mdp.maxTree - eval_mdp.startMeanDem + 1; v++) {
                for (int64_t n = 0; n < eval_mdp.NumberofNodes; n++) {
                    avg_inv_per_node_per_policy_per_vertpos_3vec.at(v).at(p).at(n) += avg_inv_per_node_per_vertpos_per_trajectory_per_policy_vec.at(p).at(i).at(v).at(n);
                }
            }

            avg_perc_periods_with_backlog_per_policy.at(p) += avg_perc_periods_with_backlog_per_policy_per_traj.at(p).at(i);
            avg_inv_costs_per_policy.at(p) += avg_inv_costs_per_policy_per_traj.at(p).at(i);
            avg_backlog_costs_per_policy.at(p) += avg_backlog_costs_per_policy_per_traj.at(p).at(i);
            avg_tot_costs_per_policy.at(p) += avg_tot_costs_per_policy_per_traj.at(p).at(i);
            avg_tot_costs_per_policy2.at(p) += avg_tot_costs_per_policy_per_traj2.at(p).at(i);
        }

        for (int64_t v = 0; v < eval_mdp.maxTree - eval_mdp.startMeanDem + 1; v++) {
            for (int64_t n = 0; n < eval_mdp.NumberofNodes; n++) {
                avg_inv_per_node_per_policy_per_vertpos_3vec.at(v).at(p).at(n) = avg_inv_per_node_per_policy_per_vertpos_3vec.at(v).at(p).at(n) / numTrajectories;
            }
        }



        for (int64_t n = 0; n < eval_mdp.NumberofNodes; n++) {
            avg_inv_per_node_per_policy_vec.at(p).at(n) = avg_inv_per_node_per_policy_vec.at(p).at(n) / numTrajectories;
        }
        avg_perc_periods_with_backlog_per_policy.at(p) = avg_perc_periods_with_backlog_per_policy.at(p) / numTrajectories;
        avg_inv_costs_per_policy.at(p) = avg_inv_costs_per_policy.at(p) / numTrajectories;
        avg_backlog_costs_per_policy.at(p) = avg_backlog_costs_per_policy.at(p) / numTrajectories;
        avg_tot_costs_per_policy.at(p) = avg_tot_costs_per_policy.at(p) / numTrajectories;
        avg_tot_costs_per_policy2.at(p) = avg_tot_costs_per_policy2.at(p) / numTrajectories;

        //double sum = std::accumulate(avg_tot_costs_per_policy_per_traj.at(p).begin(), avg_tot_costs_per_policy_per_traj.at(p).end(), 0.0);
        //double mean = sum / avg_tot_costs_per_policy_per_traj.at(p).size();

        //// to compute standard deviation, but this is very big for the non-stationary setting
        //std::vector<double> diff(avg_tot_costs_per_policy_per_traj.at(p).size());
        //std::transform(avg_tot_costs_per_policy_per_traj.at(p).begin(), avg_tot_costs_per_policy_per_traj.at(p).end(), diff.begin(), [mean](double x) { return x - mean; });
        //double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        //double stdev = std::sqrt(sq_sum / avg_tot_costs_per_policy_per_traj.at(p).size());
    }

    for (int64_t i = 0; i < avg_tot_costs_per_policy.size(); i++) {
        std::cout << " Policy costs of policy " << i << " is " << avg_tot_costs_per_policy.at(i) << std::endl;
        std::cout << " Policy costs of policy " << i << " is " << avg_tot_costs_per_policy2.at(i) << std::endl;
    }

    std::string filename;

    std::string NNodes_str = "_NN" + std::to_string(static_cast<int64_t>(NumberofNodes));
    std::string p_str = "_p" + std::to_string(p);
    std::string assembly_str;
    if (assembly) {
        assembly_str = "_A";
    }
    else {
        assembly_str = "_S";
    }
    std::string stationary_str;
    if (stationary) {
        stationary_str = "_S";
    }
    else {
        stationary_str = "_NS";
    }
    std::string featureEng_str;
    if (featureEng) {
        featureEng_str = "_FE4";
    }
    else {
        featureEng_str = "_NoFE4";
    }
    filename = id + NNodes_str + p_str + assembly_str + stationary_str + featureEng_str + "_";
    std::cout << filename << std::endl;

    //// Save data to csv's to read in python, todo: the API connection
   // int inv = SaveCSV_2vec(avg_inv_per_node_per_policy_vec, filename+"avg_inv_per_node_per_policy_vec.csv", "C:\\Users\\tjdijck\\source\\repos\\data_DRL_inv\\");
    int inv = SaveCSV_3vec(avg_inv_per_node_per_policy_per_vertpos_3vec, filename + "avg_action_per_vertpos_per_node_per_policy_vec2.csv", "C:\\Users\\tjdijck\\source\\repos\\data_DRL_inv\\");
    //inv = SaveCSV_vec(avg_perc_periods_with_backlog_per_policy, filename + "avg_perc_periods_with_backlog_per_policy.csv", "C:\\Users\\tjdijck\\source\\repos\\data_DRL_inv\\");
    //inv = SaveCSV_vec(avg_inv_costs_per_policy, filename + "avg_inv_costs_per_policy.csv", "C:\\Users\\tjdijck\\source\\repos\\data_DRL_inv\\");
    //inv = SaveCSV_vec(avg_backlog_costs_per_policy, filename + "avg_backlog_costs_per_policy.csv", "C:\\Users\\tjdijck\\source\\repos\\data_DRL_inv\\");
   // inv = SaveCSV_vec(avg_tot_costs_per_policy, filename + "avg_tot_costs_per_policy.csv", "C:\\Users\\tjdijck\\source\\repos\\data_DRL_inv\\");
}


int main() {

    // MDP config
    DynaPlex::VarGroup config;
    std::string id = "asml_assembly_nonstationary";
    const int64_t NumberofNodes = 10;
    const int64_t p = 100;
    const bool assembly = true;
    const bool stationary = false;
    const vector<bool> featureEng_vec = { false };
    const bool zerosMaxSystemOutput = false;
    const bool homogeneous_capacity = false; 
    const bool ATO = false; 


    config.Add("id", id);
    config.Add("NumberofNodes", NumberofNodes);
    // Discount factor is now 1 for the entire trajectory, where the trajectory itself is length Geom(discount_factor ) distributed
    config.Add("discount_factor", 1);
    config.Add("p", p);
    config.Add("assembly", assembly);
    config.Add("stationary", stationary);
    //config.Add("featureEngineering", featureEng);
    config.Add("zerosMaxSystemOutput", zerosMaxSystemOutput);
    config.Add("homogeneous_capacity", homogeneous_capacity);
    config.Add("ATO", ATO);


    DynaPlex::VarGroup test_config;
    test_config.Add("warmup_periods", 20);
    test_config.Add("number_of_trajectories", 10000);
    test_config.Add("periods_per_trajectory", 1000000);   // This variable will not be used as the trajectories have geometrically length, which ends earlier

    DynaPlex::VarGroup training_config{
        {"N", 25000},
        {"M", 600},
        {"num_gens", 6}
    };


    std::string folder_name = "20261109 run output revision";

    //TestTrajectories_notParallel();
    TrainandTestPaperResults(folder_name, config, test_config, training_config, featureEng_vec, false, false, false, false, false);
    //ComputeShortfall();
    //ComparePolicies(false, false);
    //TestInventoryperNode(true);


    return 0;
}
