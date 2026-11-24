#include "policies.h"
#include "dynaplex/models/asml_assembly_nonstationary/mdp.h"
#include "dynaplex/error.h"


namespace DynaPlex::Models {
	namespace asml_assembly_nonstationary /*keep this namespace name in line with the name space in which the mdp corresponding to this policy is defined*/
	{
		//MDP and State refer to the specific ones defined in current namespace
		DT_BaseStockPolicy::DT_BaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			:mdp{ mdp }
		{
			config.GetOrDefault("shortfall", shortfall, false);
			config.GetOrDefault("Huhetal", Huhetal, false);
			config.GetOrDefault("balancing_upstream", balancing_upstream, true);
			config.GetOrDefault("balancing_downstream", balancing_downstream, true);
			config.GetOrDefault("balancing_bottleneck", balancing_bottleneck, true);
			config.GetOrDefault("balancing_bottleneck_downstream", balancing_bottleneck_downstream, true);
		}

		int64_t DT_BaseStockPolicy::GetAction(const MDP::State& state) const
		{
			int64_t action = 0;

			// Initializing the balancing variables
			int64_t W_nt = 10000;
			int64_t B_nt = 10000;
			if (balancing_bottleneck || mdp->nonBottleneckNode_vec.at(state.currentNode)) {
				if (balancing_upstream) {
					// Upstream balancing
					if (mdp->parallelUpNodes_2vec.at(state.currentNode).size() > 0) {
						// vector containing all W_nt
						vector<int64_t> W_nt_vec;
						// W_nt, i.e., the minimum of the W_nt_vec

						// For all upstream nodes of state.currentNode
						for (int64_t m : mdp->parallelUpNodes_2vec.at(state.currentNode)) {
							// Compute W_{curnode - m,t}
							int64_t W_kt = 0;
							// For the upstream node of which we take a part of their inventory, we have
							const int64_t current_leadtime = mdp->Leadtime_vec[m];
							for (int64_t i = 0; i <= current_leadtime - (mdp->Echelonlead_vec[m] - mdp->Echelonlead_vec[state.currentNode]); i++) {
								W_kt += state.StateVectors.at(mdp->frontposition_vec[m] + i);
							}
							// Then we add the echelon inventory of the successors node of upstream balance node
							if (mdp->ESuccessor_2vec.at(m).size() > 0) {
								W_kt += state.echelonInv_vec.at(mdp->ESuccessor_2vec.at(m).at(0));
							}
							// For equivalent nodes with higher index, i.e., more upstream, we have to add an additional period since there is not yet ordered there
							if (mdp->Echelonlead_vec[m] == mdp->Echelonlead_vec[state.currentNode]) {
								int64_t minOrderEq = 0;
								minOrderEq = mdp->CapacityofNodes_vec.at(m);
								// Check material availability
								for (auto pred : mdp->EPredecessor_2vec[m]) {
									if (state.StateVectors.at(mdp->frontposition_vec[pred]) < minOrderEq) {
										minOrderEq = state.StateVectors.at(mdp->frontposition_vec[pred]);
									}
								}
								int64_t basestock_m = 0;
								if (!shortfall) {
									basestock_m += mdp->BSlevelsDT_2vec[state.verticalpos][m];
								}
								else if (shortfall && !Huhetal) {
									basestock_m += mdp->BSlevelsDT_short_2vec[state.verticalpos][m];
								}
								else {
									basestock_m += mdp->BSlevelsDT_shortsingle_2vec[state.verticalpos][m];
								}
								if (basestock_m - state.echelonInv_vec.at(m) < minOrderEq) {
									minOrderEq = basestock_m - state.echelonInv_vec.at(m);
								}
								W_kt += minOrderEq;
							}
							W_nt_vec.push_back(W_kt);
						}
						W_nt = std::min_element(W_nt_vec.begin(), W_nt_vec.end())[0];
					}

				}
				if (balancing_bottleneck_downstream || mdp->nonBottleneckNode_vec.at(state.currentNode)) {
					if (balancing_downstream) {
						// balance with downstream nodes
						if (mdp->parallelDownNodes_2vec.at(state.currentNode).size() > 0) {
							// vector containing all B_nt
							vector<int64_t> B_nt_vec;
							// Compute B_{curnode + m,t}
							// Downstream balancing
							for (int64_t m : mdp->parallelDownNodes_2vec.at(state.currentNode)) {
								int64_t B_kt = 0;
								B_kt = state.echelonInv_vec.at(m);
								// Consider the capacity of the downstream balance node as maximum future production decisions of that downstream balance node
								// for a total of the difference in echelon lead time time
								B_kt += (mdp->Echelonlead_vec[state.currentNode] - mdp->Echelonlead_vec[m]) * mdp->CapacityofNodes_vec[m];
								B_nt_vec.push_back(B_kt);
							}
							B_nt = std::min_element(B_nt_vec.begin(), B_nt_vec.end())[0];
						}
					}
				}
			}
			// Set order up to level
			int64_t min_S_W_B = 0;
			if (!shortfall) {
				min_S_W_B += mdp->BSlevelsDT_2vec[state.verticalpos][state.currentNode];
			}
			else if (shortfall && !Huhetal) {
				min_S_W_B += mdp->BSlevelsDT_short_2vec[state.verticalpos][state.currentNode];
			}
			else {
				min_S_W_B += mdp->BSlevelsDT_shortsingle_2vec[state.verticalpos][state.currentNode];
			}
			// If no balancing, W_nt and B_nt are very high
			if (W_nt < min_S_W_B) {
				min_S_W_B = W_nt;
			}
			if (B_nt < min_S_W_B) {
				min_S_W_B = B_nt;
			}
			//action = min_S_W_B - echeloninv - int64_t(mdp->minimumorder);
			action = min_S_W_B - state.echelonInv_vec.at(state.currentNode) - int64_t(mdp->minimumorder);

			if (action <= 0)
			{
				// Action of producing the minimum order quantity is always possible
				action = 0;
				return action;
			}
			else
			{
				// No production above capacity
				if (action > mdp->CapacityofNodes_vec[state.currentNode] - mdp->minimumorder)
				{
					action = mdp->CapacityofNodes_vec[state.currentNode] - mdp->minimumorder;
				}
				while (action >= 0)
				{
					if (mdp->IsAllowedAction(state, action))
					{
						return action;
					}
					action--;
				}
			}
			return 0;
		}

		Cur_D_BaseStockPolicy::Cur_D_BaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			:mdp{ mdp }
		{
			config.GetOrDefault("shortfall", shortfall, false);
			config.GetOrDefault("Huhetal", Huhetal, false);
			config.GetOrDefault("balancing_upstream", balancing_upstream, true);
			config.GetOrDefault("balancing_downstream", balancing_downstream, true);
		}

		int64_t Cur_D_BaseStockPolicy::GetAction(const MDP::State& state) const
		{
			int64_t action = 0;
			// calculate echelon inv but not all nodes are in the echelon
			int64_t echeloninv = 0;
			for (auto j : mdp->ESuccessors_2vec[state.currentNode]) {
				const int64_t current_leadtime = mdp->Leadtime_vec[j];
				for (int64_t i = 0; i <= current_leadtime; i++) {
					echeloninv += state.StateVectors.at(mdp->frontposition_vec[j] + i);
				}
			}


			// Initializing the balancing variables
			int64_t W_nt = 10000;
			int64_t B_nt = 10000;

			if (balancing_upstream) {
				// Upstream balancing
				if (mdp->parallelUpNodes_2vec.at(state.currentNode).size() > 0) {
					// vector containing all W_nt
					vector<int64_t> W_nt_vec;
					// W_nt, i.e., the minimum of the W_nt_vec

					// For all upstream nodes of state.currentNode
					for (int64_t m : mdp->parallelUpNodes_2vec.at(state.currentNode)) {
						// Compute W_{curnode - m,t}
						int64_t W_kt = 0;
						// For the upstream node of which we take a part of their inventory, we have
						const int64_t current_leadtime = mdp->Leadtime_vec[m];
						for (int64_t i = 0; i <= current_leadtime - (mdp->Echelonlead_vec[m] - mdp->Echelonlead_vec[state.currentNode]); i++) {
							W_kt += state.StateVectors.at(mdp->frontposition_vec[m] + i);
						}
						// Then we add the echelon inventory of the successors node of upstream balance node
						if (mdp->ESuccessor_2vec.at(m).size() > 0) {
							W_kt += state.echelonInv_vec.at(mdp->ESuccessor_2vec.at(m).at(0));
						}
						// For equivalent nodes with higher index, i.e., more upstream, we have to add an additional period since there is not yet ordered there
						// TODO: do we just want to add capacity then? Or based on material availability?
						if (mdp->Echelonlead_vec[m] == mdp->Echelonlead_vec[state.currentNode]) {
							W_kt += mdp->CapacityofNodes_vec.at(m);
						}
						W_nt_vec.push_back(W_kt);
					}
					W_nt = std::min_element(W_nt_vec.begin(), W_nt_vec.end())[0];
				}
			}
			if (balancing_downstream) {

				// balance with downstream nodes
				if (mdp->parallelDownNodes_2vec.at(state.currentNode).size() > 0) {
					// vector containing all B_nt
					vector<int64_t> B_nt_vec;
					// Compute B_{curnode + m,t}
					// Downstream balancing
					for (int64_t m : mdp->parallelDownNodes_2vec.at(state.currentNode)) {
						int64_t B_kt = 0;
						B_kt = state.echelonInv_vec.at(m);
						// Consider the capacity of the downstream balance node as maximum future production decisions of that downstream balance node
						// for a total of the difference in echelon lead time time
						B_kt += (mdp->Echelonlead_vec[state.currentNode] - mdp->Echelonlead_vec[m]) * mdp->CapacityofNodes_vec[m];
						B_nt_vec.push_back(B_kt);
					}
					B_nt = std::min_element(B_nt_vec.begin(), B_nt_vec.end())[0];
				}
			}
		
			// Set order up to level
			int64_t min_S_W_B = 0;
			if (!shortfall) {
				min_S_W_B += mdp->BSlevelsCN_2vec[state.verticalpos][state.currentNode];
			}
			else if (shortfall && !Huhetal) {
				min_S_W_B += mdp->BSlevelsCN_short_2vec[state.verticalpos][state.currentNode];
			}
			else {
				min_S_W_B += mdp->BSlevelsCN_shortsingle_2vec[state.verticalpos][state.currentNode];
			}
			// If no balancing, W_nt and B_nt are very high
			if (W_nt < min_S_W_B) {
				min_S_W_B = W_nt;
			}
			if (B_nt < min_S_W_B) {
				min_S_W_B = B_nt;
			}
			action = min_S_W_B - echeloninv - int64_t(mdp->minimumorder);

			if (action <= 0)
			{
				// Action of producing the minimum order quantity is always possible
				action = 0;
				return action;
			}
			else
			{
				// No production above capacity
				if (action > mdp->CapacityofNodes_vec[state.currentNode] - mdp->minimumorder)
				{
					action = mdp->CapacityofNodes_vec[state.currentNode] - mdp->minimumorder;
				}
				while (action >= 0)
				{
					if (mdp->IsAllowedAction(state, action))
					{
						return action;
					}
					action--;
				}
			}
			return 0;
		}

	}
}