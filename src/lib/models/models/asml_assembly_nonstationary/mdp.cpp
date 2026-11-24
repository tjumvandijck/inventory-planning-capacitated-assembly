#include "dynaplex/models/asml_assembly_nonstationary/mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"
#include <numeric>
#include <algorithm>

namespace DynaPlex::Models {
	namespace asml_assembly_nonstationary /*keep this in line with id below and with namespace name in header*/
	{
		VarGroup MDP::GetStaticInfo() const
		{
			VarGroup vars;
			vars.Add("valid_actions", MaxOrderSize + 1);
			vars.Add("discount_factor", discount_factor);
			vars.Add("horizon_type", "finite");
			vars.Add("warmup_despite_finite_horizon", true);

			//potentially add any stuff that was computed for diagnostics purposes
			//not used by dynaplex framework itself. 
			VarGroup diagnostics{};
			diagnostics.Add("p", p);
			//diagnostics.Add("discount_factor_geom", discount_factor_geom);
			diagnostics.Add("assembly", assembly);
			diagnostics.Add("stationary", stationary);
			vars.Add("diagnostics", diagnostics);

			return vars;
		}

		// IsAllowedAction checks whether there is enough material to produce the action, where infinite material availability is assumed for the most upstream node
		// If in small cases, the exact benchmark assumes a maximum capacity on the inventory per node
		// Note that 0 is most upstream node, and N most downstream node
		bool MDP::IsAllowedAction(const State& state, int64_t action) const
		{
			// Compute inventory position of the current node
			int64_t nodeinv = 0;
			const int64_t current_leadtime = Leadtime_vec[state.currentNode];
			for (int64_t i = 0; i <= current_leadtime; i++) {
				nodeinv += state.StateVectors.at(frontposition_vec[state.currentNode] + i);
			}

			// Check capacity of own node
			if (action + minimumorder > CapacityofNodes_vec.at(state.currentNode)) {
				return false;
			}

			// Check material availability
			for (auto pred : EPredecessor_2vec[state.currentNode]) {
				if (action + minimumorder > state.StateVectors.at(frontposition_vec[pred])) {
					return false;
				}
			}
			// check inventory capacity
			if (action + minimumorder + nodeinv > MaxInvatNodes_vec[state.currentNode]) {
				return false;
			}
			return true;
		}

		//// for modifystate with action there are differences between most upstream node, middle nodes and most downstream node
		//// most upstream node; has infinite supply availability and does not need to subtract action from predecessors inventory; produce action; converge one node downstream
		//// middle nodes; produce action, subtract this from predecessors inventory; converge one node downstream
		//// most downstream node; produce action, subtract this from predecessors inventory, converge to most upstream node (node 0), calculate all costs backlog and holding, process converges to awaitevent
		double MDP::ModifyStateWithAction(State& state, int64_t action) const
		{

			// Implement the production decision
			state.StateVectors.at(backposition_vec[state.currentNode]) += action + minimumorder;

			// Remove the production decision from the on-hand inv at predecessors
			for (auto pred : EPredecessor_2vec[state.currentNode]) {
				state.StateVectors.at(frontposition_vec[pred]) -= (action + minimumorder);
				if (featureEngineering) {
/*					for (int64_t i = frontpositions_FE_vec.at(pred); i <= backpositions_FE_vec.at(pred); i++) {
						state.cumInvNode_vec.at(i) -= (action + minimumorder);
					}		*/			
					for (int64_t i = frontpositions_eff_FE_vec.at(pred); i <= backpositions_eff_FE_vec.at(pred); i++) {
						state.cumInvNode_eff_vec.at(i) -= (action + minimumorder);
					}
				}
			}

			if (featureEngineering) {
				// Adjust the feature engineering vectors
				
				int64_t differenceMaxOutput2 = action + minimumorder - state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(state.currentNode) + Leadtime_vec.at(state.currentNode)) + state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(state.currentNode) + Leadtime_vec.at(state.currentNode) - 1);
				state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(state.currentNode) + Leadtime_vec.at(state.currentNode)) = state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(state.currentNode) + Leadtime_vec.at(state.currentNode) - 1) + action + minimumorder;

				// Check
				if (differenceMaxOutput2 > 0) {
					std::cout << " Current node with error: " << state.currentNode << std::endl;
					throw DynaPlex::Error(" Difference cannot be greater than zero.");
				}


				if (differenceMaxOutput2 != 0) {

					if (EPredecessor_2vec.at(state.currentNode).size() != 0) {
						recomputeFeatures_eff_singlevec(state, true);
					}
					else {
						recomputeFeatures_eff_singlevec(state, false);
					}
				}


			}

			// Add the production quantity to the echelon inventory position of that node
			// For upstream nodes, the echelon inv position remains similar
			state.echelonInv_vec.at(state.currentNode) += action + minimumorder;

			// Make decision for the next node
			// Starting from the most downstream node
			if (state.currentNode != 0) {
				state.currentNode--;


				//vector<int64_t> maxSystem_woBranch_vec;
				//maxSystem_woBranch_vec = maxSystemOutputWithout_eff_singlevec(state);
				//state.features_vec = {};
				//// Add the features
				//state.features_vec.push_back(state.echelonInv_vec.at(state.currentNode));
				//state.features_vec.push_back(Echelonlead_vec[state.currentNode]);

				//// Echelon lead - 2, - 1 is because we want to have the max system output before production decision and other -1 is as index starts at 0
				//// If echelon lead = 2, then we want to also start from index 0
				//for (int64_t l = Echelonlead_vec.at(state.currentNode) - 2; l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 2; l++)
				//{
				//	if (zerosMaxSystemOutput) {
				//		// Max system output 
				//		state.features_vec.push_back(state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(NumberofNodes - 1) + l) / avgdem_lt_vec.at(state.verticalpos).at(l));
				//		// Max system output without branch
				//		state.features_vec.push_back(maxSystem_woBranch_vec.at(l) / avgdem_lt_vec.at(state.verticalpos).at(l));

				//	}
				//	else {
				//		// Max system output 
				//		state.features_vec.push_back(state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(NumberofNodes - 1) + l));
				//		// Max system output without branch
				//		state.features_vec.push_back(maxSystem_woBranch_vec.at(l));
				//	}
				//}
				//state.features_vec.push_back(state.currentNode);
				//state.features_vec.push_back(state.verticalpos);


			}
			// If production has been decided for each node, continue with MSWE
			else {
				state.cat = StateCategory::AwaitEvent();
				state.currentNode = NumberofNodes - 1;
			}




			return 0.0;
		}

		//// The current dynamics first observe an event. In this event, first the material inflow happens; 
		//// materials for which the production lead time expires become available as stock and other products in production have one time unit less to be ready
		//// After this, demand is realized and met is possible
		double MDP::ModifyStateWithEvent(State& state, const MDP::Event& event) const
		{
			//after processing this event, we await an action
			state.cat = StateCategory::AwaitAction();

			// Determine demand + movement + reaching of final state
			int64_t bern = 0;
			int64_t finalstate = 0;
			int64_t demand = 0;
			// If no warm up phase, movement and possibility of reaching final state
			if (!state.warmUp) {
				// Here c++ always rounds down, thus 0.9 becomes 0
				finalstate = event / (demProbs_tr_2vec.at(state.verticalpos).size() / 2);
				if (finalstate == 1) {
					// Here, we still incur costs after final state has reached 
					// Equal to having the Geom distribution to #trials for first success
					// Old version - number of failures before first success
					state.cat = StateCategory::Final();
					bern = (event - (demProbs_tr_2vec.at(state.verticalpos).size() / 2)) / noiseLength;

				}
				else {
					bern = event / noiseLength;

				}
				// Update vertical position
				if (state.verticalpos != maxTree - startMeanDem) {
					state.verticalpos += bern;
				}
			}

			demand = startMeanDem + state.verticalpos + (event % noiseLength + startNoise);

			// Remove demand from most downstream on hand inventory
			state.StateVectors.at(downstreamposition) -= demand;
			auto onHandInv = state.StateVectors.at(downstreamposition);


			// Remove demand from all inventory positions
			for (int64_t i = 0; i < NumberofNodes; i++) {
				state.echelonInv_vec.at(i) -= demand;
			}

			double returnval = 0.0;
			// Holding costs not most downstream node
			for (int64_t i = 0; i < (NumberofNodes - 1); i++) {
				returnval += (double)state.StateVectors.at(frontposition_vec[i]) * h_vec[i];
			}

			// Demand can be satisfied from inventory in stock
			if (onHandInv >= 0) {
				// Holding costs
				returnval += (double)onHandInv * h_vec[NumberofNodes - 1];
			}
			// Lost sales
			else if (onHandInv < -MaxBacklog) {
				// std::cout << "Lost sales in period " << state.period << std::endl;
				// Calculate the lost demand, which is every demand below -maxbacklog
				int64_t lostsales = (-MaxBacklog - onHandInv);
				state.StateVectors.at(downstreamposition) = -MaxBacklog;
				// lost sales and backlog penalty
				returnval += (double)(lostsales * lostsalespenalty) - (double)state.StateVectors.at(downstreamposition) * p;

				// Update all inventory positions
				for (int64_t i = 0; i < NumberofNodes; i++)
				{
					state.echelonInv_vec.at(i) += lostsales;
				}
				// This is required for removing the demand subtracted from most downstream node for FE
				demand -= lostsales;
			}
			// Here there is backlog
			else {
				// backlog penalty
				returnval -= (double)onHandInv * p;
			}

			state.period++;
			//Here we go one time step further!
			for (int64_t i = 0; i < NumberofNodes; i++) {
				const int64_t pos = frontposition_vec[i];
				const int64_t current_leadtime = Leadtime_vec[i];
				auto onHand = state.StateVectors.at(pos);
				// We assume minimum leadtime of 1
				state.StateVectors.at(pos) = onHand + state.StateVectors.at(pos + 1);
				for (int64_t j = 1; j < current_leadtime; j++) {
					state.StateVectors.at(pos + j) = state.StateVectors.at(pos + j + 1);
				}
				state.StateVectors.at(pos + current_leadtime) = 0;
			}

			// Demand will just reduce the cumInvVec of most downstream node and all others will just iterate
			// Only account for the lost sales, do not just subtract demand but demand minus lost sales. Then it is done!
			// Still need to incorporate the maximum system output without a certain branch!
			if (featureEngineering) {
				//// First subtract demand from most downstream cumInvNode vec
				//for (int64_t i = frontpositions_FE_vec.at(NumberofNodes - 1); i <= backpositions_FE_vec.at(NumberofNodes - 1); i++) {
				//	state.cumInvNode_vec.at(i) -= demand;
				//}
				for (int64_t i = frontpositions_eff_FE_vec.at(NumberofNodes - 1); i <= backpositions_eff_FE_vec.at(NumberofNodes - 1); i++) {
					state.cumInvNode_eff_vec.at(i) -= demand;
				}

				for (int64_t i = 0; i < NumberofNodes; i++) {
					for (int64_t j = frontpositions_eff_FE_vec.at(i); j < backpositions_eff_FE_vec.at(i); j++) {
						state.cumInvNode_eff_vec.at(j) = state.cumInvNode_eff_vec.at(j + 1);
					}


					// If the node has no predecessor, just add the capacity to the end of the list
					if (EPredecessor_2vec.at(i).size() == 0) {
						state.cumInvNode_eff_vec.at(backpositions_eff_FE_vec.at(i)) = state.cumInvNode_eff_vec.at(backpositions_eff_FE_vec.at(i) - 1) + CapacityofNodes_vec.at(i);
					}
					// Else, the maximum system output of that node depends on the capacity and cumInvNodes of the predecessors node
					else {
						// Note that here the current node has not yet ordered
						int64_t min_max_additional_output2 = 0;
						min_max_additional_output2 = CapacityofNodes_vec.at(i) * (endEchelonLead - Echelonlead_vec.at(i));
						// The index is + 1
						int64_t l = endEchelonLead - Echelonlead_vec.at(i) - 1;
						for (auto node : EPredecessor_2vec.at(i)) {
							if ((EPredecessor_2vec.at(node).size() == 0) && (l > Leadtime_vec.at(node))) {
								if (state.cumInvNode_eff_vec.at(backpositions_eff_FE_vec.at(node)) + CapacityofNodes_vec.at(node) * (l - Leadtime_vec.at(node)) < min_max_additional_output2) {
									min_max_additional_output2 = state.cumInvNode_eff_vec.at(backpositions_eff_FE_vec.at(node)) + CapacityofNodes_vec.at(node) * (l - Leadtime_vec.at(node));
								}
							}
							else {
								if (frontpositions_eff_FE_vec.at(node) + l != backpositions_eff_FE_vec.at(node)) {
									std::cout << " Error this should be the backposition for nodes that have a predecessor " << std::endl;
								}
								if (state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(node) + l) < min_max_additional_output2) {
									min_max_additional_output2 = state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(node) + l);
								}
							}

						}
						// Cannot add more than capacity, -1 as we have not ordered yet
						if (state.cumInvNode_eff_vec.at(backpositions_eff_FE_vec.at(i) - 1) + CapacityofNodes_vec.at(i) - state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(i) + Leadtime_vec.at(i) - 1) < min_max_additional_output2) {
							min_max_additional_output2 = state.cumInvNode_eff_vec.at(backpositions_eff_FE_vec.at(i) - 1) + CapacityofNodes_vec.at(i) - state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(i) + Leadtime_vec.at(i) - 1);
						}
						// Not yet ordered, so lead time  - 1
						state.cumInvNode_eff_vec.at(backpositions_eff_FE_vec.at(i)) = state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(i) + Leadtime_vec.at(i) - 1) + min_max_additional_output2;

					}
				}

			}


			//vector<int64_t> maxSystem_woBranch_vec;
			//maxSystem_woBranch_vec = maxSystemOutputWithout_eff_singlevec(state);
			//state.features_vec = {};
			//// Add the features
			//state.features_vec.push_back(state.echelonInv_vec.at(state.currentNode));
			//state.features_vec.push_back(Echelonlead_vec[state.currentNode]);

			//// Echelon lead - 2, - 1 is because we want to have the max system output before production decision and other -1 is as index starts at 0
			//// If echelon lead = 2, then we want to also start from index 0
			//for (int64_t l = Echelonlead_vec.at(state.currentNode) - 2; l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 2; l++)
			//{
			//	if (zerosMaxSystemOutput) {
			//		// Max system output 
			//		state.features_vec.push_back(state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(NumberofNodes - 1) + l) / avgdem_lt_vec.at(state.verticalpos).at(l));
			//		// Max system output without branch
			//		state.features_vec.push_back(maxSystem_woBranch_vec.at(l) / avgdem_lt_vec.at(state.verticalpos).at(l));

			//	}
			//	else {
			//		// Max system output 
			//		state.features_vec.push_back(state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(NumberofNodes - 1) + l));
			//		// Max system output without branch
			//		state.features_vec.push_back(maxSystem_woBranch_vec.at(l));
			//	}
			//}
			//state.features_vec.push_back(state.currentNode);
			//state.features_vec.push_back(state.verticalpos);

			return returnval;
		}

		MDP::State MDP::GetInitialState() const
		{
			int64_t size = 0;
			for (int64_t i = 0; i < NumberofNodes; i++) {
				size += Leadtime_vec[i] + 1;
			}

			auto queues = Queue<int64_t>{};
			vector<int64_t> echelonInv_vec;
			std::vector<int64_t> cumInvNode_vec;
			std::vector<int64_t> cumInvNode_eff_vec;

			queues.reserve(size);
			for (int64_t i = 0; i < NumberofNodes; i++)
			{
				// Note that the initial inventories are after material inflow. Hence, the capacity on-hand is approximately capacity - mean demand on-hand before inflow
				// Make sure that in pipeline is never higher than the capacity if we want to use the exact solver
				if (i == NumberofNodes - 1) {
					queues.push_back(CapacityofNodes_vec[i] + 1);
				}
				else {
					queues.push_back(CapacityofNodes_vec[i]);
					//queues.push_back(1);
				}
				//queue.push_back(CapacityofNodes[i]);
				for (int64_t j = 0; j < Leadtime_vec[i] - 1; j++)
				{
					queues.push_back(CapacityofNodes_vec[i]);
				}
				queues.push_back(0);

			}
			for (int64_t i = 0; i < NumberofNodes; i++) {
				int64_t echeloninv = 0;
				for (int64_t j : ESuccessors_2vec[i]) {
					const int64_t current_leadtime = Leadtime_vec[j];
					for (int64_t l = 0; l <= current_leadtime; l++) {
						echeloninv += queues.at(frontposition_vec[j] + l);
					}
				}
				echelonInv_vec.push_back(echeloninv);
			}

			// Initialize the feature for feature engineering
			vector<vector<int64_t>> Echeloninv_2vec;
			vector<int64_t> maxSystemOutput_vec;
			vector<vector<int64_t>> maxOutput_woBranch_2vec;
			for (int64_t n = 0; n <= NumberofNodes - 1; n++) {
				maxOutput_woBranch_2vec.push_back({});
			}
			if (featureEngineering) {

				for (int64_t n = NumberofNodes - 1; n >= 0; n--) {
					Echeloninv_2vec.push_back({});
				}

				for (int64_t n = 0; n < NumberofNodes; n++) {
					Echeloninv_2vec.at(n).push_back(queues.at(frontposition_vec[n]));
					// not yet ordered for this node
					for (int64_t l = 1; l < Leadtime_vec[n]; l++) {
						Echeloninv_2vec[n].push_back(Echeloninv_2vec[n].at(l - 1) + queues.at(frontposition_vec[n] + l));
					}
					if (EPredecessor_2vec.at(n).size() == 0) {
						for (int64_t k = Echelonlead_vec.at(n); k < endEchelonLead; k++) {
							Echeloninv_2vec[n].push_back(Echeloninv_2vec[n].back() + CapacityofNodes_vec.at(n));
						}
					}

				}

				int64_t min_max_additional_output;
				// Backward induction to compute max output from nodes with predecessors
				// Maximum system output at all future periods for the entire system
				for (int64_t n = 0; n < NumberofNodes; n++) {
					// Greater than zero or than one
					// TODO: check in serial cases! Then only zero if nothing is in parallel
					if (EPredecessor_2vec.at(n).size() > 0) {
						vector<int64_t> minmaxOutput_vec;
						// Not yet ordered for this node
						for (int64_t l = 0; l < endEchelonLead - Echelonlead_vec.at(n); l++) {
							min_max_additional_output = CapacityofNodes_vec.at(n) * (l + 1);

							for (auto node : EPredecessor_2vec.at(n)) {
								if (Echeloninv_2vec.at(node).at(l) < min_max_additional_output) {
									min_max_additional_output = Echeloninv_2vec.at(node).at(l);
								}
							}
							// Cannot add more than capacity
							// leadtime_vec - 1 as this node has not ordered yet
							if (Echeloninv_2vec.at(n).at(Echeloninv_2vec.at(n).size() - 1) + CapacityofNodes_vec.at(n) - Echeloninv_2vec.at(n).at(Leadtime_vec.at(n) - 1) < min_max_additional_output) {
								min_max_additional_output = Echeloninv_2vec.at(n).at(Echeloninv_2vec.at(n).size() - 1) + CapacityofNodes_vec.at(n) - Echeloninv_2vec.at(n).at(Leadtime_vec.at(n) - 1);
							}

							Echeloninv_2vec.at(n).push_back(Echeloninv_2vec.at(n).at(Leadtime_vec.at(n) - 1) + min_max_additional_output);
						}
					}
				}

				maxSystemOutput_vec = Echeloninv_2vec.at(NumberofNodes - 1);
				for (int64_t n = 0; n < NumberofNodes; n++) {
					for (int64_t i : Echeloninv_2vec.at(n)) {
						cumInvNode_vec.push_back(i);
					}
				}

				int64_t count = 0;
				for (int64_t n = 0; n < NumberofNodes; n++) {
					count = 0;
					for (int64_t i : Echeloninv_2vec.at(n)) {
						if ((EPredecessor_2vec.at(n).size() == 0) && (count <= Leadtime_vec.at(n))) {
							cumInvNode_eff_vec.push_back(i);
						}
						else if (EPredecessor_2vec.at(n).size() > 0) {
							cumInvNode_eff_vec.push_back(i);
						}
						count++;
					}
				}

			}



			State state{};
			state.cat = StateCategory::AwaitAction();
			state.StateVectors = queues;
			state.currentNode = NumberofNodes - 1;
			state.echelonInv_vec = vector<int64_t>(echelonInv_vec);

			//state.cumInvNode_vec = vector<int64_t>(cumInvNode_vec);
			state.cumInvNode_eff_vec = vector<int64_t>(cumInvNode_eff_vec);
			state.features_vec = vector<int64_t>(cumInvNode_eff_vec);

			state.verticalpos = 0;
			state.period = 0;

			return state;
		}

		void MDP::GetFeatures(const State& state, DynaPlex::Features& features) const {



			if (featureEngineering) {
				vector<int64_t> maxSystem_woBranch_vec;
				maxSystem_woBranch_vec = maxSystemOutputWithout_eff_singlevec(state);

				// Add the features
				features.Add(state.echelonInv_vec.at(state.currentNode));
				features.Add(Echelonlead_vec[state.currentNode]);

				//features.Add(state.currentNode);
				//features.Add(state.verticalpos);

				// Echelon lead - 2, - 1 is because we want to have the max system output before production decision and other -1 is as index starts at 0
				// If echelon lead = 2, then we want to also start from index 0
				for (int64_t l = Echelonlead_vec.at(state.currentNode) - 2; l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 2; l++)
                {
					if (zerosMaxSystemOutput) {
						// Max system output 
						//features.Add(state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(NumberofNodes - 1) + l) / avgdem_lt_vec.at(state.verticalpos).at(l));
						//// Max system output without branch
						//features.Add(maxSystem_woBranch_vec.at(l) / avgdem_lt_vec.at(state.verticalpos).at(l));
						
						// Max system output 
						features.Add(state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(NumberofNodes - 1) + l));
						// Max system output without branch
						features.Add(maxSystem_woBranch_vec.at(l));



					}
					else {
						// Max system output 
						features.Add(state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(NumberofNodes - 1) + l));
						// Max system output without branch
						features.Add(maxSystem_woBranch_vec.at(l));
					}
				}

				//for (auto inv : state.StateVectors)
				//{
				//	features.Add(inv);
				//}
				features.Add(state.currentNode);
				features.Add(state.verticalpos);

			}
			else {
				for (auto inv : state.StateVectors)
				{
					features.Add(inv);
				}
				features.Add(state.currentNode);
				features.Add(state.verticalpos);
			}
		}

		void MDP::GetExactFeatureRepr(const State& state, DynaPlex::Features& features) const {
			for (auto inv : state.StateVectors)
			{
				features.Add(inv);
			}
			features.Add(state.currentNode);
			features.Add(state.verticalpos);
		}

		MDP::MDP(const VarGroup& config)
		{


			// Variables to determine the MDP configuration
			config.Get("NumberofNodes", NumberofNodes);
			config.Get("p", p);
			config.Get("assembly", assembly);
			config.Get("stationary", stationary);
			config.Get("featureEngineering", featureEngineering);
			config.GetOrDefault("zerosMaxSystemOutput", zerosMaxSystemOutput, false);
			config.GetOrDefault("homogeneous_capacity", homogeneous_capacity, false);
			config.GetOrDefault("ATO", ATO, false);
			config.GetOrDefault("instance", instance, 0);

			if (stationary) {
				discount_factor_geom = 0.975;
			}
			else {
				discount_factor_geom = 0.9444444;
			}
			std::cout << "Discount factor geometric distribution = " << discount_factor_geom << std::endl;

			// Load the system variables based on the configuration
			auto precomputer = preComputerNS::getPreComputer(NumberofNodes, p, assembly, discount_factor_geom, stationary, instance, homogeneous_capacity, ATO);
			auto ps = precomputer.loadSystem();

			Leadtime_vec = ps.Leadtime_vec;
			h_vec = ps.h_vec;
			CapacityofNodes_vec = ps.CapacityofNodes_vec;
			moveProb_2vec = ps.moveProb_2vec;
			noiseProb_vec = ps.noiseProb_vec;
			startNoise = ps.startNoise;
			maxTree = ps.maxTree;
			startMeanDem = ps.startMeanDem;
			BOMmatrix_2vec = ps.BOMmatrix_2vec;
			lostsalespenalty = ps.lostsalespenalty;
			MaxInvatNodes_vec = ps.MaxInvatNodes_vec;
			MaxBacklog = ps.MaxBacklog;
			minimumorder = ps.minimumorder;
			noiseLength = ps.noiseLength;
			ESuccessors_2vec = ps.ESuccessors_2vec;
			ESuccessor_2vec = ps.ESuccessor_2vec;
			EPredecessor_2vec = ps.EPredecessor_2vec;
			EqSerialParam_2vec = ps.EqSerialParam_2vec;
			Echelonlead_vec = ps.Echelonlead_vec;
			SerialEq_vec = ps.SerialEq_vec;
			SameSerial_vec = ps.SameSerial_vec;
			parallelDownNodes_2vec = ps.parallelDownNodes_2vec;
			parallelUpNodes_2vec = ps.parallelUpNodes_2vec;
			frontposition_vec = ps.frontposition_vec;
			backposition_vec = ps.backposition_vec;
			downstreamposition = ps.downstreamposition;
			nonBottleneckNode_vec = ps.nonBottleneckNode_vec;			// True if the node is a non-bottleneck node


			BSlevelsDT_2vec = ps.BSlevelsDT_2vec;
			BSlevelsDT_short_2vec = ps.BSlevelsDT_short_2vec;
			BSlevelsDT_shortsingle_2vec = ps.BSlevelsDT_shortsingle_2vec;
			BSlevelsCN_2vec = ps.BSlevelsCN_2vec;
			BSlevelsCN_short_2vec = ps.BSlevelsCN_short_2vec;
			BSlevelsCN_shortsingle_2vec = ps.BSlevelsCN_shortsingle_2vec;
			demProbs_tr_2vec = ps.demProbs_tr_2vec;
			avgdem_lt_vec = ps.avgdem_lt_vec;

			std::cout << " Noise length is " << noiseLength << std::endl;
			//providing discount_factor is optional. 
			if (config.HasKey("discount_factor"))
				config.Get("discount_factor", discount_factor);
			else
				discount_factor = 1.0;



			for (int64_t i = 0; i < demProbs_tr_2vec.size(); i++) {
				demand_dist = DiscreteDist::GetCustomDist(demProbs_tr_2vec.at(i));
				demand_dist_vec.push_back(std::move(demand_dist));
			}
			demand_dist = DiscreteDist::GetCustomDist(demProbs_tr_2vec.at(0));
			// +1 for the echelon inv positions
			leadtime_size = Echelonlead_vec[0] - Echelonlead_vec[NumberofNodes - 1] + 1;
			endEchelonLead = Echelonlead_vec[0] + leadtime_size - 1;


			// to flatten the state-space. 
			frontpositions_eff_FE_vec.reserve(NumberofNodes);
			backpositions_eff_FE_vec.reserve(NumberofNodes);
			frontpositions_eff_FE_vec.push_back(0);
			backpositions_eff_FE_vec.push_back(Leadtime_vec.at(0));
			for (int64_t i = 1; i < NumberofNodes - 1; i++) {
				frontpositions_eff_FE_vec.push_back(backpositions_eff_FE_vec[i - 1] + 1);
				if (EPredecessor_2vec.at(i).size() == 0) {
					backpositions_eff_FE_vec.push_back(frontpositions_eff_FE_vec[i] + Leadtime_vec.at(i));
				}
				else {
					backpositions_eff_FE_vec.push_back(frontpositions_eff_FE_vec[i] + endEchelonLead - Echelonlead_vec.at(ESuccessor_2vec.at(i).at(0)) - 1);
				}
			}
			frontpositions_eff_FE_vec.push_back(backpositions_eff_FE_vec[NumberofNodes - 2] + 1);
			backpositions_eff_FE_vec.push_back(frontpositions_eff_FE_vec[NumberofNodes - 1] + (endEchelonLead - 2));

			auto max_it = std::max_element(CapacityofNodes_vec.begin(), CapacityofNodes_vec.end());
			// max_it is an iterator, to get the value you need to dereference it
			if (max_it != CapacityofNodes_vec.end()) { // Always good to check if the vector isn't empty
				MaxOrderSize = *max_it - minimumorder;
			}
			else {
				throw DynaPlex::Error("asml_assembly: cannot find MaxOrderSize.");
			}

		}


		MDP::Event MDP::GetEvent(const State& state, RNG& rng) const {	
			return demand_dist_vec.at(state.verticalpos).GetSample(rng);
		}

		std::vector<std::tuple<MDP::Event, double>> MDP::EventProbabilities(const State& state) const {
			return demand_dist_vec.at(state.verticalpos).QuantityProbabilities();
		}		
		
		//MDP::Event MDP::GetEvent(RNG& rng) const {	
		//	return demand_dist_vec.at(0).GetSample(rng);
		//}

		//std::vector<std::tuple<MDP::Event, double>> MDP::EventProbabilities() const {
		//	return demand_dist_vec.at(0).QuantityProbabilities();
		//}

		bool MDP::AdaptWarmUp(State& state) const {
			state.warmUp = !state.warmUp;
			return state.warmUp;
		}



		// This function recomputes the maximum system output without the branch of the current node
		vector<int64_t> MDP::maxSystemOutputWithout_eff_singlevec(const State& state) const
		{
			// Here assume that the supply chain configuration is max 3 tier 
			vector<int64_t> maxSystemOutputWithout_suc_vec{};
			vector<int64_t> maxSystemOutputWithout_suc2_vec{};

			int64_t min_max_additional_output = 0;
			// Successor node always has a predecessors, at least the current node
			for (auto successorsnode : ESuccessor_2vec.at(state.currentNode)) {
				// Until one before the lead time of the node that we are ordering, the system output without branch is equal to the entire system output
				for (int64_t l = frontpositions_eff_FE_vec.at(successorsnode); l < frontpositions_eff_FE_vec.at(successorsnode) + Leadtime_vec.at(successorsnode) + Leadtime_vec.at(state.currentNode); l++) {
					maxSystemOutputWithout_suc_vec.push_back(state.cumInvNode_eff_vec.at(l));
				}

				// The successor is in serial with the currentNode, then assume infinite supply and thus add capacity of the successors node to the max system output wihout branch
				if (EPredecessor_2vec.at(successorsnode).size() == 1) {
					//for (int64_t l = Leadtime_vec.at(state.currentNode); l < endEchelonLead - Echelonlead_vec.at(successorsnode); l++) {
					for (int64_t l = Leadtime_vec.at(state.currentNode); l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 1 - Echelonlead_vec.at(successorsnode); l++) {
						maxSystemOutputWithout_suc_vec.push_back(maxSystemOutputWithout_suc_vec.at(maxSystemOutputWithout_suc_vec.size() - 1) + CapacityofNodes_vec.at(successorsnode));
					}
				}
				else {

					// Multiple predecessors of which one is the currentNode
					// From lead time of state.currentNode, for predecessors of successorsnode, it is from lead time of state.currentNode, before that nothing is changed
					for (int64_t l = Leadtime_vec.at(state.currentNode); l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 1 - Echelonlead_vec.at(successorsnode); l++) {
						// Compute the minimum of the maximum system output additional to the successors wip
						// Cannot add more than the capacity each period, i.e., l * Capacity
						min_max_additional_output = CapacityofNodes_vec.at(successorsnode) * (l);
						// It cannot be higher than the sum of the inventories in the predecessors nodes until l
						for (auto node : EPredecessor_2vec.at(successorsnode)) {
							// Want to compute the max system output without the current node
							if (node != state.currentNode) {
								// Either the number of periods is higher than lead time for upstream nodes, then take entire inventory + wip and add the capacity for the left over periods
								if ((EPredecessor_2vec.at(node).size() == 0) && (l > Leadtime_vec.at(node))) {
									if (state.cumInvNode_eff_vec.at(backpositions_eff_FE_vec.at(node)) + CapacityofNodes_vec.at(node) * (l - Leadtime_vec.at(node)) < min_max_additional_output) {
										min_max_additional_output = state.cumInvNode_eff_vec.at(backpositions_eff_FE_vec.at(node)) + CapacityofNodes_vec.at(node) * (l - Leadtime_vec.at(node));
									}
								}
								// Take the inventory
								else {
									if (state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(node) + l) < min_max_additional_output) {
										min_max_additional_output = state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(node) + l);
									}
								}
							}
						}

						maxSystemOutputWithout_suc_vec.push_back(state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(successorsnode) + Leadtime_vec.at(successorsnode)) + min_max_additional_output);

						// Cannot add more than the capacity compared to previous period
						if (maxSystemOutputWithout_suc_vec.at(maxSystemOutputWithout_suc_vec.size() - 2) + CapacityofNodes_vec.at(successorsnode) < maxSystemOutputWithout_suc_vec.at(maxSystemOutputWithout_suc_vec.size() - 1)) {
							maxSystemOutputWithout_suc_vec.at(maxSystemOutputWithout_suc_vec.size() - 1) = maxSystemOutputWithout_suc_vec.at(maxSystemOutputWithout_suc_vec.size() - 2) + CapacityofNodes_vec.at(successorsnode);
						}
					}
				}

				for (auto successorsnode2 : ESuccessor_2vec.at(successorsnode)) {
					// Until the lead time of current node, max system output without branch and max system output are equal
					for (int64_t l = frontpositions_eff_FE_vec.at(successorsnode2); l < frontpositions_eff_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode) + Leadtime_vec.at(state.currentNode) + Leadtime_vec.at(successorsnode2); l++) {
						maxSystemOutputWithout_suc2_vec.push_back(state.cumInvNode_eff_vec.at(l));
					}
					// The successor is in serial with the currentNode, then assume infinite supply and thus add capacity of the successors node to the max system output wihout branch
					if (EPredecessor_2vec.at(successorsnode2).size() == 1) {
						for (int64_t l = Leadtime_vec.at(successorsnode) + Leadtime_vec.at(state.currentNode); l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 1 - Echelonlead_vec.at(successorsnode2); l++) {
							maxSystemOutputWithout_suc2_vec.push_back(maxSystemOutputWithout_suc2_vec.at(l) + CapacityofNodes_vec.at(successorsnode2));
						}
					}
					else {
						// Multiple predecessors of which one is the branch with currentNode
						//for (int64_t l = Leadtime_vec.at(successorsnode) + Leadtime_vec.at(state.currentNode); l < endEchelonLead - Echelonlead_vec.at(successorsnode2); l++) {
						for (int64_t l = Leadtime_vec.at(successorsnode) + Leadtime_vec.at(state.currentNode); l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 1 - Echelonlead_vec.at(successorsnode2); l++) {
							// Compute the minimum of the maximum system output additional to the successors2 wip
							// Cannot add more than the capacity each period, i.e., l * Capacity
							min_max_additional_output = CapacityofNodes_vec.at(successorsnode2) * (l);
							// It cannot be higher than the sum of the inventories in the predecessors nodes until l
							for (auto node : EPredecessor_2vec.at(successorsnode2)) {
								// If the current node is in the branch of successor, take the just computed maxSystemOutputWithout vec
								if (node != successorsnode) {
									if ((EPredecessor_2vec.at(node).size() == 0) && (l > Leadtime_vec.at(node))) {
										if (state.cumInvNode_eff_vec.at(backpositions_eff_FE_vec.at(node)) + CapacityofNodes_vec.at(node) * (l - Leadtime_vec.at(node)) < min_max_additional_output) {
											min_max_additional_output = state.cumInvNode_eff_vec.at(backpositions_eff_FE_vec.at(node)) + CapacityofNodes_vec.at(node) * (l - Leadtime_vec.at(node));
										}
									}
									else {
										if (state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(node) + l) < min_max_additional_output) {
											min_max_additional_output = state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(node) + l);
										}
									}
								}
								// Take the just computed max system output of the successor of the state.currentNode
								else {
									if (maxSystemOutputWithout_suc_vec.at(l) < min_max_additional_output) {
										min_max_additional_output = maxSystemOutputWithout_suc_vec.at(l);
									}
								}
							}

							maxSystemOutputWithout_suc2_vec.push_back(state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode2)) + min_max_additional_output);

							// Cannot add more than the capacity compared to previous period
							if (maxSystemOutputWithout_suc2_vec.at(maxSystemOutputWithout_suc2_vec.size() - 2) + CapacityofNodes_vec.at(successorsnode2) < maxSystemOutputWithout_suc2_vec.at(maxSystemOutputWithout_suc2_vec.size() - 1)) {
								maxSystemOutputWithout_suc2_vec.at(maxSystemOutputWithout_suc2_vec.size() - 1) = maxSystemOutputWithout_suc2_vec.at(maxSystemOutputWithout_suc2_vec.size() - 2) + CapacityofNodes_vec.at(successorsnode2);
							}
						}
					}
				}
			}

			if (maxSystemOutputWithout_suc2_vec.size() > 0) {
				return maxSystemOutputWithout_suc2_vec;
			}
			else if (maxSystemOutputWithout_suc_vec.size() > 0) {
				return maxSystemOutputWithout_suc_vec;
			}
			else {
				//if (zerosMaxSystemOutput) {
					vector<int64_t> zerosOutput(endEchelonLead, 0);
					return zerosOutput;
				//}
				/*else {
					for (int64_t l = frontpositions_eff_FE_vec.at(state.currentNode); l < frontpositions_eff_FE_vec.at(state.currentNode) + Leadtime_vec.at(state.currentNode); l++) {
						maxSystemOutputWithout_suc_vec.push_back(state.cumInvNode_eff_vec.at(l));
					}
					for (int64_t l = Echelonlead_vec.at(state.currentNode) - 2; l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 2; l++) {
						maxSystemOutputWithout_suc_vec.push_back(maxSystemOutputWithout_suc_vec.at(maxSystemOutputWithout_suc_vec.size() - 1) + CapacityofNodes_vec.at(state.currentNode));
					}
				}*/
			}

			return maxSystemOutputWithout_suc_vec;
		}
		

				
		// Recompute the features of the successors nodes
		double MDP::recomputeFeatures_eff_singlevec(State& state, const bool CurNode) const
		{
			int64_t min_max_additional_output;
			int64_t index;


			// Recompute current node cumInvNode including the order, as the node has already ordered
			if (CurNode) {
				if (EPredecessor_2vec.at(state.currentNode).size() == 0) {
					std::cout << " Error recompute cumInvNode only for nodes without predecessors " << std::endl;
				}
				else {
					// Until the current order, at lead time, nothing changed
					// This is only called for nodes that do have a predecessor, i.e., the cumInvNode_eff_vec is till endEchelonLead periods
					for (int64_t i = 1; i < endEchelonLead - Echelonlead_vec.at(state.currentNode); i++) {
						// Compute the additional maximum output from the current production decision onwards
						// Cannot add more than capacity per period
						index = frontpositions_eff_FE_vec.at(state.currentNode) + Leadtime_vec.at(state.currentNode) + i;
						min_max_additional_output = CapacityofNodes_vec.at(state.currentNode) * (i);
						for (auto node : EPredecessor_2vec.at(state.currentNode)) {
							if ((EPredecessor_2vec.at(node).size() == 0) && (i > Leadtime_vec.at(node))) {
								if (state.cumInvNode_eff_vec.at(backpositions_eff_FE_vec.at(node)) + CapacityofNodes_vec.at(node) * (i - Leadtime_vec.at(node)) < min_max_additional_output) {
									min_max_additional_output = state.cumInvNode_eff_vec.at(backpositions_eff_FE_vec.at(node)) + CapacityofNodes_vec.at(node) * (i - Leadtime_vec.at(node));
								}
							}
							else {
								if (state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(node) + i) < min_max_additional_output) {
									min_max_additional_output = state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(node) + i);
								}
							}
						}
						// Cannot add more than capacity compared to previous period
						if (state.cumInvNode_eff_vec.at(index - 1) + CapacityofNodes_vec.at(state.currentNode) - state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(state.currentNode) + Leadtime_vec.at(state.currentNode)) < min_max_additional_output) {
							min_max_additional_output = state.cumInvNode_eff_vec.at(index - 1) + CapacityofNodes_vec.at(state.currentNode) - state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(state.currentNode) + Leadtime_vec.at(state.currentNode));
						}
						state.cumInvNode_eff_vec.at(index) = state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(state.currentNode) + Leadtime_vec.at(state.currentNode)) + min_max_additional_output;
					}
				}
			}


			// Backward induction to compute max output from nodes with predecessors
			// Maximum system output at all future periods for the entire system
			// NOTE successors have already ordered, but you want to keep equal until the current production as from there on it may change
			// Note successors is always a node that has a predecessor so always vector of size leadtime_size
			for (int64_t n : ESuccessor_2vec.at(state.currentNode)) {				// Greater than zero or than one
				//All successors have already ordered in this time period

				// Note that in the cumInvNode_2vec already all upstream inventories + capacities of node n are taken into account

				// Until the current order, at lead time of n which is Leadtime_vec.at(n) + Leadtime_vec.at(state.currentNode) element, nothing changed. Not including, as this has changed due to the order

				// Multiple predecessors of which one is the currentNode
				// From lead time of state.currentNode, for predecessors of successorsnode, it is from lead time of state.currentNode, before that nothing is changed
				for (int64_t l = Leadtime_vec.at(state.currentNode); l < endEchelonLead - Echelonlead_vec.at(n); l++) {
					min_max_additional_output = CapacityofNodes_vec.at(n) * (l);
					// It cannot be higher than the sum of the inventories in the predecessors nodes until l
					for (auto node : EPredecessor_2vec.at(n)) {
						if ((EPredecessor_2vec.at(node).size() == 0) && (l > Leadtime_vec.at(node))) {
							if (state.cumInvNode_eff_vec.at(backpositions_eff_FE_vec.at(node)) + CapacityofNodes_vec.at(node) * (l - Leadtime_vec.at(node)) < min_max_additional_output) {
								min_max_additional_output = state.cumInvNode_eff_vec.at(backpositions_eff_FE_vec.at(node)) + CapacityofNodes_vec.at(node) * (l - Leadtime_vec.at(node));
							}
						}
						else {
							if (state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(node) + l) < min_max_additional_output) {
								min_max_additional_output = state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(node) + l);
							}
						}
					}

					// Cannot add more than capacity
					if (state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(n) + Leadtime_vec.at(n) + l - 1) + CapacityofNodes_vec.at(n) - state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(n) + Leadtime_vec.at(n)) < min_max_additional_output) {
						min_max_additional_output = state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(n) + Leadtime_vec.at(n) + l - 1) + CapacityofNodes_vec.at(n) - state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(n) + Leadtime_vec.at(n));
					}
					state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(n) + Leadtime_vec.at(n) + l) = state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(n) + Leadtime_vec.at(n)) + min_max_additional_output;
				}

				for (auto successorsnode2 : ESuccessor_2vec.at(n)) {
					// Multiple predecessors of which one is the branch with currentNode

					// From lead time of state.currentNode + lead time of node n, for predecessors of successorsnode, it is from that lead time as before that nothing is changed
					for (int64_t l = Leadtime_vec.at(n) + Leadtime_vec.at(state.currentNode); l < endEchelonLead - Echelonlead_vec.at(successorsnode2); l++) {
						// Multiple predecessors of which maximally the capacity can be ordered per time period
						min_max_additional_output = CapacityofNodes_vec.at(successorsnode2) * (l);
						// It cannot be higher than the sum of the inventories in the predecessors nodes until l
						for (auto node : EPredecessor_2vec.at(successorsnode2)) {
							if ((EPredecessor_2vec.at(node).size() == 0) && (l > Leadtime_vec.at(node))) {
								if (state.cumInvNode_eff_vec.at(backpositions_eff_FE_vec.at(node)) + CapacityofNodes_vec.at(node) * (l - Leadtime_vec.at(node)) < min_max_additional_output) {
									min_max_additional_output = state.cumInvNode_eff_vec.at(backpositions_eff_FE_vec.at(node)) + CapacityofNodes_vec.at(node) * (l - Leadtime_vec.at(node));
								}
							}
							else {
								if (state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(node) + l) < min_max_additional_output) {
									min_max_additional_output = state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(node) + l);
								}
							}
						}
						// And cannot add more than capacity compared to the previous l
						if (state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode2) + l - 1) + CapacityofNodes_vec.at(successorsnode2) - state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode2)) < min_max_additional_output) {
							min_max_additional_output = state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode2) + l - 1) + CapacityofNodes_vec.at(successorsnode2) - state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode2));
						}
						state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode2) + l) = state.cumInvNode_eff_vec.at(frontpositions_eff_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode2)) + min_max_additional_output;

					}

				}
			}
			return 0.0;
		}


		void MDP::RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>& registry) const
		{
			registry.Register<Cur_D_BaseStockPolicy>("cur_d_base_stock",
				"Cur-D Base-stock policy");
			registry.Register<DT_BaseStockPolicy>("dt_base_stock",
				"DT2 Base-stock policy");
		}

		// One dimensional vector. All state information, does not need to be equal to features
		// To load the state from the samples
		MDP::State MDP::GetState(const DynaPlex::VarGroup& vars) const
		{
			State state{};
			vars.Get("cat", state.cat);
			vars.Get("StateVectors", state.StateVectors);
			vars.Get("verticalpos", state.verticalpos);
			vars.Get("currentNode", state.currentNode);
			vars.Get("echelonInv", state.echelonInv_vec);
			vars.Get("cumInvNode_eff", state.cumInvNode_eff_vec);
			//vars.Get("features_vec", state.features_vec);
			vars.Get("warmUp", state.warmUp);
			return state;
		}
		// To save the state as a sample
		DynaPlex::VarGroup MDP::State::ToVarGroup() const
		{
			DynaPlex::VarGroup vars;
			vars.Add("cat", cat);
			vars.Add("StateVectors", StateVectors);
			vars.Add("verticalpos", verticalpos);
			vars.Add("currentNode", currentNode);
			vars.Add("echelonInv", echelonInv_vec);
			vars.Add("cumInvNode_eff", cumInvNode_eff_vec);
			//vars.Add("features_vec", features_vec);
			vars.Add("warmUp", warmUp);
			return vars;
		}

		void Register(DynaPlex::Registry& registry)
		{
			DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel(
				/*=id though which the MDP will be retrievable*/ "asml_assembly_nonstationary",
				/*description*/ "Production-inventory planning in a high-tech assembly system.",
				/*reference to passed registry*/registry);
		}

		DynaPlex::StateCategory MDP::GetStateCategory(const State& state) const
		{
			return state.cat;
		}

		// This function recomputes the maximum system output without the branch of the current node
		//vector<int64_t> MDP::maxSystemOutputWithout_singlevec(const State& state) const
		//{
		//	// Here assume that the supply chain configuration is max 3 tier 
		//	vector<int64_t> maxSystemOutputWithout_suc_vec{};
		//	vector<int64_t> maxSystemOutputWithout_suc2_vec{};

		//	int64_t min_max_additional_output = 0;

		//	for (auto successorsnode : ESuccessor_2vec.at(state.currentNode)) {
		//		for (int64_t l = frontpositions_FE_vec.at(successorsnode); l < frontpositions_FE_vec.at(successorsnode) + Leadtime_vec.at(successorsnode) + Leadtime_vec.at(state.currentNode); l++) {
		//			maxSystemOutputWithout_suc_vec.push_back(state.cumInvNode_vec.at(l));
		//		}

		//		// The successor is in serial with the currentNode, thus removing branch of currentNode
		//		// then we only have the inventory that is already in pipeline for the currentNode
		//		if (EPredecessor_2vec.at(successorsnode).size() == 1) {
		//			//for (int64_t l = Leadtime_vec.at(state.currentNode); l < endEchelonLead - Echelonlead_vec.at(successorsnode); l++) {
		//			for (int64_t l = Leadtime_vec.at(state.currentNode); l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 1 - Echelonlead_vec.at(successorsnode); l++) {
		//				maxSystemOutputWithout_suc_vec.push_back(maxSystemOutputWithout_suc_vec.at(maxSystemOutputWithout_suc_vec.size() - 1) + CapacityofNodes_vec.at(successorsnode));
		//			}
		//		}
		//		else {

		//			// Multiple predecessors of which one is the currentNode
		//			// From lead time of state.currentNode, for predecessors of successorsnode, it is from lead time of state.currentNode, before that nothing is changed
		//			for (int64_t l = Leadtime_vec.at(state.currentNode); l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 1 - Echelonlead_vec.at(successorsnode); l++) {
		//				min_max_additional_output = CapacityofNodes_vec.at(successorsnode) * (l);
		//				// It cannot be higher than the sum of the inventories in the predecessors nodes until l
		//				for (auto node : EPredecessor_2vec.at(successorsnode)) {
		//					if (node != state.currentNode) {
		//						if (state.cumInvNode_vec.at(frontpositions_FE_vec.at(node) + l) < min_max_additional_output) {
		//							min_max_additional_output = state.cumInvNode_vec.at(frontpositions_FE_vec.at(node) + l);
		//						}
		//					}
		//				}

		//				// Cannot add more than capacity
		//				//if (state.cumInvNode_vec.at(frontpositions_FE_vec.at(successorsnode) + Leadtime_vec.at(successorsnode) + l - 1) + CapacityofNodes_vec.at(successorsnode) - state.cumInvNode_vec.at(frontpositions_FE_vec.at(successorsnode) + Leadtime_vec.at(successorsnode)) < min_max_additional_output) {
		//				//	min_max_additional_output = state.cumInvNode_vec.at(frontpositions_FE_vec.at(successorsnode) + Leadtime_vec.at(successorsnode) + l - 1) + CapacityofNodes_vec.at(successorsnode) - state.cumInvNode_vec.at(frontpositions_FE_vec.at(successorsnode) + Leadtime_vec.at(successorsnode));
		//				//}
		//				maxSystemOutputWithout_suc_vec.push_back(state.cumInvNode_vec.at(frontpositions_FE_vec.at(successorsnode) + Leadtime_vec.at(successorsnode)) + min_max_additional_output);

		//				// Check the capacity
		//				if (maxSystemOutputWithout_suc_vec.at(maxSystemOutputWithout_suc_vec.size() - 2) + CapacityofNodes_vec.at(successorsnode) < maxSystemOutputWithout_suc_vec.at(maxSystemOutputWithout_suc_vec.size() - 1)) {
		//					maxSystemOutputWithout_suc_vec.at(maxSystemOutputWithout_suc_vec.size() - 1) = maxSystemOutputWithout_suc_vec.at(maxSystemOutputWithout_suc_vec.size() - 2) + CapacityofNodes_vec.at(successorsnode);
		//				}
		//			}
		//		}

		//		for (auto successorsnode2 : ESuccessor_2vec.at(successorsnode)) {

		//			for (int64_t l = frontpositions_FE_vec.at(successorsnode2); l < frontpositions_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode) + Leadtime_vec.at(state.currentNode) + Leadtime_vec.at(successorsnode2); l++) {
		//				maxSystemOutputWithout_suc2_vec.push_back(state.cumInvNode_vec.at(l));
		//			}
		//			if (EPredecessor_2vec.at(successorsnode2).size() == 1) {
		//				for (int64_t l = Leadtime_vec.at(successorsnode) + Leadtime_vec.at(state.currentNode); l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 1 - Echelonlead_vec.at(successorsnode2); l++) {
		//					maxSystemOutputWithout_suc2_vec.push_back(maxSystemOutputWithout_suc2_vec.at(l) + CapacityofNodes_vec.at(successorsnode2));
		//				}
		//			}
		//			else {
		//				// Multiple predecessors of which one is the branch with currentNode
		//				//for (int64_t l = Leadtime_vec.at(successorsnode) + Leadtime_vec.at(state.currentNode); l < endEchelonLead - Echelonlead_vec.at(successorsnode2); l++) {
		//				for (int64_t l = Leadtime_vec.at(successorsnode) + Leadtime_vec.at(state.currentNode); l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 1 - Echelonlead_vec.at(successorsnode2); l++) {
		//					// Multiple predecessors of which maximally the capacity can be ordered per time period
		//					min_max_additional_output = CapacityofNodes_vec.at(successorsnode2) * (l);
		//					for (auto node : EPredecessor_2vec.at(successorsnode2)) {
		//						// Take the echelon inv vec of the other predecessors nodes
		//						if (node != successorsnode) {
		//							if (state.cumInvNode_vec.at(frontpositions_FE_vec.at(node) + l) < min_max_additional_output) {
		//								min_max_additional_output = state.cumInvNode_vec.at(frontpositions_FE_vec.at(node) + l);
		//							}
		//						}
		//						// Take the just computed max system output of the successor of the state.currentNode
		//						else {
		//							if (maxSystemOutputWithout_suc_vec.at(l) < min_max_additional_output) {
		//								min_max_additional_output = maxSystemOutputWithout_suc_vec.at(l);
		//							}
		//						}
		//					}
		//					// Cannot add more than capacity
		//					//if (state.cumInvNode_vec.at(frontpositions_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode2) + l - 1) + CapacityofNodes_vec.at(successorsnode2) - state.cumInvNode_vec.at(frontpositions_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode2)) < min_max_additional_output) {
		//					//	min_max_additional_output = state.cumInvNode_vec.at(frontpositions_FE_vec.at(successorsnode) + Leadtime_vec.at(successorsnode2) + l - 1) + CapacityofNodes_vec.at(successorsnode2) - state.cumInvNode_vec.at(frontpositions_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode2));
		//					//}
		//					maxSystemOutputWithout_suc2_vec.push_back(state.cumInvNode_vec.at(frontpositions_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode2)) + min_max_additional_output);

		//					// Check the capacity
		//					if (maxSystemOutputWithout_suc2_vec.at(maxSystemOutputWithout_suc2_vec.size() - 2) + CapacityofNodes_vec.at(successorsnode2) < maxSystemOutputWithout_suc2_vec.at(maxSystemOutputWithout_suc2_vec.size() - 1)) {
		//						maxSystemOutputWithout_suc2_vec.at(maxSystemOutputWithout_suc2_vec.size() - 1) = maxSystemOutputWithout_suc2_vec.at(maxSystemOutputWithout_suc2_vec.size() - 2) + CapacityofNodes_vec.at(successorsnode2);
		//					}
		//				}
		//			}
		//		}
		//	}

		//	if (maxSystemOutputWithout_suc2_vec.size() > 0) {
		//		return maxSystemOutputWithout_suc2_vec;
		//	}
		//	else if (maxSystemOutputWithout_suc_vec.size() > 0) {
		//		return maxSystemOutputWithout_suc_vec;
		//	}
		//	else {
		//		vector<int64_t> zerosOutput(leadtime_size, 0);
		//		return zerosOutput;
		//	}

		//	return maxSystemOutputWithout_suc_vec;
		// }	




		// Recompute the features of the successors nodes
		//double MDP::recomputeFeatures_singlevec(State& state, const bool CurNode) const
		//{
		//	int64_t min_max_additional_output;


		//	// Recompute current node cumInvNode including the order, as the node has already ordered
		//	if (CurNode) {
		//		// Until the current order, at lead time, nothing changed
		//		// Nothing changed, so why would we want to do this?
		//		// From 1 of cumInvNode vec of predecessors as currentNode has just ordered
		//		for (int64_t i = 1; i < endEchelonLead - Echelonlead_vec.at(state.currentNode); i++) {
		//			min_max_additional_output = CapacityofNodes_vec.at(state.currentNode) * (i);
		//			for (auto node : EPredecessor_2vec.at(state.currentNode)) {
		//				if (state.cumInvNode_vec.at(frontpositions_FE_vec.at(node) + i) < min_max_additional_output) {
		//					min_max_additional_output = state.cumInvNode_vec.at(frontpositions_FE_vec.at(node) + i);
		//				}
		//			}
		//			// Cannot add more than capacity
		//			if (state.cumInvNode_vec.at(frontpositions_FE_vec.at(state.currentNode) + Leadtime_vec.at(state.currentNode) + i - 1) + CapacityofNodes_vec.at(state.currentNode) - state.cumInvNode_vec.at(frontpositions_FE_vec.at(state.currentNode) + Leadtime_vec.at(state.currentNode)) < min_max_additional_output) {
		//				min_max_additional_output = state.cumInvNode_vec.at(frontpositions_FE_vec.at(state.currentNode) + Leadtime_vec.at(state.currentNode) + i - 1) + CapacityofNodes_vec.at(state.currentNode) - state.cumInvNode_vec.at(frontpositions_FE_vec.at(state.currentNode) + Leadtime_vec.at(state.currentNode));
		//			}
		//			state.cumInvNode_vec.at(frontpositions_FE_vec.at(state.currentNode) + Leadtime_vec.at(state.currentNode) + i) = state.cumInvNode_vec.at(frontpositions_FE_vec.at(state.currentNode) + Leadtime_vec.at(state.currentNode)) + min_max_additional_output;
		//		}
		//	}


		//	// Backward induction to compute max output from nodes with predecessors
		//	// Maximum system output at all future periods for the entire system
		//	// NOTE successors have already ordered, but you want to keep equal until the current production as from there on it may change
		//	for (int64_t n : ESuccessor_2vec.at(state.currentNode)) {				// Greater than zero or than one
		//		//All successors have already ordered in this time period

		//		// Note that in the cumInvNode_2vec already all upstream inventories + capacities of node n are taken into account

		//		// Until the current order, at lead time of n which is Leadtime_vec.at(n) + Leadtime_vec.at(state.currentNode) element, nothing changed. Not including, as this has changed due to the order

		//		// Multiple predecessors of which one is the currentNode
		//		// From lead time of state.currentNode, for predecessors of successorsnode, it is from lead time of state.currentNode, before that nothing is changed
		//		for (int64_t l = Leadtime_vec.at(state.currentNode); l < endEchelonLead - Echelonlead_vec.at(n); l++) {
		//			min_max_additional_output = CapacityofNodes_vec.at(n) * (l);
		//			// It cannot be higher than the sum of the inventories in the predecessors nodes until l
		//			for (auto node : EPredecessor_2vec.at(n)) {
		//				if (state.cumInvNode_vec.at(frontpositions_FE_vec.at(node) + l) < min_max_additional_output) {
		//					min_max_additional_output = state.cumInvNode_vec.at(frontpositions_FE_vec.at(node) + l);
		//				}
		//			}

		//			// Cannot add more than capacity
		//			if (state.cumInvNode_vec.at(frontpositions_FE_vec.at(n) + Leadtime_vec.at(n) + l - 1) + CapacityofNodes_vec.at(n) - state.cumInvNode_vec.at(frontpositions_FE_vec.at(n) + Leadtime_vec.at(n)) < min_max_additional_output) {
		//				min_max_additional_output = state.cumInvNode_vec.at(frontpositions_FE_vec.at(n) + Leadtime_vec.at(n) + l - 1) + CapacityofNodes_vec.at(n) - state.cumInvNode_vec.at(frontpositions_FE_vec.at(n) + Leadtime_vec.at(n));
		//			}
		//			state.cumInvNode_vec.at(frontpositions_FE_vec.at(n) + Leadtime_vec.at(n) + l) = state.cumInvNode_vec.at(frontpositions_FE_vec.at(n) + Leadtime_vec.at(n)) + min_max_additional_output;
		//		}

		//		for (auto successorsnode2 : ESuccessor_2vec.at(n)) {
		//			// Multiple predecessors of which one is the branch with currentNode

		//			// From lead time of state.currentNode + lead time of node n, for predecessors of successorsnode, it is from that lead time as before that nothing is changed
		//			for (int64_t l = Leadtime_vec.at(n) + Leadtime_vec.at(state.currentNode); l < endEchelonLead - Echelonlead_vec.at(successorsnode2); l++) {
		//				// Multiple predecessors of which maximally the capacity can be ordered per time period
		//				min_max_additional_output = CapacityofNodes_vec.at(successorsnode2) * (l);
		//				// It cannot be higher than the sum of the inventories in the predecessors nodes until l
		//				for (auto node : EPredecessor_2vec.at(successorsnode2)) {
		//					if (state.cumInvNode_vec.at(frontpositions_FE_vec.at(node) + l) < min_max_additional_output) {
		//						min_max_additional_output = state.cumInvNode_vec.at(frontpositions_FE_vec.at(node) + l);
		//					}
		//				}
		//				// And cannot add more than capacity compared to the previous l
		//				if (state.cumInvNode_vec.at(frontpositions_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode2) + l - 1) + CapacityofNodes_vec.at(successorsnode2) - state.cumInvNode_vec.at(frontpositions_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode2)) < min_max_additional_output) {
		//					min_max_additional_output = state.cumInvNode_vec.at(frontpositions_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode2) + l - 1) + CapacityofNodes_vec.at(successorsnode2) - state.cumInvNode_vec.at(frontpositions_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode2));
		//				}
		//				state.cumInvNode_vec.at(frontpositions_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode2) + l) = state.cumInvNode_vec.at(frontpositions_FE_vec.at(successorsnode2) + Leadtime_vec.at(successorsnode2)) + min_max_additional_output;

		//			}

		//		}
		//	}
		//	return 0.0;
		//}



		// SAVED the function below as it attains the code if we want to completely discard the entire branch without
		// considering the in-transit inventories at that node
		// This function recomputes the maximum system output without the branch of the current node
		//vector<int64_t> MDP::maxSystemOutputWithout(const State& state) const
		//{
		//	// Here assume that the supply chain configuration is max 3 tier 
		//	vector<int64_t> maxSystemOutputWithout_suc_vec{};
		//	vector<int64_t> maxSystemOutputWithout_suc2_vec{};

		//	int64_t min_max_additional_output;

			// Now compute from lead time = 0, to endLeadTime, but based on the current node we know how far to compute this. Could save some time
			// We could also check whether the function is difference from state.cumInvNode.at(successorsnode), if not then no need to compute it further
			//for (auto successorsnode : ESuccessor_2vec.at(state.currentNode)) {
			//	//All successors have already ordered in this time period
			//	// For the predecessor computing the maximum system output of the state.currentNode then until that nothing changes
			//	
			//	// This is wrong I think, without that branch is not equal to keeping everything similar until lead time of current node
			//	// Since after lead time of successor this system output is changed by not considering the current branch

			//	for (int64_t l = 0; l <= Leadtime_vec.at(successorsnode); l++) {
			//		maxSystemOutputWithout_suc_vec.push_back(state.cumInvNode_2vec.at(successorsnode).at(l));
			//	}
			//	// The successor is in serial with the currentNode, thus removing branch of currentNode
			//	// then we only have the inventory that is already in pipeline for the currentNode
			//	if (EPredecessor_2vec.at(successorsnode).size() == 1) {
			//		//for (int64_t l = Leadtime_vec.at(state.currentNode); l < endEchelonLead - Echelonlead_vec.at(successorsnode); l++) {
			//		for (int64_t l = 1; l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 1 - Echelonlead_vec.at(successorsnode); l++) {
			//			maxSystemOutputWithout_suc_vec.push_back(maxSystemOutputWithout_suc_vec.at(maxSystemOutputWithout_suc_vec.size() - 1) + CapacityofNodes_vec.at(successorsnode));
			//		}
			//	}
			//	else {
			//		// Multiple predecessors of which one is the currentNode
			//		// From lead time of state.currentNode, for predecessors of successorsnode, it is from lead time of state.currentNode, before that nothing is changed
			//		//for (int64_t l = Leadtime_vec.at(state.currentNode); l < endEchelonLead - Echelonlead_vec.at(successorsnode); l++) {
			//		
			//		// For all predecessors of the successors node, take the second element (not on-hand element as the successor of state.currentNode has already ordered)
			//		for (int64_t l = 1; l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 1 - Echelonlead_vec.at(successorsnode); l++) {
			//			min_max_additional_output = CapacityofNodes_vec.at(successorsnode) * (l);
			//			for (auto node : EPredecessor_2vec.at(successorsnode)) {
			//				// Skip branch of currentNode
			//				if (node != state.currentNode) {
			//					if (state.cumInvNode_2vec.at(node).at(l) < min_max_additional_output) {
			//						min_max_additional_output = state.cumInvNode_2vec.at(node).at(l);
			//					}
			//				}
			//			}
			//			// Cannot add more than capacity
			//			if (maxSystemOutputWithout_suc_vec.at(maxSystemOutputWithout_suc_vec.size() - 1) + CapacityofNodes_vec.at(successorsnode) - state.cumInvNode_2vec.at(successorsnode).at(Leadtime_vec.at(successorsnode)) < min_max_additional_output) {
			//				min_max_additional_output = maxSystemOutputWithout_suc_vec.at(maxSystemOutputWithout_suc_vec.size() - 1) + CapacityofNodes_vec.at(successorsnode) - state.cumInvNode_2vec.at(successorsnode).at(Leadtime_vec.at(successorsnode));
			//			}
			//			maxSystemOutputWithout_suc_vec.push_back(state.cumInvNode_2vec.at(successorsnode).at(Leadtime_vec.at(successorsnode)) + min_max_additional_output);
			//		}
			//	}


			//	for (auto successorsnode2 : ESuccessor_2vec.at(successorsnode)) {
			//		//All successors have already ordered in this time period

			//		// Only for after the lead time of the successorsnode somethings changes, thus <= Leadtime_vec.at(successorsnode) + Leadtime_vec.at(successorsnode2)
			//		for (int64_t l = 0; l <= Leadtime_vec.at(successorsnode) + Leadtime_vec.at(successorsnode2); l++) {
			//			maxSystemOutputWithout_suc2_vec.push_back(state.cumInvNode_2vec.at(successorsnode2).at(l));
			//		}

			//		if (EPredecessor_2vec.at(successorsnode2).size() == 1) {
			//			for (int64_t l = Leadtime_vec.at(successorsnode) + Leadtime_vec.at(state.currentNode); l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 1 - Echelonlead_vec.at(successorsnode2); l++) {
			//				maxSystemOutputWithout_suc2_vec.push_back(maxSystemOutputWithout_suc2_vec.at(l) + CapacityofNodes_vec.at(successorsnode2));
			//			}
			//		}
			//		else {
			//			// Multiple predecessors of which one is the branch with currentNode
			//			//for (int64_t l = Leadtime_vec.at(successorsnode) + Leadtime_vec.at(state.currentNode); l < endEchelonLead - Echelonlead_vec.at(successorsnode2); l++) {
			//			for (int64_t l = Leadtime_vec.at(successorsnode) + 1; l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 1 - Echelonlead_vec.at(successorsnode2); l++) {
			//				// Multiple predecessors of which maximally the capacity can be ordered per time period
			//				min_max_additional_output = CapacityofNodes_vec.at(successorsnode2) * (l);
			//				for (auto node : EPredecessor_2vec.at(successorsnode2)) {
			//					// Take the echelon inv vec of the other predecessors nodes
			//					if (node != successorsnode) {
			//						if (state.cumInvNode_2vec.at(node).at(l) < min_max_additional_output) {
			//							min_max_additional_output = state.cumInvNode_2vec.at(node).at(l);
			//						}
			//					}
			//					// Take the just computed max system output of the successor of the state.currentNode
			//					else {
			//						if (maxSystemOutputWithout_suc_vec.at(l) < min_max_additional_output) {
			//							min_max_additional_output = maxSystemOutputWithout_suc_vec.at(l);
			//						}
			//					}
			//				}
			//				// Cannot add more than capacity
			//				if (maxSystemOutputWithout_suc2_vec.at(maxSystemOutputWithout_suc2_vec.size()-1) + CapacityofNodes_vec.at(successorsnode2) - state.cumInvNode_2vec.at(successorsnode2).at(Leadtime_vec.at(successorsnode2)) < min_max_additional_output) {
			//					min_max_additional_output = maxSystemOutputWithout_suc2_vec.at(maxSystemOutputWithout_suc2_vec.size() - 1) + CapacityofNodes_vec.at(successorsnode2) - state.cumInvNode_2vec.at(successorsnode2).at(Leadtime_vec.at(successorsnode2));
			//				}

			//				maxSystemOutputWithout_suc2_vec.push_back(state.cumInvNode_2vec.at(successorsnode2).at(Leadtime_vec.at(successorsnode2)) + min_max_additional_output);
			//			}
			//		}


			// CODE BELOW is when we consider the maximum system output without that branch as of current time period
			// Thus, what is already ordered and in pipeline for that node is taken into account still. In code above not
			// TODO: question what is the right thing to do?
		//	for (auto successorsnode : ESuccessor_2vec.at(state.currentNode)) {

		//		for (int64_t l = 0; l < Leadtime_vec.at(successorsnode) + Leadtime_vec.at(state.currentNode); l++) {
		//			maxSystemOutputWithout_suc_vec.push_back(state.cumInvNode_2vec.at(successorsnode).at(l));
		//		}
		//		// The successor is in serial with the currentNode, thus removing branch of currentNode
		//		// then we only have the inventory that is already in pipeline for the currentNode
		//		if (EPredecessor_2vec.at(successorsnode).size() == 1) {
		//			//for (int64_t l = Leadtime_vec.at(state.currentNode); l < endEchelonLead - Echelonlead_vec.at(successorsnode); l++) {
		//			for (int64_t l = Leadtime_vec.at(state.currentNode); l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 1 - Echelonlead_vec.at(successorsnode); l++) {
		//				maxSystemOutputWithout_suc_vec.push_back(maxSystemOutputWithout_suc_vec.at(maxSystemOutputWithout_suc_vec.size() - 1) + CapacityofNodes_vec.at(successorsnode));
		//			}
		//		}
		//		else {
		//			// Multiple predecessors of which one is the currentNode
		//			// From lead time of state.currentNode, for predecessors of successorsnode, it is from lead time of state.currentNode, before that nothing is changed
		//			//for (int64_t l = Leadtime_vec.at(state.currentNode); l < endEchelonLead - Echelonlead_vec.at(successorsnode); l++) {
		//			for (int64_t l = Leadtime_vec.at(state.currentNode); l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 1 - Echelonlead_vec.at(successorsnode); l++) {
		//				min_max_additional_output = CapacityofNodes_vec.at(successorsnode) * (l);
		//				for (auto node : EPredecessor_2vec.at(successorsnode)) {
		//					// Skip branch of currentNode
		//					if (node != state.currentNode) {
		//						if (state.cumInvNode_2vec.at(node).at(l) < min_max_additional_output) {
		//							min_max_additional_output = state.cumInvNode_2vec.at(node).at(l);
		//						}
		//					}
		//				}
		//				// Cannot add more than capacity
		//				if (maxSystemOutputWithout_suc_vec.at(maxSystemOutputWithout_suc_vec.size() - 1) + CapacityofNodes_vec.at(successorsnode) - state.cumInvNode_2vec.at(successorsnode).at(Leadtime_vec.at(successorsnode)) < min_max_additional_output) {
		//					min_max_additional_output = maxSystemOutputWithout_suc_vec.at(maxSystemOutputWithout_suc_vec.size() - 1) + CapacityofNodes_vec.at(successorsnode) - state.cumInvNode_2vec.at(successorsnode).at(Leadtime_vec.at(successorsnode));
		//				}
		//				maxSystemOutputWithout_suc_vec.push_back(state.cumInvNode_2vec.at(successorsnode).at(Leadtime_vec.at(successorsnode)) + min_max_additional_output);
		//			}
		//		}

		//		for (auto successorsnode2 : ESuccessor_2vec.at(successorsnode)) {

		//			for (int64_t l = 0; l < Leadtime_vec.at(successorsnode) + Leadtime_vec.at(state.currentNode) + Leadtime_vec.at(successorsnode2); l++) {
		//				maxSystemOutputWithout_suc2_vec.push_back(state.cumInvNode_2vec.at(successorsnode2).at(l));
		//			}

		//			if (EPredecessor_2vec.at(successorsnode2).size() == 1) {
		//				for (int64_t l = Leadtime_vec.at(successorsnode) + Leadtime_vec.at(state.currentNode); l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 1 - Echelonlead_vec.at(successorsnode2); l++) {
		//					maxSystemOutputWithout_suc2_vec.push_back(maxSystemOutputWithout_suc2_vec.at(l) + CapacityofNodes_vec.at(successorsnode2));
		//				}
		//			}
		//			else {
		//				// Multiple predecessors of which one is the branch with currentNode
		//				//for (int64_t l = Leadtime_vec.at(successorsnode) + Leadtime_vec.at(state.currentNode); l < endEchelonLead - Echelonlead_vec.at(successorsnode2); l++) {
		//				for (int64_t l = Leadtime_vec.at(successorsnode) + Leadtime_vec.at(state.currentNode); l < leadtime_size + Echelonlead_vec.at(state.currentNode) - 1 - Echelonlead_vec.at(successorsnode2); l++) {
		//					// Multiple predecessors of which maximally the capacity can be ordered per time period
		//					min_max_additional_output = CapacityofNodes_vec.at(successorsnode2) * (l);
		//					for (auto node : EPredecessor_2vec.at(successorsnode2)) {
		//						// Take the echelon inv vec of the other predecessors nodes
		//						if (node != successorsnode) {
		//							if (state.cumInvNode_2vec.at(node).at(l) < min_max_additional_output) {
		//								min_max_additional_output = state.cumInvNode_2vec.at(node).at(l);
		//							}
		//						}
		//						// Take the just computed max system output of the successor of the state.currentNode
		//						else {
		//							if (maxSystemOutputWithout_suc_vec.at(l) < min_max_additional_output) {
		//								min_max_additional_output = maxSystemOutputWithout_suc_vec.at(l);
		//							}
		//						}
		//					}
		//					// Cannot add more than capacity
		//					if (maxSystemOutputWithout_suc2_vec.at(maxSystemOutputWithout_suc2_vec.size() - 1) + CapacityofNodes_vec.at(successorsnode2) - state.cumInvNode_2vec.at(successorsnode2).at(Leadtime_vec.at(successorsnode2)) < min_max_additional_output) {
		//						min_max_additional_output = maxSystemOutputWithout_suc2_vec.at(maxSystemOutputWithout_suc2_vec.size() - 1) + CapacityofNodes_vec.at(successorsnode2) - state.cumInvNode_2vec.at(successorsnode2).at(Leadtime_vec.at(successorsnode2));
		//					}

		//					maxSystemOutputWithout_suc2_vec.push_back(state.cumInvNode_2vec.at(successorsnode2).at(Leadtime_vec.at(successorsnode2)) + min_max_additional_output);
		//				}
		//			}
		//		}
		//	}

		//	if (maxSystemOutputWithout_suc2_vec.size() > 0) {
		//		return maxSystemOutputWithout_suc2_vec;
		//	}
		//	else if (maxSystemOutputWithout_suc_vec.size() > 0) {
		//		return maxSystemOutputWithout_suc_vec;
		//	}
		//	else {
		//		vector<int64_t> zerosOutput(leadtime_size, 0);
		//		return zerosOutput;
		//	}

		//	return maxSystemOutputWithout_suc_vec;
		//}
		



		//// Recompute the features of the successors nodes
		//double MDP::recomputeFeatures(State& state, const bool CurNode) const
		//{
		//	vector<int64_t> maxOutput_successor2_vec;
		//	vector<int64_t> maxOutput_successor_vec;
		//	vector<int64_t> maxOutput_curN_vec;

		//	int64_t min_max_additional_output;


		//	// Recompute current node cumInvNode
		//	if (CurNode) {
		//		// Until the current order, at lead time, nothing changed
		//		for (int64_t l = 0; l <= Leadtime_vec.at(state.currentNode); l++) {
		//			maxOutput_curN_vec.push_back(state.cumInvNode_2vec.at(state.currentNode).at(l));
		//		}
		//		// From 1 of cumInvNode vec of predecessors as currentNode has just ordered
		//		for (int64_t l = 1; l < endEchelonLead - Echelonlead_vec.at(state.currentNode); l++) {
		//			min_max_additional_output = CapacityofNodes_vec.at(state.currentNode) * (l);
		//			for (auto node : EPredecessor_2vec.at(state.currentNode)) {
		//				if (state.cumInvNode_2vec.at(node).at(l) < min_max_additional_output) {
		//					min_max_additional_output = state.cumInvNode_2vec.at(node).at(l);
		//				}
		//			}
		//			// Cannot add more than capacity
		//			if (maxOutput_curN_vec.at(maxOutput_curN_vec.size() - 1) + CapacityofNodes_vec.at(state.currentNode) - state.cumInvNode_2vec.at(state.currentNode).at(Leadtime_vec.at(state.currentNode)) < min_max_additional_output) {
		//				min_max_additional_output = maxOutput_curN_vec.at(maxOutput_curN_vec.size() - 1) + CapacityofNodes_vec.at(state.currentNode) - state.cumInvNode_2vec.at(state.currentNode).at(Leadtime_vec.at(state.currentNode));
		//			}
		//			maxOutput_curN_vec.push_back(state.cumInvNode_2vec.at(state.currentNode).at(Leadtime_vec.at(state.currentNode)) + min_max_additional_output);
		//		}
		//		state.cumInvNode_2vec.at(state.currentNode) = maxOutput_curN_vec;
		//	}


		//	// Backward induction to compute max output from nodes with predecessors
		//	// Maximum system output at all future periods for the entire system
		//	// NOTE successors have already ordered, but you want to keep equal until the current production as from there on it may change
		//	for (int64_t n : ESuccessor_2vec.at(state.currentNode)) {				// Greater than zero or than one
		//		//All successors have already ordered in this time period

		//		// Note that in the cumInvNode_2vec already all upstream inventories + capacities of node n are taken into account

		//		// Until the current order, at lead time of n which is Leadtime_vec.at(n) + Leadtime_vec.at(state.currentNode) element, nothing changed
		//		for (int64_t l = 0; l < Leadtime_vec.at(n) + Leadtime_vec.at(state.currentNode); l++) {
		//			maxOutput_successor_vec.push_back(state.cumInvNode_2vec.at(n).at(l));
		//		}

		//		// Multiple predecessors of which one is the currentNode
		//		// From lead time of state.currentNode, for predecessors of successorsnode, it is from lead time of state.currentNode, before that nothing is changed
		//		for (int64_t l = Leadtime_vec.at(state.currentNode); l < endEchelonLead - Echelonlead_vec.at(n); l++) {
		//			min_max_additional_output = CapacityofNodes_vec.at(n) * (l);
		//			// It cannot be higher than the sum of the inventories in the predecessors nodes until l
		//			for (auto node : EPredecessor_2vec.at(n)) {
		//				if (state.cumInvNode_2vec.at(node).at(l) < min_max_additional_output) {
		//					min_max_additional_output = state.cumInvNode_2vec.at(node).at(l);
		//				}
		//			}
		//			// And cannot add more than capacity compared to the previous l
		//			if (maxOutput_successor_vec.at(maxOutput_successor_vec.size() - 1) + CapacityofNodes_vec.at(n) - state.cumInvNode_2vec.at(n).at(Leadtime_vec.at(n)) < min_max_additional_output) {
		//				min_max_additional_output = maxOutput_successor_vec.at(maxOutput_successor_vec.size() - 1) + CapacityofNodes_vec.at(n) - state.cumInvNode_2vec.at(n).at(Leadtime_vec.at(n));
		//			}
		//			maxOutput_successor_vec.push_back(state.cumInvNode_2vec.at(n).at(Leadtime_vec.at(n)) + min_max_additional_output);
		//		}
		//		state.cumInvNode_2vec.at(n) = maxOutput_successor_vec;


		//		for (auto successorsnode2 : ESuccessor_2vec.at(n)) {
		//			//All successors have already ordered in this time period

		//			// Until lead time of the current node nothing changes, which is for successorsnode2, the ... index (see below)
		//			for (int64_t l = 0; l < Leadtime_vec.at(n) + Leadtime_vec.at(state.currentNode) + Leadtime_vec.at(successorsnode2); l++) {
		//				maxOutput_successor2_vec.push_back(state.cumInvNode_2vec.at(successorsnode2).at(l));
		//			}

		//			// Multiple predecessors of which one is the branch with currentNode

		//			// From lead time of state.currentNode + lead time of node n, for predecessors of successorsnode, it is from that lead time as before that nothing is changed
		//			for (int64_t l = Leadtime_vec.at(n) + Leadtime_vec.at(state.currentNode); l < endEchelonLead - Echelonlead_vec.at(successorsnode2); l++) {
		//				// Multiple predecessors of which maximally the capacity can be ordered per time period
		//				min_max_additional_output = CapacityofNodes_vec.at(successorsnode2) * (l);
		//				// It cannot be higher than the sum of the inventories in the predecessors nodes until l
		//				for (auto node : EPredecessor_2vec.at(successorsnode2)) {
		//					if (state.cumInvNode_2vec.at(node).at(l) < min_max_additional_output) {
		//						min_max_additional_output = state.cumInvNode_2vec.at(node).at(l);
		//					}
		//				}
		//				// And cannot add more than capacity compared to the previous l
		//				if (maxOutput_successor2_vec.at(maxOutput_successor2_vec.size() - 1) + CapacityofNodes_vec.at(successorsnode2) - state.cumInvNode_2vec.at(successorsnode2).at(Leadtime_vec.at(successorsnode2)) < min_max_additional_output) {
		//					min_max_additional_output = maxOutput_successor2_vec.at(maxOutput_successor2_vec.size() - 1) + CapacityofNodes_vec.at(successorsnode2) - state.cumInvNode_2vec.at(successorsnode2).at(Leadtime_vec.at(successorsnode2));
		//				}
		//				// Add the cumInvNode to the right vector
		//				maxOutput_successor2_vec.push_back(state.cumInvNode_2vec.at(successorsnode2).at(Leadtime_vec.at(successorsnode2)) + min_max_additional_output);
		//			}
		//			state.cumInvNode_2vec.at(successorsnode2) = maxOutput_successor2_vec;

		//		}
		//	}


		//	return 0.0;
		//}
	}
}

