#pragma once
#include <cstdint>
#include "dynaplex/models/asml_assembly_nonstationary/mdp.h"
#include "dynaplex/vargroup.h"
#include <memory>

namespace DynaPlex::Models {
	namespace asml_assembly_nonstationary /*must be consistent everywhere for complete mdp defininition and associated policies.*/
	{
		// Forward declaration
		class MDP;

		// The base-stock policy
		// The balancing input determines whether it is: BS-N, BS-R, or BS-TW
		// The shortfall boolean is for the base-stock level computation
		class BalancingBaseStockPolicy
		{
			//this is the MDP defined inside the current namespace!
			std::shared_ptr<const MDP> mdp;
			const VarGroup varGroup;
			// Variables for the base-stock policy
			bool shortfall;			// True if we use our shortfall distribution for the for the base-stock level computation
			bool balancing_upstream;				// True if inventories are balanced with upsteam nodes, i.e., the direct adoption of 
			bool balancing_downstream;				// True if inventories are balanced with downstream nodes
			bool balancing_bottleneck;				// True if inventories are balanced for the bottleneck nodes
			bool balancing_bottleneck_downstream;	// True if inventories are balanced downstream for the bottleneck nodes

		public:
			BalancingBaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};
	}
}

