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

		// The BS-DT policy
		class DT_BaseStockPolicy
		{
			//this is the MDP defined inside the current namespace!
			std::shared_ptr<const MDP> mdp;
			const VarGroup varGroup;
			// Variables for the base-stock policy
			bool shortfall;			// True if we use our shortfall distribution for the for the base-stock level computation
			bool Huhetal;			// True if we use the shortfall computed as in Huh et al. for the base-stock level computation
			bool balancing_upstream;			// True if inventories are balanced with upsteam nodes, i.e., the direct adoption of 
			bool balancing_downstream;			// True if inventories are balanced with downstream nodes
			bool balancing_bottleneck;
			bool balancing_bottleneck_downstream;

		public:
			DT_BaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};

		class Cur_D_BaseStockPolicy
		{
			//this is the MDP defined inside the current namespace!
			std::shared_ptr<const MDP> mdp;
			const VarGroup varGroup;
			bool shortfall;			// True if we use our shortfall distribution for the for the base-stock level computation
			bool Huhetal;			// True if we use the shortfall computed as in Huh et al. for the base-stock level computation
			bool balancing_upstream;			// True if inventories are balanced with upsteam nodes, i.e., the direct adoption of 
			bool balancing_downstream;			// True if inventories are balanced with downstream nodes
		public:
			Cur_D_BaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};
	}
}

