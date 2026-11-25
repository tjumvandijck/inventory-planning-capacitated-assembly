#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"
#include "dynaplex/modelling/queue.h"
#include <map>

// Library imports
#include <deque>
#include <string>
#include <queue>
#include <vector>
using std::deque;
using std::string;
using std::vector;

// Local imports
#include "preComputerNS.h"

namespace DynaPlex::Models {
	namespace asml_assembly_nonstationary /*must be consistent everywhere for complete mdp definition and associated policies and states.*/
	{
		class MDP
		{
		public:
			//This is optional, if you want the discount_factor to be configurable.
			//The static MDP information returned by GetStaticMDP sets the actual discount_factor used by DynaPlex. 
			//if not provided, defaults to 1.0.  
			double  discount_factor;

			// Note that 0 is most upstream node, and N most downstream node

			// MDP configuration variables
			int64_t NumberofNodes{ 0 };						// The number of nodes in the system
			vector<int64_t> Leadtime_vec;					// Contains the local production lead times
			double p{ 0.0 };								// Contains the backlog penalty
			vector<double> h_vec;							// Contains the holding costs per node
			vector<int64_t> CapacityofNodes_vec;			// Contains the capacity per node
			vector<vector<int64_t>> BOMmatrix_2vec;			// Contains the BOM matrix

			//Other MDP variables:
			int64_t lostsalespenalty{ 0 };				// Contains the lost sales penalty
			int64_t MaxBacklog{ 0 };					// Contains the maximum backlog for state space refinement
			vector<int64_t> MaxInvatNodes_vec;			// Contains the maximum inventory per node for state space refinement
			int64_t minimumorder;						// Contains the minimum order quantity, i.e., the minimum demand realization
			vector<vector<double>> moveProb_2vec;		// Contains the probability of a move per vertical pos
			vector<double> noiseProb_vec;				// Contains the forecast error distribution
			int64_t startNoise;							// Contains the start of the noise
			int64_t maxTree;							// Contains the max vertical position in demand tree
			int64_t startMeanDem;						// The start mean demand of demand 
			int64_t noiseLength;						// The size of the noiseProb vector
			vector<vector<double>> demandProbsTr_2vec;		// Contains the event probability, including the move and error				
			vector<double> demandProbsTr_vec;				// Contains the event probability, including the move and error				
			int64_t MaxOrderSize;						// Contains the maximum order size, required for number of valid actions


			double  discount_factor_geom;					// Contains the parameter for the Geom distribution for determining the horizon length
			vector<vector<double>> demProbs_tr_2vec;		// Contains demand probabilities including the probability of reaching the final state
			vector<double> demProbs_tr_vec;					// Contains demand probabilities including the probability of reaching the final state
			int64_t instance;								// Contains the instance variable to test characteristics within same configuration

			// For feature engineering
			bool featureEngineering;						// True when feature engineering
			int64_t leadtime_size;							// Contains the size of the vectors in feature engineering
			int64_t endEchelonLead;							// Contains the longest end lead time to consider in feature engineering
			//vector<int64_t> frontpositions_FE_vec;		// Contains the front positions of the cumInvNode_vec
			//vector<int64_t> backpositions_FE_vec;			// Contains the back positions of the cumInvNode_vec
			//int64_t downstreamposition_FE;				// Contains the index of the front position of the most downstream node

			vector<int64_t> frontpositions_eff_FE_vec;			// Contains the front positions of the cumInvNode_eff_vec
			vector<int64_t> backpositions_eff_FE_vec;			// Contains the back positions of the cumInvNode_eff_vec

			vector<DynaPlex::DiscreteDist> demand_dist_vec;		// Class of demand distribution
			DynaPlex::DiscreteDist demand_dist;					// Class of demand distribution


			// Contains variables for the network structure
			bool assembly;										// True when system is assembly system
			bool stationary;									// True when demand is stationary
			bool homogeneous_capacity;							// True when capacity is homogeneous 11
			bool ATO;											// True when system is assemble-to-order
			vector<vector<int64_t>> ESuccessors_2vec;			// Contains the vector of successors per node, including the node itself. For echelon inv pos computation.
			vector<vector<int64_t>> ESuccessor_2vec;			// Contains vector of immediate successor	
			vector<vector<int64_t>> EPredecessor_2vec;			// Contains the immediate predecessor
			vector<vector<double>> EqSerialParam_2vec;			// Vector consisting of the equivalent serial parameters; [0] is added value, [1] is echelon lead
			vector<int64_t> Echelonlead_vec;					// Contains the echelon lead times
			vector<int64_t> SerialEq_vec;						// Vector containing the lead time of node n-1 which we consider for balancing
			vector<int64_t> SameSerial_vec;						// Contains the equivalent nodes
			vector<int64_t> frontposition_vec;					// Contains the on-hand inv position of the flattened state space per node				
			vector<int64_t> backposition_vec;					// Contains the production lead time position of the flattened state space per node
			int64_t downstreamposition;							// Contains the index of the on-hand inventory of the most downstream node
			vector<vector<int64_t>> parallelDownNodes_2vec;		// Contains the indices of downstream nodes to balance with for node n
			vector<vector<int64_t>> parallelUpNodes_2vec;		// Contains the indices of upstrteam nodes to balance with for node n
			vector<bool> nonBottleneckNode_vec;					// True if the node is a non-bottleneck node


			// Base-stock levels
			vector<vector<int64_t>> base_stock_levels;					// Contains the demand tree base-stock levels per vertical pos
			vector<vector<int64_t>> base_stock_levels_shortfall;		// Contains the demand tree base-stock levels with shortfall

			// For testing varying feature engineering options
			bool zerosMaxSystemOutput;									// True if we add zeros in max system output wo brach for most downstream node / serial systems
			vector<vector<double>> avgdem_lt_vec;							


			//A state is a struct (or class) that represents state information for the MDP:
			struct State {
				DynaPlex::StateCategory cat;
				Queue<int64_t> StateVectors;
				vector<int64_t> echelonInv_vec;
				int64_t currentNode{ 0 };
				int64_t verticalpos{ 0 };
				int64_t period{ 0 };
				bool warmUp{ false };
				//vector<int64_t> cumInvNode_vec;
				vector<int64_t> cumInvNode_eff_vec;
				vector<int64_t> features_vec;
				// A vector containing the maximum system output corresponding to feature engineering
				// [node, with /without own branch, #periods in future (i.e., above echelon lead)] output is maximum system output
				vector<vector<vector<int64_t>>> FeatureEngineering;
				DynaPlex::VarGroup ToVarGroup() const;
				bool operator==(const State& other) const = default;
			};

			using Event = int64_t;


			//Remainder of the DynaPlex API:
			// Function used to check if an action is allowed,
			// returns 'false' if it is unfeasible
			bool IsAllowedAction(const State&, int64_t action) const;

			// Function used to update the MDP's state after an action has been taken
			double ModifyStateWithAction(State&, int64_t action) const;

			// Function used to update the MDP's state after an event updates the environment 
			double ModifyStateWithEvent(State&, const Event&) const;

			// Function that defines the demand probabilities (= event probabilities)
			Event GetEvent(const State&, DynaPlex::RNG&) const;

			// Function that (does not) return the event probabilities
			vector<std::tuple<Event, double>> EventProbabilities(const State&) const;			
			
			//// Function that defines the demand probabilities (= event probabilities)
			//Event GetEvent(DynaPlex::RNG&) const;

			//// Function that (does not) return the event probabilities
			//vector<std::tuple<Event, double>> EventProbabilities() const;

			//You may also define this with a parameter DynaPlex::RNG&, for random initial states:
			// Function that constructs the MDP's initial state
			State GetInitialState() const;

			// Function that returns the features that make up the state
			void GetFeatures(const State&, DynaPlex::Features&) const;

			/** Function that returns the flat state vector for the exact solver when the features are not unique per state */
			void GetExactFeatureRepr(const State&, DynaPlex::Features&) const;

			// Function that determines whether in warm-up phase
			bool AdaptWarmUp(State&) const;

			// Function to recompute cumInvNode & maxSystemOutput from currentNode, for Feature Engineering
			//double recomputeFeatures(State&, const bool CurNode) const;
			double recomputeFeatures_eff_singlevec(State&, const bool CurNode) const;

			// Function to compute the max System output without branch
			//vector<int64_t> maxSystemOutputWithout(const State&) const;
			vector<int64_t> maxSystemOutputWithout_eff_singlevec(const State&) const;

			//Remainder of the DynaPlex API:
			DynaPlex::VarGroup GetStaticInfo() const;
			DynaPlex::StateCategory GetStateCategory(const State&) const;
			State GetState(const VarGroup&) const;
			//Enables all MDPs to be constructer in a uniform manner.
			explicit MDP(const DynaPlex::VarGroup&);
			void RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>&) const;

		};
	}
}

