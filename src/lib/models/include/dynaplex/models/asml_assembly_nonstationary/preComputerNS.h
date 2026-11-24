#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"
#include "dynaplex/modelling/queue.h"
#include <map>

// Library imports for reading the files
#include <fstream>
#include <iostream>
#include <istream>
using std::ifstream;
using std::istream;

// Library imports
#include <deque>
#include <string>
#include <queue>
#include <vector>
using std::deque;
using std::string;
using std::vector;

// Precompute the systems variables - class definition
class preComputerNS {

private:

	// Mark this as private
	preComputerNS() {}

public:

	// Initialization function definition
	static preComputerNS getPreComputer(const int64_t NumberofNodes, const double p, const bool assembly, const double discount_factor_geom, const bool stationary, const int64_t instance, const bool homogeneous_capacity, const bool ATO);


	// MDP configuration variables
	int64_t NumberofNodes;						// The number of nodes in the system
	double p;									// Contains the backlog penalty
	bool assembly;								// True when we consider an assembly system
	double discount_factor_geom;				// Contains the parameter for the Geom distribution for determining the horizon length
	bool stationary;							// True when we consider stationary demand
	int64_t instance;							// Contains the instance to test multiple things
	bool homogeneous_capacity;				// True when we consider an instance with equal capacity 11 
	bool ATO;									// True when we consider an instance with assemble-to-order network

	DynaPlex::DiscreteDist cust_dist;
	DynaPlex::DiscreteDist cust_dist2;

	// Return type class definition, 
	// used to return parameters in one return statement
	struct paramSystemInstance {
		// MDP configuration variables
		const vector<vector<int64_t>> BOMmatrix_2vec;		// Contains the BOM matrix
		const vector<int64_t> Leadtime_vec;					// Contains the local production lead times
		const vector<double> h_vec;							// Contains the holding costs per node
		const vector<int64_t> CapacityofNodes_vec;			// Contains the capacity per node
		const vector<vector<double>> moveProb_2vec;					// Contains the probability of a move
		const vector<double> noiseProb_vec;					// Contains the forecast error distribution
		const int64_t noiseLength;							// The size of the noiseProb vector
		const int64_t startNoise;							// Contains the start of the noise
		const int64_t maxTree;								// Contains the max vertical position in demand tree
		const int64_t startMeanDem;							// The start mean demand of demand 
		const string BSlevelsDT_round;						// Contains the best rounding for DT base-stock levels without shortfall
		const string BSlevelsDT_shortfall_round;			// Contains the best rounding for DT base-stock levels with shortfall
		const string BSlevelsCN_round;						// Contains the best rounding for CN base-stock levels without shortfall
		const string BSlevelsCN_shortfall_round;			// Contains the best rounding for CN base-stock levels with shortfall
		const vector<int64_t> MaxInvatNodes_vec;			// Contains the maximum inventory per node for state space refinement
		const int64_t minimumorder;							// Contains the minimum order quantity, i.e., the minimum demand realization
		const int64_t lostsalespenalty;						// Contains the lost sales penalty
		const int64_t MaxBacklog;							// Contains the maximum backlog for state space refinement

		const vector<vector<int64_t>> ESuccessors_2vec;		// Contains the vector of successors per node, including the node itself
		const vector<vector<int64_t>> ESuccessor_2vec;		// Contains vector of immediate successor	
		const vector<vector<int64_t>> EPredecessor_2vec;	// Contains the immediate predecessors
		const vector<vector<double>> EqSerialParam_2vec;	// Vector consisting of the equivalent serial parameters; [0] is added value, [1] is echelon lead
		const vector<int64_t> Echelonlead_vec;				// Contains the echelon lead times
		const vector<int64_t> SerialEq_vec;					// Vector containing the lead time of node n-1 which we consider for balancing
		const vector<int64_t> SameSerial_vec;				// Contains the equivalent nodes
		const vector<int64_t> frontposition_vec;			// Contains the on-hand inv position of the flattened state space per node				
		const vector<int64_t> backposition_vec;				// Contains the production lead time position of the flattened state space per node
		const int64_t downstreamposition;					// Contains the index of the on-hand inventory of the most downstream node
		const vector<vector<int64_t>> parallelDownNodes_2vec;	// Contains the indices of downstream nodes to balance with for node n
		const vector<vector<int64_t>> parallelUpNodes_2vec;	// Contains the indices of upstrteam nodes to balance with for node n
		const vector<bool> nonBottleneckNode_vec;			// True if the node is a non-bottleneck node

		const vector<vector<double>> demandProbsTr_2vec;				// Contains the event probability, including the move and error	per vertical pos			
		const vector<vector<double>> demProbs_tr_2vec;				// Contains demand probabilities including the probability of reaching the final state per vertical pos
		const vector<vector<double>> avgdem_lt_vec;							// Contains the average demand over lead time

		// Base-stock levels
		const vector<vector<int64_t>> BSlevelsDT_2vec;					// Contains the echelon DT base stock levels
		const vector<vector<int64_t>> BSlevelsDT_short_2vec;			// Contains the echelon DT base stock levels with our shortfall distribution
		const vector<vector<int64_t>> BSlevelsDT_shortsingle_2vec;		// Contains the echelon DT base stock levels with shortfall distribution computed by Huh et al.
		const vector<vector<int64_t>> BSlevelsCN_2vec;					// Contains the echelon CN base stock levels
		const vector<vector<int64_t>> BSlevelsCN_short_2vec;			// Contains the echelon CN base stock levels with our shortfall distribution
		const vector<vector<int64_t>> BSlevelsCN_shortsingle_2vec;		// Contains the echelon CN base stock levels with shortfall distribution computed by Huh et al.
		const vector<vector<double>> fractiles;							// Contains the fractiles

		paramSystemInstance(
			const vector<int64_t> Leadtime_vec, const vector<double> h_vec, const vector<int64_t> CapacityofNodes_vec, const vector<vector<double>> moveProb_2vec, const vector<double> noiseProb_vec,
			const int64_t startNoise, const int64_t maxTree, const int64_t startMeanDem, const vector<vector<int64_t>> BOMmatrix_2vec, const int64_t lostsalespenalty, const int64_t MaxBacklog,
			const int64_t minimumorder, const int64_t noiseLength, const vector<int64_t> MaxInvatNodes_vec,
			const vector<vector<int64_t>> ESuccessors_2vec, const vector<vector<int64_t>> ESuccessor_2vec, const vector<vector<int64_t>> EPredecessor_2vec,
			const vector<vector<double>> EqSerialParam_2vec, const vector<int64_t> Echelonlead_vec, const vector<int64_t> SerialEq_vec, const vector<int64_t> SameSerial_vec,
			const vector<vector<int64_t>> BSlevelsDT_2vec, const vector<vector<int64_t>> BSlevelsDT_short_2vec, const vector<vector<int64_t>> BSlevelsDT_shortsingle_2vec,
			const vector<vector<int64_t>> BSlevelsCN_2vec, const vector<vector<int64_t>> BSlevelsCN_short_2vec, const vector<vector<int64_t>> BSlevelsCN_shortsingle_2vec,
			const vector<int64_t> frontposition_vec, const vector<int64_t> backposition_vec, const int64_t downstreamposition, const vector<vector<int64_t>> parallelDownNodes_2vec,
			const vector<vector<int64_t>> parallelUpNodes_2vec, const vector<vector<double>> demProbs_tr_2vec, const vector<vector<double>> avgdem_lt_vec, const vector<bool> nonBottleneckNode_vec)
			:Leadtime_vec{ Leadtime_vec }, h_vec{ h_vec }, CapacityofNodes_vec{ CapacityofNodes_vec }, moveProb_2vec{ moveProb_2vec }, noiseProb_vec{ noiseProb_vec },
			startNoise{ startNoise }, maxTree{ maxTree }, startMeanDem{ startMeanDem }, BOMmatrix_2vec{ BOMmatrix_2vec }, lostsalespenalty{ lostsalespenalty }, MaxBacklog{ MaxBacklog },
			minimumorder{ minimumorder }, noiseLength{ noiseLength }, MaxInvatNodes_vec{ MaxInvatNodes_vec },
			ESuccessors_2vec{ ESuccessors_2vec }, ESuccessor_2vec{ ESuccessor_2vec }, EPredecessor_2vec{ EPredecessor_2vec },
			EqSerialParam_2vec{ EqSerialParam_2vec }, Echelonlead_vec{ Echelonlead_vec }, SerialEq_vec{ SerialEq_vec }, SameSerial_vec{ SameSerial_vec },
			BSlevelsDT_2vec{ BSlevelsDT_2vec }, BSlevelsDT_short_2vec{ BSlevelsDT_short_2vec }, BSlevelsDT_shortsingle_2vec{ BSlevelsDT_shortsingle_2vec },
			BSlevelsCN_2vec{ BSlevelsCN_2vec }, BSlevelsCN_short_2vec{ BSlevelsCN_short_2vec }, BSlevelsCN_shortsingle_2vec{ BSlevelsCN_shortsingle_2vec },
			frontposition_vec{ frontposition_vec }, backposition_vec{ backposition_vec }, downstreamposition{ downstreamposition },
			parallelDownNodes_2vec{ parallelDownNodes_2vec }, parallelUpNodes_2vec{ parallelUpNodes_2vec },
			demProbs_tr_2vec{ demProbs_tr_2vec }, avgdem_lt_vec{ avgdem_lt_vec }, nonBottleneckNode_vec{ nonBottleneckNode_vec }
		{
		}
	};

	// Function that reads rows,
	// splits them in cells using ';' as separator
	// and returns them as a vector of strings
	vector<string> readRow(const string& row) const;


	// Function that reads files using the readRow(...) function
	vector<vector<string>> readFile(istream& in_stream) const;

	// Read a semicolon-delimited shortfall file and verify that every row is a
	// probability distribution before the base-stock calculations use it.
	vector<vector<double>> readAndValidateShortfallFile(const string& filePath,
		const int64_t expectedRowCount) const;

	paramSystemInstance loadSystem() const;

	// Function to compute the demand over lead time for DT base-stock policy
	//vector<vector<vector<vector<double>>>> getDemandLeadtimeDist(const vector<double> moveProb_vec, const vector<double> noiseProb_vec, const int64_t startNoise, const int64_t maxTree, const int64_t startMeanDem, const vector<int64_t> Echelonlead_vec, const vector<int64_t> SameSerial_vec) const;
	vector<vector<vector<vector<double>>>> getDemandLeadtimeDist(const vector<vector<double>> moveProb_2vec, const vector<double> noiseProb_vec, const int64_t startNoise, const int64_t maxTree, const int64_t startMeanDem, const vector<int64_t> Echelonlead_vec, const vector<int64_t> SameSerial_vec) const;

	// Function to compute the DT base-stock levels
	vector<vector<int64_t>> GetBaseStockLevelsDT(const vector<double> h_vec, const vector<int64_t> Echelonlead_vec, const vector<vector<double>> EqSerialParam_2vec, const vector<int64_t> SameSerial_vec, bool shortfall, vector<vector<vector<double>>> shortfalls_3vec, string rounding_BS_level, const int64_t maxTree, const int64_t startMeanDem, const vector<vector<vector<vector<double>>>> demLeadTimeDistr, const vector<vector<double>> fractiles, const vector<double> noiseProb_vec, const int64_t startNoise) const;

	// Function to compute the CN base-stock levels
	vector<vector<int64_t>> GetBaseStockLevelsCN(const vector<double> h_vec, const vector<int64_t> Echelonlead_vec, const vector<vector<double>> EqSerialParam_2vec, const vector<int64_t> SameSerial_vec, bool shortfall, vector<vector<vector<double>>> shortfalls_3vec, string rounding_BS_level, const vector<double> noiseProb_vec, const int64_t startNoise, const int64_t maxTree, const int64_t startMeanDem, const vector<vector<double>> fractiles) const;

	// Function to compute the factiles in the Shang and Song base-stock level computation
	vector<vector<double>> GetFractileSS(const vector<double> h_vec, const vector<vector<double>> EqSerialParam_2vec, const vector<int64_t> SameSerial_vec, const double p, const int64_t NumberofNodes) const;

	// Function to compute the average demand for each number of periods in the future for GetFeatures
	vector<vector<double>> GetAvgDemand(const vector<vector<double>> moveProb_2vec, const vector<double> noiseProb_vec, const int64_t startNoise, const int64_t maxTree, const int64_t startMeanDem, const vector<int64_t> Echelonlead_vec, const vector<vector<vector<vector<double>>>> demLeadTimeDistr) const;


};

