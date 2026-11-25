// Local imports
#include "dynaplex/models/asml_assembly_nonstationary/preComputerNS.h"
#include <cmath>
#include <cctype>
#include <numeric>
#include <chrono>


// Function that reads rows,
// splits them in cells using ';' as separator
// and returns them as a vector of strings
vector<string> preComputerNS::readRow(const string& row) const {
	vector<string> cells{ "" };
	int64_t i = 0;
	for (char c : row) {
		switch (c) {
		case ';':
			cells.push_back("");
			++i;
			break;
		default:
			cells[i].push_back(c);
			break;
		}
	}
	return cells;
}

// Function that reads files using the readRow(...) function
vector<vector<string>> preComputerNS::readFile(istream& in_stream) const {
	vector<vector<string>> table;
	string row;
	while (!in_stream.eof()) {
		getline(in_stream, row);
		if ((in_stream.bad()) || (in_stream.fail())) {
			break;
		}
		vector<string> fields = readRow(row);
		table.push_back(fields);
	}
	return table;
}

// Read the shortfall distributions once, while validating the file structure
// and its numerical content.  Each row is one probability mass function: its
// column index is the shortfall amount (0, 1, 2, ...).
vector<vector<double>> preComputerNS::readAndValidateShortfallFile(
	const string& filePath, const int64_t expectedRowCount) const {
	ifstream shortfallFile(filePath);
	if (!shortfallFile.is_open()) {
		throw DynaPlex::Error("asml_assembly_nonstationary: cannot open shortfall CSV file '" + filePath + "'");
	}

	const vector<vector<string>> rows = readFile(shortfallFile);
	if (static_cast<int64_t>(rows.size()) != expectedRowCount) {
		throw DynaPlex::Error(
			"asml_assembly_nonstationary: shortfall CSV file '" + filePath + "' has "
			+ std::to_string(rows.size()) + " rows; expected "
			+ std::to_string(expectedRowCount));
	}

	vector<vector<double>> distributions;
	distributions.reserve(rows.size());
	constexpr double probabilityTolerance = 1e-8;

	for (size_t rowIndex = 0; rowIndex < rows.size(); ++rowIndex) {
		const vector<string>& row = rows.at(rowIndex);
		if (row.empty() || (row.size() == 1 && row.front().empty())) {
			throw DynaPlex::Error("asml_assembly_nonstationary: shortfall CSV file '" + filePath
				+ "' contains an empty row at row " + std::to_string(rowIndex + 1));
		}

		// Rows may have different lengths: each node can require a different
		// maximum represented shortfall amount.  Validate each probability mass
		// function independently rather than imposing a common support length.
		vector<double> distribution;
		distribution.reserve(row.size());
		double probabilitySum = 0.0;
		for (size_t columnIndex = 0; columnIndex < row.size(); ++columnIndex) {
			const string& entry = row.at(columnIndex);
			size_t parsedCharacters = 0;
			double probability = 0.0;
			try {
				probability = std::stod(entry, &parsedCharacters);
			}
			catch (const std::exception&) {
				throw DynaPlex::Error("asml_assembly_nonstationary: shortfall CSV file '" + filePath
					+ "' has a non-numeric value at row " + std::to_string(rowIndex + 1)
					+ ", column " + std::to_string(columnIndex + 1));
			}
			for (size_t characterIndex = parsedCharacters; characterIndex < entry.size(); ++characterIndex) {
				if (!std::isspace(static_cast<unsigned char>(entry.at(characterIndex)))) {
					throw DynaPlex::Error("asml_assembly_nonstationary: shortfall CSV file '" + filePath
						+ "' has a non-numeric value at row " + std::to_string(rowIndex + 1)
						+ ", column " + std::to_string(columnIndex + 1));
				}
			}
			if (!std::isfinite(probability) || probability < 0.0) {
				throw DynaPlex::Error("asml_assembly_nonstationary: shortfall CSV file '" + filePath
					+ "' has an invalid probability at row " + std::to_string(rowIndex + 1)
					+ ", column " + std::to_string(columnIndex + 1));
			}
			probabilitySum += probability;
			distribution.push_back(probability);
		}
		if (std::abs(probabilitySum - 1.0) > probabilityTolerance) {
			throw DynaPlex::Error("asml_assembly_nonstationary: probabilities in shortfall CSV file '" + filePath
				+ "' do not sum to one at row " + std::to_string(rowIndex + 1)
				+ " (sum is " + std::to_string(probabilitySum) + ")");
		}
		distributions.push_back(std::move(distribution));
	}

	return distributions;
}

// Initialization function definition
preComputerNS preComputerNS::getPreComputer(const int64_t NumberofNodes, const double p, const bool assembly, const double discount_factor_geom, const bool stationary, const int64_t instance, const bool homogeneous_capacity, const bool ATO)
{
	preComputerNS preComputerNS;		// Create a preComputer instance

	// Set the preComputers parameters
	preComputerNS.NumberofNodes = NumberofNodes;
	preComputerNS.p = p;
	preComputerNS.discount_factor_geom = discount_factor_geom;
	preComputerNS.assembly = assembly;
	preComputerNS.stationary = stationary;
	preComputerNS.instance = instance;
	preComputerNS.homogeneous_capacity = homogeneous_capacity;
	preComputerNS.ATO = ATO;

	return preComputerNS;				// Return the preComputer instance
}

// Function that loads system parameters from input
// parses them 
// and returns them in a parameter Instance
preComputerNS::paramSystemInstance preComputerNS::loadSystem() const {

	// Initialize variables
	vector<vector<int64_t>> BOMmatrix_2vec;				// Contains the BOM matrix
	vector<int64_t> Leadtime_vec;						// Contains the local production lead times
	vector<double> h_vec;								// Contains the holding costs per node
	vector<int64_t> CapacityofNodes_vec;				// Contains the capacity per node
	vector<vector<double>> moveProb_2vec;				// Contains the probability of a move
	vector<double> noiseProb_vec;						// Contains the forecast error distribution
	int64_t noiseLength;								// Contains the length of the noise vector
	int64_t startNoise;									// Contains the start of the noise
	int64_t maxTree;									// Contains the max vertical position in demand tree
	int64_t startMeanDem;								// The start mean demand of demand 
	string base_stock_levels_round;							// Contains the best rounding for DT base-stock levels without shortfall
	string BSlevelsDT_shortfall_round;					// Contains the best rounding for DT base-stock levels with shortfall
	string base_stock_levels_shortfall_round;			// Contains the best rounding for DT base-stock levels with shortfall
	vector<int64_t> MaxInvatNodes_vec;					// Contains the maximum inventory per node for state space refinement
	int64_t minimumorder;								// Contains the minimum order quantity, i.e., the minimum demand realization
	int64_t lostsalespenalty;							// Contains the lost sales penalty
	int64_t MaxBacklog;									// Contains the maximum backlog for state space refinement

	vector<vector<int64_t>> ESuccessors_2vec;		// Contains the vector of successors per node, including the node itself
	vector<vector<int64_t>> ESuccessor_2vec;		// Contains vector of immediate successor	
	vector<vector<int64_t>> EPredecessor_2vec;		// Contains the immediate predecessor
	vector<vector<double>> EqSerialParam_2vec;		// Vector consisting of the equivalent serial parameters; [0] is added value, [1] is echelon lead
	vector<int64_t> Echelonlead_vec;				// Contains the echelon lead times
	vector<int64_t> SerialEq_vec;					// Vector containing the lead time of node n-1 which we consider for balancing
	vector<int64_t> SameSerial_vec;					// Contains the equivalent nodes
	vector<int64_t> basestocklevels_vec;			// Contains the echelon base stock levels
	vector<int64_t> frontposition_vec;				// Contains the on-hand inv position of the flattened state space per node				
	vector<int64_t> backposition_vec;				// Contains the production lead time position of the flattened state space per node
	int64_t downstreamposition;						// Contains the index of the on-hand inventory of the most downstream node
	vector<vector<int64_t>> parallelDownNodes_2vec;			// Contains the indices of downstream nodes to balance with for node n
	vector<vector<int64_t>> parallelUpNodes_2vec;			// Contains the indices of upstream nodes to balance with for node n
	vector<bool> nonBottleneckNode_vec;						// True if the node is a non-bottleneck node

	vector<double> demandProbsTr_vec;						// Contains the event probability, including the move and error			
	vector<double> demProbs_tr_vec;							// Contains demand probabilities including the probability of reaching the final state	
	vector<vector<double>> demandProbsTr_2vec;				// Contains the event probability, including the move and error			
	vector<vector<double>> demProbs_tr_2vec;				// Contains demand probabilities including the probability of reaching the final state	

	// Base-stock levels
	vector<vector<int64_t>> base_stock_levels;					// Contains the echelon DT base stock levels
	vector<vector<int64_t>> base_stock_levels_shortfall;		// Contains the echelon DT base stock levels with shortfall distribution computed by Huh et al.

	std::vector<double> noiseprob4 = { 0.08,0.15, 0.2, 0.23, 0.15, 0.08, 0.05, 0.03, 0.02, 0.01 };
	std::vector<double> noiseprobE1 = { 0.16,0.23,0.26, 0.2, 0.1, 0.05 };

	std::vector<double> demandlist_20 = { 0.02, 0.05, 0.1, 0.14, 0.20, 0.19, 0.1, 0.06, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01 };
	std::vector<double> demandlistE_3 = { 0.04, 0.12, 0.16, 0.2, 0.23, 0.14, 0.06, 0.03, 0.02 };

	// read the shortfall input file
	// First decide which file to read depending on the input
	string path = "../../../src/lib/models/models/asml_assembly_nonstationary/data_shortfall/";
	string filename_shortfall = "small_case_NS_L";


	// Determine the case of the corresponding MDP configuration
	if ((NumberofNodes == 3) && (!assembly))
	{
		// General parameters		
		Leadtime_vec = { 1, 1, 1 };
		h_vec = { 1,3,8 };
		lostsalespenalty = 5000;
		BOMmatrix_2vec = {
				{ 0, 1, 0 },
				{ 0, 0, 1 },
				{ 0, 0, 0 },
		};

		// non-stationary & stationary specific parameters
		if (!stationary) {
			CapacityofNodes_vec = { 4,4,4 };
			MaxBacklog = 26;
			moveProb_2vec = { {0.8, 0.2}, {0.8, 0.2}, {0.8, 0.2}, {0.8, 0.2} };
			noiseProb_vec = noiseprobE1;
			startNoise = -2;
			startMeanDem = 2;
			maxTree = 4;
			MaxInvatNodes_vec = { 12,12,26 };
			minimumorder = 0;
			if (p == 50) {
				// Best rounding
				base_stock_levels_round = "round-up";
				base_stock_levels_shortfall_round = "round-down";
			}
			else {
				// Best rounding
				base_stock_levels_round = "round-down";
				base_stock_levels_shortfall_round = "round-down";
			}
			filename_shortfall = "small_case_NS_L";

		}
		else {
			// TODO: Check shortfall!!
			CapacityofNodes_vec = { 10, 10, 10 };
			MaxBacklog = 25;
			moveProb_2vec = { {1.0,0} };
			noiseProb_vec = demandlist_20;
			startNoise = -3;
			startMeanDem = 7;
			maxTree = 7;
			MaxInvatNodes_vec = { 21, 21, 35 };
			minimumorder = 4;
			if (p == 50) {
				// Best rounding
				base_stock_levels_round = "round-down";
				base_stock_levels_shortfall_round = "round-down";
			}
			else {
				// Best rounding
				base_stock_levels_round = "round-down";
				base_stock_levels_shortfall_round = "minimum";
			}
			filename_shortfall = "small_case_S_H";
		}

		noiseLength = noiseProb_vec.size();
	}
	else if ((NumberofNodes == 3) && (assembly) && (homogeneous_capacity))
	{
		// General parameters		
		Leadtime_vec = { 2, 1, 1 };
		h_vec = { 2,1,8 };
		lostsalespenalty = 5000;
		BOMmatrix_2vec = {
				{ 0, 0, 1 },
				{ 0, 0, 1 },
				{ 0, 0, 0 },
		};
		CapacityofNodes_vec = { 4,4,4 };


		// non-stationary & stationary specific parameters
		if (!stationary) {
			moveProb_2vec = { {0.8, 0.2}, {0.8, 0.2}, {0.8, 0.2}, {0.8, 0.2} };
			noiseProb_vec = noiseprobE1;
			startNoise = -2;
			startMeanDem = 2;
			maxTree = 4;
			minimumorder = 0;
			if (p == 50) {
				// Best rounding
				base_stock_levels_round = "maximum";
				base_stock_levels_shortfall_round = "maximum";

				MaxBacklog = 10;							// Changed this to 10 as inv cap has been reduced
				MaxInvatNodes_vec = { 14,12,19 };
			}
			else {
				// Best rounding
				base_stock_levels_round = "round-down";
				base_stock_levels_shortfall_round = "round-down";
				MaxBacklog = 14;
				MaxInvatNodes_vec = { 12,8,23 };
			}
			filename_shortfall = "small_case_NS_L";

		}
		else {
			moveProb_2vec = { { 1.0, 0} };
			noiseProb_vec = demandlistE_3;
			startNoise = -2;
			startMeanDem = 2;
			maxTree = 2;
			MaxInvatNodes_vec = { 12,11,18 };
			MaxBacklog = 18;
			minimumorder = 0;
			if (p == 50) {
				// Best rounding
				base_stock_levels_round = "round-down";
				base_stock_levels_shortfall_round = "minimum";

			}
			else {
				// Best rounding
				base_stock_levels_round = "minimum";
				base_stock_levels_shortfall_round = "minimum";
			}
			filename_shortfall = "small_case_S_L";

		}
		noiseLength = noiseProb_vec.size();
	}
	else if ((NumberofNodes == 3) && (assembly) && (!homogeneous_capacity))
	{
		// General parameters		
		Leadtime_vec = { 1, 1, 1 };
		h_vec = { 1.5,2.5,8 };
		lostsalespenalty = 5000;
		BOMmatrix_2vec = {
				{ 0, 0, 1 },
				{ 0, 0, 1 },
				{ 0, 0, 0 },
		};

		if (!stationary) {
			CapacityofNodes_vec = { 4,4,4 };
			moveProb_2vec = { {0.8, 0.2}, {0.8, 0.2}, {0.8, 0.2}, {0.8, 0.2} };
			noiseProb_vec = noiseprobE1;
			startNoise = -2;
			startMeanDem = 2;
			maxTree = 4;
			minimumorder = 0;
			if (p == 50) {
				// Best rounding
				base_stock_levels_round = "round-up";
				base_stock_levels_shortfall_round = "round-up";
			}
			else {
				// Best rounding
				base_stock_levels_round = "round-up";
				base_stock_levels_shortfall_round = "round-up";
			}
			
			filename_shortfall = "small_case_NS_L";

			MaxInvatNodes_vec = { 11, 11, 22 };
			MaxBacklog = 34;							
		}
		else {
			CapacityofNodes_vec = { 10, 10, 10 };
			moveProb_2vec = { {1.0,0} };
			noiseProb_vec = demandlist_20;
			startNoise = -3;
			startMeanDem = 7;
			maxTree = 7;
			MaxInvatNodes_vec = { 12, 12, 30 }; 
			MaxBacklog = 35; 
			minimumorder = 4;
			if (p == 50) {
				// Best rounding
				base_stock_levels_round = "maximum";
				base_stock_levels_shortfall_round = "round-up";
			}
			else {
				// Best rounding
				base_stock_levels_round = "maximum";
				base_stock_levels_shortfall_round = "round-up";

			}
			filename_shortfall = "small_case_S_H";

		}
		noiseLength = noiseProb_vec.size();
	}
	else if ((NumberofNodes == 2) && (!assembly))
	{
		// General parameters		
		Leadtime_vec = { 2, 1 };
		h_vec = { 1,8 };
		lostsalespenalty = 5000;
		BOMmatrix_2vec = {
			{ 0, 1 },
			{ 0, 0 },
		};
		CapacityofNodes_vec = { 10, 10 };
		MaxBacklog = 35;
		MaxInvatNodes_vec = { 43, 43 };
		minimumorder = 4;
		// non-stationary & stationary specific parameters
		if (!stationary) {
			moveProb_2vec = { {0.8, 0.2}, {0.8, 0.2}, {0.8, 0.2}, {0.8, 0.2} };
			noiseProb_vec = noiseprob4;
			startNoise = -3;
			startMeanDem = 7;
			maxTree = 10;
			if (p == 50) {

				base_stock_levels_round = "round-up";
				base_stock_levels_shortfall_round = "round-up";
			}
			else {

				base_stock_levels_round = "round-down";
				base_stock_levels_shortfall_round = "round-down";
			}
			filename_shortfall = "small_case_NS_H";

		}
		else {
			throw DynaPlex::Error("asml_assembly_nonstationary: no setting found for the given number of nodes");
		}
	}
	else if ((NumberofNodes == 10) && (assembly)) {
		if (!ATO) {
			Leadtime_vec = { 6, 5, 2, 4, 1, 2, 3, 2, 1, 2 };
			h_vec = { 1.05, 0.4, 0.25, 0.2, 0.15, 0.1, 1.5, 0.2, 1.25, 8 };
			BOMmatrix_2vec = {
					{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
					{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
					{ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
					{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
					{ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
					{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
					{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
					{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
					{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
					{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			};
			if (!homogeneous_capacity) {
				CapacityofNodes_vec = { 10, 11, 10, 10, 10, 10, 10, 11, 11, 11 };
			}
			else if (homogeneous_capacity) {
				CapacityofNodes_vec = { 11, 11, 11, 11, 11, 11, 11, 11, 11, 11 };
			}
		}
		else {
			Leadtime_vec = { 6, 5, 4, 3, 2, 2, 2, 1, 1, 2 };
			h_vec = { 1.05, 0.4, 0.2, 1.1, 0.25, 0.1, 0.2, 0.15, 0.55, 8 };
			BOMmatrix_2vec = {
					{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
					{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
					{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
					{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
					{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
					{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
					{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
					{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
					{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
					{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			};
			CapacityofNodes_vec = { 10, 11, 10, 10, 10, 10, 11, 10, 11, 11 };
		}

		// For the leading case, we do not set a maximum backlog and maximum inventory constraint
		lostsalespenalty = 5000;
		MaxBacklog = 2000;
		std::cout << "Maxbacklog is " << MaxBacklog << std::endl;
		MaxInvatNodes_vec = { 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000 };
		minimumorder = 4;

		if (stationary) {
			moveProb_2vec = { {1.0,0} };
			noiseProb_vec = demandlist_20;
			startNoise = -3;
			startMeanDem = 7;
			maxTree = 7;

			if (p == 100) {
				base_stock_levels_round = "round-down";
				base_stock_levels_shortfall_round = "round-down";
				filename_shortfall = "leading_case_S";
			}
			else {
				throw DynaPlex::Error("asml_assembly_nonstationary: no setting found for the given number of nodes");
			}
		}
		else {
			moveProb_2vec = { {0.803, 0.197}, {0.805, 0.195}, {0.8300, 0.1700}, {1, 0} };
			noiseProb_vec = { 0.04, 0.18, 0.21, 0.23, 0.14, 0.09, 0.07, 0.04 };

			startNoise = -3;
			startMeanDem = 7;
			maxTree = 10;

			if (!homogeneous_capacity) {
				if (!ATO) {
					// Best rouding options 
					if (p == 50) {
						base_stock_levels_round = "maximum";
						base_stock_levels_shortfall_round = "maximum";
					}
					else {
						base_stock_levels_round = "maximum";
						base_stock_levels_shortfall_round = "maximum";
					}
					filename_shortfall = "leading_case_NS";
				}
				if (ATO) {
					if (p == 100) {
						// Best rounding options
						base_stock_levels_round = "maximum";
						base_stock_levels_shortfall_round = "maximum";
					}
					filename_shortfall = "leading_case_ATO";
				}
			}
			else if (homogeneous_capacity) {
				filename_shortfall = "leading_case_cap_11";
				if (p == 100) {
					// Best rounding
					base_stock_levels_round = "round-up";
					base_stock_levels_shortfall_round = "round-up";
				}
			}
			else {
				throw DynaPlex::Error("asml_assembly_nonstationary: no setting found for the given number of nodes");
			}
		}
	}
	else {
		throw DynaPlex::Error("asml_assembly_nonstationary: no setting found for the given number of nodes");
	}


	noiseLength = noiseProb_vec.size();
	const auto base_stock_prec_start = std::chrono::steady_clock::now();

	for (int64_t j = 0; j < NumberofNodes; j++) {
		// Node itself in ESuccessors
		ESuccessors_2vec.push_back({ j });
		ESuccessor_2vec.push_back({  });
		EPredecessor_2vec.push_back({  });
		parallelDownNodes_2vec.push_back({});
		parallelUpNodes_2vec.push_back({});
	}


	// Find the immediate predecessors and successors per node
	for (int64_t j = NumberofNodes - 1; j > 0; j--) {
		for (int64_t k = 0; k < j; k++) {
			if (BOMmatrix_2vec[k][j] > 0) {
				EPredecessor_2vec[j].push_back(k);
				ESuccessor_2vec[k].push_back(j);
			}
		}
	}

	// Determine all successors of a node
	for (int64_t j = 0; j < NumberofNodes - 1; j++) {
		int64_t k = ESuccessor_2vec[j][0];
		ESuccessors_2vec[j].push_back(k);
		while (k != NumberofNodes - 1) {
			k = ESuccessor_2vec[k][0];
			ESuccessors_2vec[j].push_back(k);
		}
	}

	// EqSerialParam is the vector consisting of the equivalent serial parameters
	// [0] is added value, [1] is echelon lead
	EqSerialParam_2vec.push_back({});
	EqSerialParam_2vec.push_back({});
	for (int64_t i = 0; i < NumberofNodes; i++) {
		double prevhold = 0.0;
		int64_t lead = 0;
		for (int64_t j : EPredecessor_2vec[i]) {
			prevhold += h_vec[j];
		}
		for (int64_t j : ESuccessors_2vec[i]) {
			lead += Leadtime_vec[j];
		}
		Echelonlead_vec.push_back(lead);
		EqSerialParam_2vec[0].push_back(h_vec[i] - prevhold);
	}

	// Periodic review with costs calculation at the end of the period, add one for echelon lead time for base-stock computation
	for (int64_t i = 0; i < NumberofNodes; i++) {
		Echelonlead_vec[i]++;
		EqSerialParam_2vec[1].push_back(Echelonlead_vec[i]);
	}

	// Consecutive nodes with equal echelon lead time are equivalent serial nodes:
	// their added-value parameters form one equivalent-serial stage.  Build that
	// stage directly, rather than repeatedly erasing pairs from EqSerialParam_2vec.
	// This supports a run of any length, e.g. nodes 2, 3, 4, and 5 all equivalent.
	vector<vector<double>> aggregatedEqSerialParam_2vec(2);
	for (int64_t firstNodeInRun = 0; firstNodeInRun < NumberofNodes;) {
		int64_t lastNodeInRun = firstNodeInRun;
		double aggregatedAddedValue = EqSerialParam_2vec.at(0).at(firstNodeInRun);

		// Extend this run while the next serial node has the same echelon lead.
		while (lastNodeInRun + 1 < NumberofNodes
			&& Echelonlead_vec.at(lastNodeInRun) == Echelonlead_vec.at(lastNodeInRun + 1)) {
			// SameSerial_vec stores an edge: i means that original nodes i and i+1
			// belong to the same equivalent-serial run. Other routines use these
			// edges to duplicate demand distributions and base-stock levels.
			SameSerial_vec.push_back(lastNodeInRun);
			++lastNodeInRun;
			aggregatedAddedValue += EqSerialParam_2vec.at(0).at(lastNodeInRun);
		}

		// One compressed parameter represents the whole run. All nodes in the run
		// have the same echelon lead time, so retain the lead time of its first node.
		aggregatedEqSerialParam_2vec.at(0).push_back(aggregatedAddedValue);
		aggregatedEqSerialParam_2vec.at(1).push_back(
			Echelonlead_vec.at(firstNodeInRun));
		firstNodeInRun = lastNodeInRun + 1;
	}
	EqSerialParam_2vec = std::move(aggregatedEqSerialParam_2vec);

	// Vector consisting of the equivalent serial network 
	// how many periods we need to consider of i-1 if they are not predecessors (parallel network)
	// the balance inventory equations; is this echelon lead time somewhere in the middle of production of i-1 in parallel network
	std::vector<int64_t> _SerialEq(NumberofNodes, 0);

	// Assume most downstream node has no other balance equation in base-stock
	for (int64_t j = 0; j < NumberofNodes - 1; j++) {
		if (std::find(SameSerial_vec.begin(), SameSerial_vec.end(), j) != SameSerial_vec.end()) {
			_SerialEq[j + 1] += Leadtime_vec[j];
		}
		else {
			_SerialEq[j + 1] += Leadtime_vec[j] - (Echelonlead_vec[j] - Echelonlead_vec[j + 1]);
		}
	}
	SerialEq_vec = _SerialEq;


	// Determine the number of parallel nodes, based on the definition in our paper
	for (int64_t j = 0; j < NumberofNodes - 2; j++) {
		for (int64_t k = j + 1; k < NumberofNodes - 1; k++) {
			if ((Echelonlead_vec[j] - Echelonlead_vec[k] < Leadtime_vec[j]) && (0 <= Echelonlead_vec[j] - Echelonlead_vec[k])) {
				parallelDownNodes_2vec.at(j).push_back(k);
				parallelUpNodes_2vec.at(k).push_back(j);
			}
		}
	}
	// Find the nodes to balance with for a node are downstream but lower lead time that we do not balance yet and do not have predecessors
	// These nodes, you want to balance with as well. In 10 nodes case, this is node 3 for node 9 and 6 for example
	for (int64_t n = NumberofNodes - 1; n >= 0; n--) {
		if (EPredecessor_2vec.at(n).size() == 0) {
			for (int64_t j = n - 1; j >= 0; j--) {
				if (Echelonlead_vec.at(n) <= Echelonlead_vec.at(j) - Leadtime_vec.at(j)) {
					parallelDownNodes_2vec.at(j).push_back(n);
				}
			}
		}
	}

	for (int64_t n = 0; n < NumberofNodes; n++) {
		nonBottleneckNode_vec.push_back(true);
		for (int64_t down : parallelDownNodes_2vec.at(n)) {
			if (CapacityofNodes_vec.at(down) > CapacityofNodes_vec.at(n)) {
				nonBottleneckNode_vec.at(n) = false;
				break;
			}
		}
		for (int64_t up : parallelUpNodes_2vec.at(n)) {
			if (CapacityofNodes_vec.at(up) > CapacityofNodes_vec.at(n)) {
				nonBottleneckNode_vec.at(n) = false;
				break;
			}
		}
	}

	vector<vector<vector<vector<double>>>> demLeadTimeDistr_DT;
	demLeadTimeDistr_DT = getDemandLeadtimeDist(moveProb_2vec, noiseProb_vec, startNoise, maxTree, startMeanDem, Echelonlead_vec, SameSerial_vec);

	vector<vector<double>> fractiles;
	fractiles = GetFractileSS(h_vec, EqSerialParam_2vec, SameSerial_vec, p, NumberofNodes);

	const auto base_stock_prec_end = std::chrono::steady_clock::now();
	const auto base_stock_pred_time = std::chrono::duration<double, std::milli>(base_stock_prec_end - base_stock_prec_start).count();


	// Compute the base-stock levels
	const auto base_stock_level_start = std::chrono::steady_clock::now();
	base_stock_levels = GetBaseStockLevelsDT(h_vec, Echelonlead_vec, EqSerialParam_2vec, SameSerial_vec, false, {}, base_stock_levels_round, maxTree, startMeanDem, demLeadTimeDistr_DT, fractiles, noiseProb_vec, startNoise);
	const auto base_stock_level_end = std::chrono::steady_clock::now();
	const auto base_stock_level_time = std::chrono::duration<double, std::milli>(base_stock_level_end - base_stock_level_start).count();

	const auto total_base_stock_time_no_shortfall = (base_stock_pred_time + base_stock_level_time)/1000;
	std::cout << "Base-stock computation time (no shortfall): " << total_base_stock_time_no_shortfall << " s" << std::endl;

	const auto base_stock_level_shortfall_start = std::chrono::steady_clock::now();
	// Read and validate the input file.  It contains, for each vertical
	// position, NumberofNodes rows for the DT policy followed by NumberofNodes
	// rows for the Huh et al. single-policy shortfall distribution.  Stationary
	// demand therefore needs 2 * NumberofNodes rows; non-stationary demand needs
	// 2 * NumberofNodes rows per vertical position used by GetBaseStockLevelsDT.
	const int64_t numberOfVerticalPositions = stationary
		? 1
		: maxTree - startMeanDem + 1;
	const int64_t expectedShortfallRows = NumberofNodes * numberOfVerticalPositions;
	const string shortfallFilePath = path + filename_shortfall + ".csv";
	const vector<vector<double>> shortfallRows =
		readAndValidateShortfallFile(shortfallFilePath, expectedShortfallRows);

	vector<vector<vector<double>>> shortfalls_single_3vec;
	shortfalls_single_3vec.reserve(numberOfVerticalPositions);
	for (int64_t verticalPosition = 0; verticalPosition < numberOfVerticalPositions; ++verticalPosition) {
		shortfalls_single_3vec.push_back({});
		shortfalls_single_3vec.back().reserve(NumberofNodes);

		const int64_t firstRow = verticalPosition * NumberofNodes;

		for (int64_t node = 0; node < NumberofNodes; ++node) {
			shortfalls_single_3vec.back().push_back(
				shortfallRows.at(firstRow + node));
		}
	}

	base_stock_levels_shortfall = GetBaseStockLevelsDT(h_vec, Echelonlead_vec, EqSerialParam_2vec, SameSerial_vec, true, shortfalls_single_3vec, base_stock_levels_shortfall_round, maxTree, startMeanDem, demLeadTimeDistr_DT, fractiles, noiseProb_vec, startNoise);
	const auto base_stock_level_shortfall_end = std::chrono::steady_clock::now();
	const auto base_stock_level_shortfall_time = std::chrono::duration<double, std::milli>(base_stock_level_shortfall_end - base_stock_level_shortfall_start).count();

	const auto total_base_stock_time_shortfall = (base_stock_pred_time + base_stock_level_shortfall_time)/1000;
	std::cout << "Base-stock computation time (shortfall): " << total_base_stock_time_shortfall << " s" << std::endl;

	// Non base-stock related computations
	vector<vector<double>> avgdem_lt_vec;
	avgdem_lt_vec = GetAvgDemand(moveProb_2vec, noiseProb_vec, startNoise, maxTree, startMeanDem, Echelonlead_vec, demLeadTimeDistr_DT);


	// Make event probability which include both the prob of upwards movement and the noise
	for (int64_t i = 0; i < moveProb_2vec.size(); i++) {
		vector<double> demandProbsTr_vec2;
		for (int64_t k = 0; k < moveProb_2vec.at(i).size(); k++) {
			for (int64_t j = 0; j < noiseProb_vec.size(); j++) {
				demandProbsTr_vec2.push_back(moveProb_2vec.at(i).at(k) * noiseProb_vec[j]);
			}
		}
		demandProbsTr_2vec.push_back(std::move(demandProbsTr_vec2));
	}

	for (int64_t j = 0; j < moveProb_2vec.size(); j++) {
		vector<double> demProbs_tr_vec2;
		// For continuation of the trajectory
		for (int64_t i = 0; i < demandProbsTr_2vec.at(j).size(); i++) {
			demProbs_tr_vec2.push_back(discount_factor_geom * demandProbsTr_2vec.at(j).at(i));
		}
		// For reaching the final state and the corresponding demand for that period
		for (int64_t i = 0; i < demandProbsTr_2vec.at(j).size(); i++) {
			demProbs_tr_vec2.push_back((1 - discount_factor_geom) * demandProbsTr_2vec.at(j).at(i));
		}
		demProbs_tr_2vec.push_back(std::move(demProbs_tr_vec2));
	}

	// to flatten the state-space. 
	frontposition_vec.reserve(NumberofNodes);
	backposition_vec.reserve(NumberofNodes);
	frontposition_vec.push_back(0);
	backposition_vec.push_back(Leadtime_vec[0]);
	for (int64_t i = 1; i < NumberofNodes; i++) {
		frontposition_vec.push_back(backposition_vec[i - 1] + 1);
		backposition_vec.push_back(frontposition_vec[i] + Leadtime_vec[i]);
	}
	downstreamposition = frontposition_vec[NumberofNodes - 1];

	// Store the parameters in an object and return them
	const paramSystemInstance instance(
		Leadtime_vec, h_vec, CapacityofNodes_vec, moveProb_2vec, noiseProb_vec,
		startNoise, maxTree, startMeanDem, BOMmatrix_2vec, lostsalespenalty, MaxBacklog, minimumorder, noiseLength, MaxInvatNodes_vec,
		ESuccessors_2vec, ESuccessor_2vec, EPredecessor_2vec, EqSerialParam_2vec,
		Echelonlead_vec, SerialEq_vec, SameSerial_vec, base_stock_levels, base_stock_levels_shortfall, 
		frontposition_vec, backposition_vec, downstreamposition, parallelDownNodes_2vec, parallelUpNodes_2vec, demProbs_tr_2vec, avgdem_lt_vec, nonBottleneckNode_vec);

	return instance;
}


vector<vector<vector<vector<double>>>> preComputerNS::getDemandLeadtimeDist(const vector<vector<double>> moveProb_2vec, const vector<double> noiseProb_vec, const int64_t startNoise, const int64_t maxTree, const int64_t startMeanDem,
	const vector<int64_t> Echelonlead_vec, const vector<int64_t> SameSerial_vec) const
{
	// first vector of vertical pos in demand tree, second vector node, then vector of vector of demand [cumulative demand, per period demand, probability] 
	vector<vector<vector<vector<double>>>> cumleadtimedemand;

	int64_t count = 0;

	vector<vector<double> > demandlist_old;
	vector<vector<double> > demandlistleadtime;
	vector<vector<double> > demandlist_new;
	vector<vector<double> > demandlist_new2;

	for (int64_t o = 0; o < moveProb_2vec.size(); o++) {
		cumleadtimedemand.push_back({});
		cumleadtimedemand[o].reserve(NumberofNodes);
		int64_t currentnode = NumberofNodes - 1;

		// Assume lead time is at least one period
		for (int64_t i = 1; i <= Echelonlead_vec[0]; i++)
		{
			// For the cumulative distribution; we need to keep track of
			// [cumulative demand, per period demand, probability] 
			// Because even if similar cumulative demand we can have different pdf demand hence different futures of the demand tree

			if (i == 1)
			{
				for (size_t h = 0; h < moveProb_2vec.at(o).size(); h++)
				{
					std::vector<double> demandlisti;
					if (startMeanDem + h + o == maxTree) {

						demandlisti.push_back(startMeanDem + h + o);
						demandlisti.push_back(startMeanDem + h + o);
						demandlisti.push_back(0);
						for (size_t l = h; l < moveProb_2vec.at(o).size(); l++) {
							demandlisti[2] += moveProb_2vec.at(o).at(l);
						}
						demandlist_new2.push_back(demandlisti);
						break;
					}
					else {
						demandlisti.push_back(startMeanDem + h + o);
						demandlisti.push_back(startMeanDem + h + o);
						demandlisti.push_back(moveProb_2vec.at(o).at(h));
						demandlist_new2.push_back(demandlisti);
					}
				}
			}
			else
			{
				for (int64_t k = 0; k < demandlist_old.size(); k++)
				{
					for (int64_t h = 0; h < moveProb_2vec.at(o).size(); h++)
					{
						// Here check if pdf  +  h = maxtree
						// Then no increase with probability DemandProbs[h:]!!!
						// otherwise have the increase

						vector<double> demandlisti;
						if (demandlist_old[k][1] + h == maxTree) {
							demandlisti.push_back(demandlist_old[k][0] + demandlist_old[k][1] + h);
							demandlisti.push_back(demandlist_old[k][1] + h);
							//demandlisti.push_back(demandlist_old[k][2]);
							double probs = 0.0;
							for (size_t l = h; l < moveProb_2vec.at(o).size(); l++) {
								probs += moveProb_2vec.at(o)[l];
							}
							demandlisti.push_back(demandlist_old[k][2] * probs);

							demandlist_new2.push_back(demandlisti);
							break;
						}
						else {
							demandlisti.push_back(demandlist_old[k][0] + demandlist_old[k][1] + h);
							demandlisti.push_back(demandlist_old[k][1] + h);
							demandlisti.push_back(demandlist_old[k][2] * moveProb_2vec.at(o)[h]);
							demandlist_new2.push_back(demandlisti);
						}
					}
				}
			}

			// Adding noise in each time period
			for (int64_t j = 0; j < demandlist_new2.size(); j++) {
				for (int64_t h = 0; h < noiseProb_vec.size(); h++) {
					vector<double> demandlisti;
					demandlisti.push_back(demandlist_new2[j][0] + startNoise + h);
					demandlisti.push_back(demandlist_new2[j][1]);
					demandlisti.push_back(demandlist_new2[j][2] * noiseProb_vec[h]);
					demandlist_new.push_back(demandlisti);
				}
			}

			// Check similarity
			for (int64_t n = 0; n < demandlist_new.size() - 1; n++) {
				for (int64_t m = n + 1; m < demandlist_new.size(); m++) {
					if ((demandlist_new[n][0] == demandlist_new[m][0]) && (demandlist_new[n][1] == demandlist_new[m][1]))
					{
						demandlist_new[n][2] += demandlist_new[m][2];
						demandlist_new[m].clear();
						demandlist_new.erase(std::find(demandlist_new.begin(), demandlist_new.end(), demandlist_new[m]));
					}
				}
			}
			demandlist_new2.clear();

			// update demandlists
			demandlist_old.clear();
			for (int64_t u = 0; u < demandlist_new.size(); u++)
			{
				demandlist_old.push_back(demandlist_new[u]);
			}
			demandlist_new.clear();


			// Need to check whether this time step corresponds to the lead time of an echelon
			if (std::find(Echelonlead_vec.begin(), Echelonlead_vec.end(), i) != Echelonlead_vec.end())
			{
				// check if this node is an equivalent one to another one
				// we count down from N - 1 to 0, hence currentnode - 1 should be in SameSerial for currentnode to be an equivalent node
				vector<vector<double > > row;
				cumleadtimedemand[o].push_back(row);
				for (int64_t x = 0; x < demandlist_old.size(); x++) {
					cumleadtimedemand[o][count].push_back(demandlist_old[x]);
				}
				count++;

				while (std::find(SameSerial_vec.begin(), SameSerial_vec.end(), currentnode - 1) != SameSerial_vec.end())
				{
					vector<vector<double > > row;
					cumleadtimedemand[o].push_back(row);
					for (int64_t x = 0; x < demandlist_old.size(); x++) {
						cumleadtimedemand[o][count].push_back(demandlist_old[x]);
					}
					currentnode--;
					count++;
				}
				currentnode--;

			}
		}
		count = 0;

		// cumleadtimedemand[0] is for most downstream node / echelon, [NumberofNodes - 1] is for most upstream node

		// add all probabilities with same cumulative demand
		for (int64_t i = 0; i < NumberofNodes; i++) {
			for (int64_t j = 0; j < cumleadtimedemand[o][i].size() - 1; j++) {
				for (int64_t u = j + 1; u < cumleadtimedemand[o][i].size(); u++) {
					if (cumleadtimedemand[o][i][j][0] == cumleadtimedemand[o][i][u][0]) {
						cumleadtimedemand[o][i][j][2] += cumleadtimedemand[o][i][u][2];
						cumleadtimedemand[o][i][u].clear();
						cumleadtimedemand[o][i].erase(std::find(cumleadtimedemand[o][i].begin(), cumleadtimedemand[o][i].end(), cumleadtimedemand[o][i][u]));
					}
				}
			}
		}

		demandlist_old.clear();
		demandlistleadtime.clear();
		demandlist_new.clear();
		demandlist_new2.clear();

	}

	return cumleadtimedemand;
}


vector<vector<double>> preComputerNS::GetFractileSS(const vector<double> h_vec, const vector<vector<double>> EqSerialParam_2vec, const vector<int64_t> SameSerial_vec, const double p, const int64_t NumberofNodes) const
{
	vector<vector<double>> fractiles;
	fractiles.push_back({});			// Upper bound
	fractiles.push_back({});			// Lower bound

	// sameAsNext[i] means that original serial nodes i and i+1 were compressed
	// into the same equivalent-serial stage.
	vector<bool> sameAsNext(NumberofNodes - 1, false);
	for (const int64_t node : SameSerial_vec) {
		if (node >= 0 && node + 1 < NumberofNodes) {
			sameAsNext.at(node) = true;
		}
	}

	// The parameters are compressed by equivalent-serial run, whereas the
	// returned fractiles must have one entry for every original node.
	int64_t firstNodeInRun = 0;
	double holdingCostsUpstream = 0.0;
	for (int64_t equivalentStage = 0;
		equivalentStage < static_cast<int64_t>(EqSerialParam_2vec.at(0).size());
		++equivalentStage) {
		const double upperFractile = (p + holdingCostsUpstream)
			/ (p + h_vec.at(NumberofNodes - 1));
		const double lowerFractile = (p + holdingCostsUpstream)
			/ (p + holdingCostsUpstream + EqSerialParam_2vec.at(0).at(equivalentStage));

		// Duplicate the stage's fractiles for every node in this run. This works
		// for one node, a pair, or an arbitrary number of equivalent serial nodes.
		while (true) {
			fractiles.at(0).push_back(upperFractile);
			fractiles.at(1).push_back(lowerFractile);

			if (firstNodeInRun + 1 >= NumberofNodes
				|| !sameAsNext.at(firstNodeInRun)) {
				break;
			}
			++firstNodeInRun;
		}

		++firstNodeInRun;
		holdingCostsUpstream += EqSerialParam_2vec.at(0).at(equivalentStage);
	}

	return fractiles;
}

vector<vector<int64_t>> preComputerNS::GetBaseStockLevelsDT(const vector<double> h_vec, const vector<int64_t> Echelonlead_vec, const vector<vector<double>> EqSerialParam_2vec, const vector<int64_t> SameSerial_vec, bool shortfall, vector<vector<vector<double>>> shortfalls_3vec, string rounding_BS_level,
	const int64_t maxTree, const int64_t startMeanDem, const vector<vector<vector<vector<double>>>> demLeadTimeDistr, const vector<vector<double>> fractiles, const vector<double> noiseProb_vec, const int64_t startNoise) const {
	vector<vector<int64_t>> basestocklevels;
	for (int64_t o = 0; o <= maxTree - startMeanDem; o++) {
		basestocklevels.push_back({});
		for (int64_t j = 0; j < NumberofNodes; j++) {
			// Required for computing the fractiles in Shang and Song heuristic
			int64_t basestockupperbound = 0;
			int64_t basestocklowerbound = 0;


			std::vector<double> probabilitieslt;
			auto DemOverLeadtime = cust_dist.GetZeroDist();
			for (size_t u = 0; u < demLeadTimeDistr[o][NumberofNodes - 1 - j].size(); u++) {
				probabilitieslt.push_back(demLeadTimeDistr[o][NumberofNodes - 1 - j][u][2]);
			}
			auto probLTDist = cust_dist.GetCustomDist(probabilitieslt);
			DemOverLeadtime = cust_dist.Add(probLTDist);
			if (shortfall) {
				auto shortfallprob_dist = cust_dist.GetCustomDist(shortfalls_3vec[o][j]);
				DemOverLeadtime = DemOverLeadtime.Add(shortfallprob_dist);

			}

			//// Fractiles[0] is for node 0, demLeadTimeDistr[0] is for node N-1, therefore, the changes with j
			//std::cout << (p + holdingcostsupper) / (p + h_vec[NumberofNodes - 1]) << std::endl;
			//std::cout << (p + holdingcostsupper) / (p + holdingcostsupper + EqSerialParam_2vec[0][j]) << std::endl;
			basestocklowerbound = DemOverLeadtime.Fractile(fractiles.at(0).at(j)) + demLeadTimeDistr[o][NumberofNodes - 1 - j][0][0];
			basestockupperbound = DemOverLeadtime.Fractile(fractiles.at(1).at(j)) + demLeadTimeDistr[o][NumberofNodes - 1 - j][0][0];

			// Rounding possibilities
			if (rounding_BS_level == "round-up") {
				auto k = 0.5 * (basestocklowerbound + basestockupperbound);
				if (std::floor(k) == k) {
					basestocklevels[o].push_back(0.5 * (basestocklowerbound + basestockupperbound));
				}
				else {
					// round up
					basestocklevels[o].push_back(0.5 * (basestocklowerbound + basestockupperbound + 1));
				}
			}
			else if (rounding_BS_level == "round-down") {
				basestocklevels[o].push_back(0.5 * (basestocklowerbound + basestockupperbound));
			}
			else if (rounding_BS_level == "minimum") {
				basestocklevels[o].push_back(basestocklowerbound);
			}
			// maximum
			else {
				basestocklevels[o].push_back(basestockupperbound);
			}
		}
	}


	return basestocklevels;
}

vector<vector<double>> preComputerNS::GetAvgDemand(const vector<vector<double>> moveProb_2vec, const vector<double> noiseProb_vec, const int64_t startNoise, const int64_t maxTree, const int64_t startMeanDem, const vector<int64_t> Echelonlead_vec, const vector<vector<vector<vector<double>>>> demLeadTimeDistr) const
{
	vector<vector<double>> avg_dem_vec;
	vector<vector<double>> avg_dem_vec2;

	if (stationary) {
		avg_dem_vec.push_back({});
		avg_dem_vec2.push_back({});
		double mean_dem = 0;
		for (int64_t i = 0; i < noiseProb_vec.size(); i++) {
			mean_dem += noiseProb_vec.at(i) * (i + startNoise + startMeanDem);
		}

		for (int64_t i = 1; i <= Echelonlead_vec.at(0) + Echelonlead_vec[0] - Echelonlead_vec[NumberofNodes - 1]; i++) {
			avg_dem_vec.at(0).push_back(mean_dem * i);
		}
	}
	else {

		double mean_dem = 0;
		vector<double> probabilities_mean(maxTree - startMeanDem + 1, 0.0);
		vector<double> probabilities_mean_new(maxTree - startMeanDem + 1, 0.0);

		for (int64_t o = 0; o <= maxTree - startMeanDem; o++) {
			avg_dem_vec.push_back({});
			probabilities_mean.clear();
			for (int64_t t = 0; t <= maxTree - startMeanDem; t++) {
				probabilities_mean.push_back(0);
			}
			probabilities_mean.at(o) = 1;
			mean_dem = 0;

			for (int64_t i = 1; i <= Echelonlead_vec.at(0) + Echelonlead_vec[0] - Echelonlead_vec[NumberofNodes - 1]; i++) {
				probabilities_mean_new.clear();
				for (int64_t t = 0; t <= maxTree - startMeanDem; t++) {
					probabilities_mean_new.push_back(0);
				}
				for (int64_t j = o; j <= maxTree - startMeanDem; j++) {
					probabilities_mean_new.at(j) += probabilities_mean.at(j) * moveProb_2vec.at(j).at(0);
					for (int64_t k = 1; k < moveProb_2vec.at(j).size(); k++) {
						if (j + k <= maxTree - startMeanDem) {
							probabilities_mean_new.at(j + k) += probabilities_mean.at(j) * moveProb_2vec.at(j).at(k);
						}
						else {
							probabilities_mean_new.at(maxTree - startMeanDem) += probabilities_mean.at(j) * moveProb_2vec.at(j).at(k);

						}
					}
				}
				probabilities_mean.clear();
				for (int64_t u = 0; u < probabilities_mean_new.size(); u++) {
					probabilities_mean.push_back(probabilities_mean_new.at(u));
					mean_dem += probabilities_mean_new.at(u) * (startMeanDem + u);
				}

				avg_dem_vec.at(o).push_back(mean_dem);
			}
		}
	}

	return avg_dem_vec;
}