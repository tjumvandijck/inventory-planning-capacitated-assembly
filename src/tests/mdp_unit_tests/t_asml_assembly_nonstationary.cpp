#include "dynaplex/vargroup.h"
#include "dynaplex/error.h"
#include <gtest/gtest.h>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/trajectory.h"
#include "dynaplex/demonstrator.h"
#include "testutils.h" // for ExecuteTest
namespace DynaPlex::Tests {
	
	
	TEST(asml_assembly_nonstationary, mdp_config_0) {

		//Note: models/model_name/mdp_config_name is valid json config file for mdp "model_name". 
		std::string model_name = "asml_assembly_nonstationary";
		std::string config_name = "mdp_config_0.json";
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		//Opt in for testing the functionality that exposes exact event probabilities. 
		tester.TestEventProbs = true;

		tester.ExecuteTest(model_name, config_name);
	}

	TEST(asml_assembly_nonstationary, mdp_config_0_policy_config_1) {
		std::string model_name = "asml_assembly_nonstationary";
		//Note: models/model_name/mdp_config_name is valid json config file for mdp "model_name". 
		std::string mdp_config_name = "mdp_config_0.json";
		//Note: models/model_name/policy_config_name is valid json config file for a policy for "model_name".  
		std::string policy_config_name = "policy_config_1.json";
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		//Opt in for testing the functionality that exposes exact event probabilities. 
		tester.TestEventProbs = true;

		tester.ExecuteTest(model_name, mdp_config_name, policy_config_name);
	}

	TEST(asml_assembly_nonstationary, mdp_config_0_policy_config_2) {
		std::string model_name = "asml_assembly_nonstationary";
		//Note: models/model_name/mdp_config_name is valid json config file for mdp "model_name". 
		std::string mdp_config_name = "mdp_config_0.json";
		//Note: models/model_name/policy_config_name is valid json config file for a policy for "model_name".  
		std::string policy_config_name = "policy_config_2.json";
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		//Opt in for testing the functionality that exposes exact event probabilities. 
		tester.TestEventProbs = true;

		tester.ExecuteTest(model_name, mdp_config_name, policy_config_name);
	}

	TEST(asml_assembly_nonstationary, mdp_config_0_policy_config_3) {
		std::string model_name = "asml_assembly_nonstationary";
		//Note: models/model_name/mdp_config_name is valid json config file for mdp "model_name". 
		std::string mdp_config_name = "mdp_config_0.json";
		//Note: models/model_name/policy_config_name is valid json config file for a policy for "model_name".  
		std::string policy_config_name = "policy_config_3.json";
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		//Opt in for testing the functionality that exposes exact event probabilities. 
		tester.TestEventProbs = true;

		tester.ExecuteTest(model_name, mdp_config_name, policy_config_name);
	}

	TEST(asml_assembly_nonstationary, mdp_config_0_policy_config_4) {
		std::string model_name = "asml_assembly_nonstationary";
		//Note: models/model_name/mdp_config_name is valid json config file for mdp "model_name". 
		std::string mdp_config_name = "mdp_config_0.json";
		//Note: models/model_name/policy_config_name is valid json config file for a policy for "model_name".  
		std::string policy_config_name = "policy_config_4.json";
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		//Opt in for testing the functionality that exposes exact event probabilities. 
		tester.TestEventProbs = true;

		tester.ExecuteTest(model_name, mdp_config_name, policy_config_name);
	}

	TEST(asml_assembly_nonstationary, mdp_config_1_policy_config_1) {
		std::string model_name = "asml_assembly_nonstationary";
		//Note: models/model_name/mdp_config_name is valid json config file for mdp "model_name". 
		std::string mdp_config_name = "mdp_config_1.json";
		//Note: models/model_name/policy_config_name is valid json config file for a policy for "model_name".  
		std::string policy_config_name = "policy_config_1.json";
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		//Opt in for testing the functionality that exposes exact event probabilities. 
		tester.TestEventProbs = true;

		tester.ExecuteTest(model_name, mdp_config_name, policy_config_name);
	}

	TEST(asml_assembly_nonstationary, mdp_config_2_policy_config_1) {
		std::string model_name = "asml_assembly_nonstationary";
		//Note: models/model_name/mdp_config_name is valid json config file for mdp "model_name". 
		std::string mdp_config_name = "mdp_config_2.json";
		//Note: models/model_name/policy_config_name is valid json config file for a policy for "model_name".  
		std::string policy_config_name = "policy_config_1.json";
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		//Opt in for testing the functionality that exposes exact event probabilities. 
		tester.TestEventProbs = true;

		tester.ExecuteTest(model_name, mdp_config_name, policy_config_name);
	}

	TEST(asml_assembly_nonstationary, mdp_config_3_policy_config_1) {
		std::string model_name = "asml_assembly_nonstationary";
		//Note: models/model_name/mdp_config_name is valid json config file for mdp "model_name". 
		std::string mdp_config_name = "mdp_config_3.json";
		//Note: models/model_name/policy_config_name is valid json config file for a policy for "model_name".  
		std::string policy_config_name = "policy_config_1.json";
		Tester tester{};
		tester.AssertFlatFeatureAvailability = true;
		//Opt in for testing the functionality that exposes exact event probabilities. 
		tester.TestEventProbs = true;

		tester.ExecuteTest(model_name, mdp_config_name, policy_config_name);
	}

	TEST(asml_assembly_nonstationary, Basics) {
		auto& dp = DynaPlexProvider::Get();
		DynaPlex::VarGroup vars;
		vars.Add("id", "asml_assembly_nonstationary");
		vars.Add("NumberofNodes", 3);

		DynaPlex::MDP mdp;
		DynaPlex::Policy policy;

		ASSERT_NO_THROW(
			mdp = dp.GetMDP(vars);
		);
		ASSERT_NO_THROW(
			policy = mdp->GetPolicy("random");
		);

		//int64_t numEventTrajectories;
		//ASSERT_NO_THROW(
		//	numEventTrajectories = mdp->NumEventRNGs();
		//);
		//Trajectory trajectory{numEventTrajectories};

		Trajectory trajectory{};

		
		ASSERT_NO_THROW(
			mdp->InitiateState({ &trajectory,1 });
		);
		ASSERT_NO_THROW(
			trajectory.RNGProvider.SeedEventStreams(true, 123);
		);

	    int64_t max_period_count = 10;
		bool finalreached = false;
		while (trajectory.PeriodCount < max_period_count && !finalreached)
		{
			auto& cat = trajectory.Category;
			if (cat.IsAwaitEvent())
			{
				ASSERT_NO_THROW(
					mdp->IncorporateEvent({ &trajectory,1 });
				);
			}
			else if (cat.IsAwaitAction())
			{
				ASSERT_NO_THROW(
					policy->SetAction({ &trajectory,1 });
				);
				ASSERT_NO_THROW(
					mdp->IncorporateAction({ &trajectory,1 });
				);
			}
			else if (cat.IsFinal())
			{
				finalreached = true;
			}
		}			
	}
}