#include "dynaplex/models/registrationmanager.h"
#include "dynaplex/registry.h"

namespace DynaPlex::Models {
	//forward declarations of the registration functions of MDPs:
	namespace asml_assembly_nonstationary {
		void Register(DynaPlex::Registry&);
	}
	void RegistrationManager::RegisterAll(DynaPlex::Registry& registry) {
		asml_assembly_nonstationary::Register(registry);
	}
}
