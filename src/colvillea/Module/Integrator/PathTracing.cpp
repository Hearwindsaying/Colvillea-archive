#include "PathTracing.h"

#include "colvillea/Application/TWAssert.h"
#include "colvillea/Application/GlobalDefs.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"


optix::Material PathTracing::initializeIntegratorMaterialNode()
{
	auto material = Integrator::initializeIntegratorMaterialNode();

	/* Additional setup for PTRay program. */
	material->setClosestHitProgram(toUnderlyingValue(CommonStructs::RayType::Detection), this->m_closestHitPTRayProgram);

	return material;
}
