#include "PathTracing.h"

#include "../../Application/TWAssert.h"
#include "../../Device/Toolkit/CommonStructs.h"
#include "../../Application/GlobalDefs.h"

optix::Material PathTracing::createIntegratorMaterialNode()
{
	auto material = Integrator::createIntegratorMaterialNode();

	/* Additional setup for PTRay program. */
	material->setClosestHitProgram(toUnderlyingValue(CommonStructs::RayType::Detection), this->m_closestHitPTRayProgram);

	return material;
}
