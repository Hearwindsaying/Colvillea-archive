#include "Integrator.h"
#include "colvillea/Application/TWAssert.h"
#include "colvillea/Application/GlobalDefs.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"

optix::Material Integrator::initializeIntegratorMaterialNode()
{
	TW_ASSERT(this->m_closestHitProgram && this->m_anyHitShadowRayProgram);

	this->m_integratorMaterial = this->m_context->createMaterial();
	this->m_integratorMaterial->setClosestHitProgram(toUnderlyingValue(CommonStructs::RayType::Radiance), this->m_closestHitProgram);
	this->m_integratorMaterial->setAnyHitProgram(toUnderlyingValue(CommonStructs::RayType::Shadow), this->m_anyHitShadowRayProgram);

	return this->m_integratorMaterial;
}
