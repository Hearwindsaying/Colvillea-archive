#pragma once

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include <map>
#include <memory>

#include "colvillea/Application/TWAssert.h"
#include "colvillea/Application/GlobalDefs.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"

/**
 * @brief Integrator class serves as a helper class
 * to do some initial work on host side, such as loading
 * programs, creating Material node (integrator) for 
 * TriangleMesh.
 */
class Integrator
{
public:
    /**
     * @brief Constructor of Integrator.
     * 
     * @note Integrator::initializeIntegratorMaterialNode()
     * should be called shortly after constructing the Integrator.
     * For this reason, you should not call the constructor directly
     * and use factory method Integrator::createIntegrator()
     */
	Integrator(const optix::Context context, const std::map<std::string, optix::Program> &programsMap, const std::string &integratorClassName):m_context(context), m_programsMap(programsMap)
	{
		std::cout << "[Info] Derived class name from Integrator is: " << integratorClassName << std::endl;

		/* Load closest hit program */

        // todo:it's not possible construct Integrator object!
		auto programItr = this->m_programsMap.find("ClosestHit_" + integratorClassName);
		TW_ASSERT(programItr != this->m_programsMap.end());
		this->m_closestHitProgram = programItr->second;

		/* Load any hit program for shadow ray, same for all shapes. */
		programItr = this->m_programsMap.find("ClosestHit_ShadowRay_GeometryTriangles");
		TW_ASSERT(programItr != this->m_programsMap.end());
		this->m_closestHit_ShadowRayProgram = programItr->second;
	}

	/**
	 * @brief Create material node for the integrator which could
	 * be used for creating Shape later. This function should be
	 * called shortly after invoking constructor.
	 * 
	 * @note Remember that the integrator is represented as Material
	 * node in SceneGraph.
	 */
    virtual optix::Material initializeIntegratorMaterialNode()
    {
        TW_ASSERT(this->m_closestHitProgram && this->m_closestHit_ShadowRayProgram);

        this->m_integratorMaterial = this->m_context->createMaterial();
        this->m_integratorMaterial->setClosestHitProgram(toUnderlyingValue(CommonStructs::RayType::Radiance), this->m_closestHitProgram);
        //this->m_integratorMaterial->setAnyHitProgram(toUnderlyingValue(CommonStructs::RayType::Shadow), this->m_anyHitShadowRayProgram);
        this->m_integratorMaterial->setClosestHitProgram(toUnderlyingValue(CommonStructs::RayType::Shadow), this->m_closestHit_ShadowRayProgram);

        return this->m_integratorMaterial;
    }

	/**
	 * @brief Getter for |m_integratorMaterial|. Note that this could
	 * be called only after createIntegratorMaterialNode().
	 * 
	 * @see createIntegratorMaterialNode()
	 */
    optix::Material getIntegratorMaterial() const
    {
        return this->m_integratorMaterial;
    }

protected:
    optix::Context m_context;
    const std::map<std::string, optix::Program> &m_programsMap;

	optix::Program m_closestHitProgram;
	//optix::Program m_anyHitShadowRayProgram;
    optix::Program m_closestHit_ShadowRayProgram;
private:
	optix::Material m_integratorMaterial; //todo:perhaps not needed?
};