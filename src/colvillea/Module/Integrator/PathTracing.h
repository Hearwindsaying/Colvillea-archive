#pragma once
//#include "../../Application/Application.h"

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "Integrator.h"

class PathTracing : public Integrator
{
public:
	PathTracing(optix::Context context, const std::map<std::string, optix::Program> &programsMap, bool enableRoussianRoulette, int maxDepth) : Integrator(context, programsMap, "PathTracing")
	{
		TW_ASSERT(maxDepth > 0);
        this->setEnableRoussianRoulette(enableRoussianRoulette);
        this->setMaxDepth(maxDepth);

		/* Load additional closest hit for PTRay program. */

		auto programItr = this->m_programsMap.find("ClosestHit_PTRay_PathTracing");
		TW_ASSERT(programItr != this->m_programsMap.end());
		this->m_closestHitPTRayProgram = programItr->second;
	}

	optix::Material createIntegratorMaterialNode() override;

	bool getEnableRoussianRoulette() const
	{
		return this->m_enableRoussianRoulette;
	}

	void setEnableRoussianRoulette(bool enable)
	{
		static_assert(static_cast<int>(true) == 1, "True cast to int != 1");

		this->m_enableRoussianRoulette = enable;
		updateEnableRoussianRoulette();
	}

	/**
	 * @brief Getter for max depth. Only limited max depth
	 * is support for current version.
	 */
	int getMaxDepth() const
	{
		return this->m_maxDepth;
	}

	void setMaxDepth(int maxDepth)
	{
		TW_ASSERT(maxDepth > 0);
		this->m_maxDepth = maxDepth;
		updateMaxDepth();
	}

private:
	void updateEnableRoussianRoulette()
	{
        //todo:bind to program instead of context wide
		this->m_context["ptIntegrator_disableRR"]->setInt(static_cast<int>(!m_enableRoussianRoulette));
		std::cout << "[Info] " << (m_enableRoussianRoulette ? "Enable" : "Disable") << " RoussianRoulette for the path tracing integrator" << std::endl;
	}

	void updateMaxDepth()
	{
		this->m_context["ptIntegrator_maxDepth"]->setInt(this->m_maxDepth);
		std::cout << "[Info] Update maxDepth: " << this->m_maxDepth << " RoussianRoulette for the path tracing integrator" << std::endl;
	}



private:
	bool m_enableRoussianRoulette;
	int  m_maxDepth;

	optix::Program m_closestHitPTRayProgram;
};
