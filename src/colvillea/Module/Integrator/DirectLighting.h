#pragma once
//#include "../../Application/Application.h"

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "Integrator.h"

class DirectLighting : public Integrator
{
public:
    /**
     * @brief Factory method for creating a DirectLighting instance.
     *
     * @param[in] context
     * @param[in] programsMap  map to store Programs
     */
    static std::unique_ptr<DirectLighting> createIntegrator(const optix::Context context, const std::map<std::string, optix::Program> &programsMap)
    {
        std::unique_ptr<DirectLighting> directLighting = std::make_unique<DirectLighting>(context, programsMap);
        directLighting->initializeIntegratorMaterialNode();
        return directLighting;
    }

	DirectLighting(const optix::Context context, const std::map<std::string, optix::Program> &programsMap) : Integrator(context, programsMap, "DirectLighting")
	{

	}


};