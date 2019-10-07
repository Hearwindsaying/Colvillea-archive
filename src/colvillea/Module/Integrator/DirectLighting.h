#pragma once
//#include "../../Application/Application.h"

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "Integrator.h"

class DirectLighting : public Integrator
{
public:
	DirectLighting(const optix::Context context, const std::map<std::string, optix::Program> &programsMap) :Integrator(context, programsMap, "DirectLighting")
	{

	}


};