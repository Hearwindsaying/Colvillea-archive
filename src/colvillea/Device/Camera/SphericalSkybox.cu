#include <optix_world.h>
#include <optix_device.h>
#include <optixu_math_namespace.h>

#include "../../Device/Toolkit/CommonStructs.h"
#include "../../Device/Toolkit/Utility.h"
#include "../../Device/Light/LightUtil.h"

using namespace optix;

//////////////////////////////////////////////////////////////////////////
//Forward declarations:
rtDeclareVariable(CommonStructs::PerRayData_radiance,  prdRadiance, rtPayload, );
rtDeclareVariable(optix::Ray,                          ray,         rtCurrentRay, );

#ifndef TWRT_DELCARE_LIGHTBUFFER
#define TWRT_DELCARE_LIGHTBUFFER
rtDeclareVariable(CommonStructs::LightBuffers, sysLightBuffers, , );
#endif

//////////////////////////////////////////////////////////////////////////
//Program definitions:

RT_PROGRAM void Miss_Default()
{
	prdRadiance.radiance = (sysLightBuffers.hdriLight.hdriEnvmap == RT_TEXTURE_ID_NULL ? make_float4(0.f) : TwUtil::Le_HDRILight(ray.direction, sysLightBuffers.hdriLight.hdriEnvmap, sysLightBuffers.hdriLight.worldToLight));
}