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
rtDeclareVariable(optix::Ray,           ray,         rtCurrentRay, );

rtDeclareVariable(int,                  hdriEnvmap, ,) = RT_TEXTURE_ID_NULL;
rtDeclareVariable(CommonStructs::HDRILight,			hdriLight, , );

//////////////////////////////////////////////////////////////////////////
//Program definitions:

RT_PROGRAM void Miss_Default()
{
	prdRadiance.radiance = (hdriEnvmap == RT_TEXTURE_ID_NULL ? make_float4(0.f) : TwUtil::Le_HDRILight(ray.direction, hdriEnvmap, hdriLight.worldToLight));
}