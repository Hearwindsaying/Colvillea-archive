#include <optix_world.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"
#include "LightUtil.h"

using namespace optix;
using namespace TwUtil;

//////////////////////////////////////////////////////////////////////////
//Forward declarations:

//system variables:->Context
rtBuffer<CommonStructs::PointLight> pointLightBuffer;

//////////////////////////////////////////////////////////////////////////
//Pointlight Sample_Ld function:
RT_CALLABLE_PROGRAM float4 Sample_Ld_Point(const float3 &point, const float & rayEpsilon, float3 & outwi, float & outpdf, float2 lightSample, uint lightBufferIndex, Ray & outShadowRay)
{
	float3 &lightPos = pointLightBuffer[lightBufferIndex].lightPos;
	
	outwi = safe_normalize(lightPos - point);
	outpdf = 1.f;

	float distanceSqr = sqr_length(lightPos - point);
	outShadowRay = MakeShadowRay(point, rayEpsilon, lightPos, 1e-3f);

	return pointLightBuffer[lightBufferIndex].intensity / distanceSqr;
}

//Useless program:
RT_CALLABLE_PROGRAM float LightPdf_Point(const float3 & p, const float3 & wi, const int lightId, Ray &shadowRay)
{
    return 0.f;
}