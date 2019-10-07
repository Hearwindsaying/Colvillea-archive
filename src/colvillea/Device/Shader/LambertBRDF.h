#pragma once
#ifndef COLVILLEA_DEVICE_SHADER_LAMBERTBRDF_H_
#define COLVILLEA_DEVICE_SHADER_LAMBERTBRDF_H_

#include <optix_world.h>
#include <optix_device.h>
#include <optixu_math_namespace.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"
#include "../Toolkit/NvRandom.h"



using namespace optix; 
using namespace TwUtil;

//////////////////////////////////////////////////////////////////////////
//Lambert BRDF:


/*Modified on 7/14/2019: we decided to use one-sided BSDF in our shader system.
 *Two-sided BSDF is not supported yet while the BSDF invovling transmitted BTDF 
 *is undoubtedly two-sided.*/

//////////////////////////////////////////////////////////////////////////
//[BRDF]Lambert InnerPdf function:
static __device__ __inline__ float4 Lambert_InnerEval_f(const float3 & wo_Local, const float3 & wi_Local, const CommonStructs::ShaderParams & shaderParams)
{
	/*one-sided BSDF detection*/
	if (BSDFMath::CosTheta(wo_Local) <= 0.f || BSDFMath::CosTheta(wi_Local) <= 0.f)
		return make_float4(0.f);

	if (shaderParams.ReflectanceID != RT_TEXTURE_ID_NULL)
	{
		return rtTex2D<float4>(shaderParams.ReflectanceID, shaderParams.dgShading.uv.x, shaderParams.dgShading.uv.y) * M_1_PIf;
	}
	else//InnerEval_f_Lambertian BxDF
		return shaderParams.Reflectance * M_1_PIf;
}

static __device__ __inline__ float Lambert_InnerPdf(const float3 & wo_Local, const float3 & wi_Local, const CommonStructs::ShaderParams & shaderParams)
{
	/*one-sided BSDF detection*/
	if (BSDFMath::CosTheta(wo_Local) <= 0.f || BSDFMath::CosTheta(wi_Local) <= 0.f)
		return 0.f;

	/*square to cosineHemisphere pdf evaluation*/
	return BSDFMath::CosTheta(wi_Local) * M_1_PIf;
}
//[BRDF]Lambert InnerSample_f function:
static __device__ __inline__ float4 Lambert_InnerSample_f(const float3 &wo_Local, float3 & outwi_Local, float2 & urand, float & outPdf, const CommonStructs::ShaderParams & shaderParams)
{
	/*one-sided BSDF detection*/
	if (BSDFMath::CosTheta(wo_Local) <= 0.f)
		/*could be omitted here due to the usage of Inner_ functions*/
		return make_float4(0.f);

	outwi_Local = TwUtil::MonteCarlo::CosineSampleHemisphere(urand);
	if (wo_Local.z < 0.) outwi_Local.z *= -1.f;
	outPdf = Lambert_InnerPdf(wo_Local, outwi_Local, shaderParams);

	return Lambert_InnerEval_f(wo_Local, outwi_Local, shaderParams);
}

#endif // COLVILLEA_DEVICE_SHADER_LAMBERTBRDF_H_