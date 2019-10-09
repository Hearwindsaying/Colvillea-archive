#pragma once
#include <optix_world.h>
#include <optix_device.h>
#include <optixu_math_namespace.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"
#include "../Toolkit/NvRandom.h"
#include "Fresnel.h"

using namespace optix;
using namespace CommonStructs;

/*Smooth reflection material representing an ideal Mirror(Fresnel=1),
 *which is quite different from SmoothConductor(not supported yet)
 *using FresnelConductor.
 *
 *Note that this material is one-sided, reversing the geometric normal
 *if necessary before which normal-mapping or shading normal modification
 *performed.*/

//////////////////////////////////////////////////////////////////////////
//
RT_CALLABLE_PROGRAM float4 Eval_f_SmoothMirror(const float3 & wo_World, const float3 & wi_World, const CommonStructs::ShaderParams & shaderParams)
{
	return make_float4(0.f);
}

RT_CALLABLE_PROGRAM float Pdf_SmoothMirror(const float3 & wo_World, const float3 & wi_World, const CommonStructs::ShaderParams & shaderParams)
{
	return 0.f;
}

RT_CALLABLE_PROGRAM float4 Sample_f_SmoothMirror(const float3 &wo_World, float3 & outwi_World, float2 & urand, float & outPdf, float bsdfChoiceRand, const CommonStructs::ShaderParams & shaderParams)
{
	float3 wo_Local = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
	float3 outwi_Local = make_float3(0.f);

	/*Inner sample_f function goes here:*/

	/*one-sided BSDF detection*/
	if (BSDFMath::CosTheta(wo_Local) <= 0.f)
		return make_float4(0.f);

	// Compute perfect specular reflection direction
	outwi_Local = make_float3(-wo_Local.x, -wo_Local.y, wo_Local.z);
	outPdf = 1.f;
	outwi_World = TwUtil::BSDFMath::LocalToWorld(outwi_Local, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

	return make_float4(1.f /** shaderParams.Reflectance*/ / BSDFMath::AbsCosTheta(outwi_Local));
}