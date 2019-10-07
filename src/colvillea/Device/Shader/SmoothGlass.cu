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

namespace TwUtil
{
	/*Refract incident direction lying in local shading space into transmitted direction in local shading space*/
	static __device__ __inline__ float3 Refract(const float3 & wi, const float cosThetaT, const float eta)
	{
		/*figure out the correct etaI/etaT*/
		float etaI_etaT = (cosThetaT < 0.f) ? (1.f / eta) : eta;

		/* the refracted direction is (-eta*wi.x,-eta*wi.y,cosThetaT) 
		  -- due to the simple config in local shading space */
		return make_float3(-etaI_etaT * wi.x, -etaI_etaT * wi.y, cosThetaT);
	}
};

//////////////////////////////////////////////////////////////////////////
//
RT_CALLABLE_PROGRAM float4 SmoothGlass_Eval_f(const float3 & wo_World, const float3 & wi_World, const ShaderParams & shaderParams)
{
	return make_float4(0.f);
}

RT_CALLABLE_PROGRAM float SmoothGlass_Pdf(const float3 & wo_World, const float3 & wi_World, const ShaderParams & shaderParams)
{
	return 0.f;
}

RT_CALLABLE_PROGRAM float4 SmoothGlass_Sample_f(const float3 &wo_World, float3 & outwi_World, float2 & urand, float & outPdf, float bsdfChoiceRand, const ShaderParams & shaderParams)
{

	/*old version*/
#if 0
	float3 wo_Local = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
	float3 outwi_Local = make_float3(0.f);

	float etaA = 1.f, etaB = shaderParams.ior;
	float F = FrDiel_Evaluate(BSDFMath::CosTheta(wo_Local), etaA, etaB);//correct

	if (bsdfChoiceRand < F)
	{
		// Compute specular reflection for _FresnelSpecular_

		// Compute perfect specular reflection direction
		outwi_Local = make_float3(-wo_Local.x, -wo_Local.y, wo_Local.z);
		outPdf = F;
		outwi_World = TwUtil::BSDFMath::LocalToWorld(outwi_Local, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

		return make_float4(F /** shaderParams.Reflectance*/ / BSDFMath::AbsCosTheta(outwi_Local));
	}
	else
	{
		// Compute specular transmission for _FresnelSpecular_

		// Figure out which $\eta$ is incident and which is transmitted
		bool entering = BSDFMath::CosTheta(wo_Local) > 0;
		float etaI = entering ? etaA : etaB;
		float etaT = entering ? etaB : etaA;

		// Compute ray direction for specular transmission
		if (!TwUtil::refract(wo_Local, TwUtil::faceforward(make_float3(0, 0, 1), wo_Local), etaI / etaT, outwi_Local))
			return make_float4(0.f);

		outPdf = 1 - F;
		outwi_World = TwUtil::BSDFMath::LocalToWorld(outwi_Local, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

		//todo: when using bidirectional methods, etaI*etaI / (etaT*etaT) is not needed

		return /*T **/(etaI * etaI) / (etaT * etaT) * make_float4(1 - F) / BSDFMath::AbsCosTheta(outwi_Local);

	}
#endif
	
	/*convert direction into local space*/
	float3 wo_Local = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
	float3 outwi_Local = make_float3(0.f);

	/*compute Fresnel reflectance for dielectric*/
	float cosThetaT = 0.f;
	float F = FresnelDielectricExt(BSDFMath::CosTheta(wo_Local), cosThetaT, shaderParams.ior / 1.f);

	if (bsdfChoiceRand < F)
	{
		/*sample reflection*/
		outwi_Local = /*TwUtil::reflect(wo_Local, make_float3(0.f, 0.f, 1.f))*/
			          make_float3(-wo_Local.x, -wo_Local.y ,wo_Local.z);
		outPdf = F;

		/*todo:a eta record for RR in path tracing is necessary*/

		/*convert sampled wi direction back to world space*/
		outwi_World = TwUtil::BSDFMath::LocalToWorld(outwi_Local, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

		return make_float4(F /** shaderParams.Reflectance*/ / BSDFMath::AbsCosTheta(outwi_Local));
	}
	else
	{
		/*sample refraction*/
		outwi_Local = TwUtil::Refract(wo_Local, cosThetaT, shaderParams.ior / 1.f);
		outPdf = 1.f - F;

		/*todo:a eta record for RR in path tracing is necessary*/

		outwi_World = TwUtil::BSDFMath::LocalToWorld(outwi_Local, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

		float factor = (cosThetaT < 0.f ? 1.f / shaderParams.ior : shaderParams.ior);

		return /*T **/(factor) * (factor) * make_float4(1 - F) / BSDFMath::AbsCosTheta(outwi_Local);
	}
}