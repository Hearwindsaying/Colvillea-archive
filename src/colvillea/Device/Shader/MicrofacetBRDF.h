#pragma once

#ifndef COLVILLEA_DEVICE_SHADER_MICROFACETBRDF_H_
#define COLVILLEA_DEVICE_SHADER_MICROFACETBRDF_H_
#include <optix_world.h>
#include <optix_device.h>
#include <optixu_math_namespace.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"
#include "../Toolkit/NvRandom.h"

#include "Microsurface.h"
#include "Fresnel.h"

using namespace optix;
using namespace TwUtil;

//////////////////////////////////////////////////////////////////////////
//[BRDF]MicrofacetReflection BRDF Eval_f()
//todo:revise material parameters:ks kd
static __device__ __inline__ float4 MicrofacetReflection_InnerEval_f(const float3 & wo_Local, const float3 & wi_Local, const CommonStructs::ShaderParams & shaderParams, const bool isConductor)
{
	/*one-sided BSDF detection*/
	if (BSDFMath::CosTheta(wo_Local) <= 0.f || BSDFMath::CosTheta(wi_Local) <= 0.f)
		return make_float4(0.f);

	float cosThetaO = BSDFMath::AbsCosTheta(wo_Local);
	float cosThetaI = BSDFMath::AbsCosTheta(wi_Local);

	float3 wh = wi_Local + wo_Local;

	// Handle degenerate cases for microfacet reflection
	if (cosThetaI == 0 || cosThetaO == 0)
	{
		return make_float4(0.f);
	}
	if (wh.x == 0 && wh.y == 0 && wh.z == 0)
	{
		return make_float4(0.f);
	}
	wh = safe_normalize(wh);

	//note that when wh is downward,D(wh) could fail

	float4 F = isConductor ? make_float4(FrCond_Evaluate(dot(wi_Local, wh), shaderParams)) : make_float4(FrDiel_Evaluate(dot(wi_Local, wh), 1.f, shaderParams.ior));
	float4 Rs = shaderParams.Specular;
	if (shaderParams.SpecularID != RT_TEXTURE_ID_NULL)
	{
		Rs = rtTex2D<float4>(shaderParams.SpecularID, shaderParams.dgShading.uv.x, shaderParams.dgShading.uv.y);
	}

	return Rs * GGX_D(wh, shaderParams) * Smith_G_Sep(wo_Local, wi_Local, wh, shaderParams) * F / (4 * cosThetaI * cosThetaO);
}

//////////////////////////////////////////////////////////////////////////
//[BRDF]MicrofacetReflection BRDF Sample_f()
static __device__ __inline__ float4 MicrofacetReflection_InnerSample_f(const float3 &wo_Local, float3 & outwi_Local, float2 & urand, float & outPdf, const CommonStructs::ShaderParams & shaderParams, const bool isConductor)
{
	/*one-sided BSDF detection*/
	if (BSDFMath::CosTheta(wo_Local) <= 0.f)
		return make_float4(0.f);

	if (wo_Local.z == 0)
	{
		return make_float4(0.f);
	}
	
	/*bug fixed:flipping viewing direction(wo_Local)
	 *to the upper hemisphere when sampling VNDF*/
	float3 wh_Local = SamplingMicrofacetNormal_Sample_wh(
		TwUtil::signum(BSDFMath::CosTheta(wo_Local))*wo_Local, urand, shaderParams);
	float microfacetPdf = SamplingMicrofacetNormal_Pdf(TwUtil::signum(BSDFMath::CosTheta(wo_Local))*wo_Local, wh_Local, shaderParams);

	outwi_Local = TwUtil::reflect(wo_Local, wh_Local);//review reflect
	if (!SameHemisphere(wo_Local, outwi_Local))
	{
		return make_float4(0.f);
	}

	//p_wo(wo) = p_wh(wh) * |dwh / dwo| = p_wh(wh) * 1 / (4 * |dot(wo,wh)|)
	outPdf = microfacetPdf / (4 * dot(wo_Local, wh_Local));
	return MicrofacetReflection_InnerEval_f(wo_Local, outwi_Local, shaderParams, isConductor);
}

//////////////////////////////////////////////////////////////////////////
//[BRDF]MicrofacetReflection BRDF Pdf()
static __device__ __inline__ float MicrofacetReflection_InnerPdf(const float3 & wo_Local, const float3 & wi_Local, const CommonStructs::ShaderParams & shaderParams)
{
	/*one-sided BSDF detection*/
	if (BSDFMath::CosTheta(wo_Local) <= 0.f || BSDFMath::CosTheta(wi_Local) <= 0.f)
		return 0.f;

	float3 wh_Local = safe_normalize(wo_Local + wi_Local);
	//p_wo(wo) = p_wh(wh) * |dwh / dwo| = p_wh(wh) * 1 / (4 * |dot(wo,wh)|)
	return SamplingMicrofacetNormal_Pdf(TwUtil::signum(BSDFMath::CosTheta(wo_Local))*wo_Local, wh_Local, shaderParams) / (4 * dot(wo_Local, wh_Local));
}

#if 0


//////////////////////////////////////////////////////////////////////////
//[BRDF]MicrofacetTransmission BRDF Sample_f()
//todo:reviewing IOR issue
static __device__ __inline__ float4 MicrofacetTransmission_InnerEval_f(const float3 & wo_Local, const float3 & wi_Local, const CommonStructs::ShaderParams & shaderParams)
{
	float etaI = 1.f, etaT = shaderParams.ior;
	if (SameHemisphere(wo_Local, wi_Local))
	{
		return make_float4(0.f);
	}

	float cosThetaO = BSDFMath::AbsCosTheta(wo_Local);
	float cosThetaI = BSDFMath::AbsCosTheta(wi_Local);

	float eta = TwUtil::BSDFMath::CosTheta(wo_Local) > 0 ? (etaT / etaI) : (etaI / etaT);
	float3 wh = safe_normalize(wo_Local + wi_Local * eta);
	if (wh.z < 0) wh = -wh;

	// Handle degenerate cases for microfacet reflection
	if (cosThetaI == 0 || cosThetaO == 0)
	{
		return make_float4(0.f);
	}

	float F = 1.f - FrDiel_Evaluate(dot(wo_Local, wh), etaI, etaT);
	float sqrtDenom = dot(wo_Local, wh) + eta * dot(wi_Local, wh);

	//todo:
	/* Missing term in the original paper: account for the solid angle
			   compression when tracing radiance -- this is necessary for
			   bidirectional methods */
	float factor = 1 / eta;

	if ((GGX_D(wh, shaderParams) == 0.f))
		rtPrintf("%f %f %f\n", wh.x, wh.y, wh.z);

	//todo:missing transmittance coefficient from shaderParams
	return make_float4(F *
		fabsf(GGX_D(wh, shaderParams) * Smith_G(wo_Local, wi_Local, shaderParams) * eta * eta *
			fabsf(dot(wi_Local, wh)) * fabs(dot(wo_Local, wh)) * factor * factor /
			(cosThetaI * cosThetaO * sqrtDenom * sqrtDenom)));
}

//////////////////////////////////////////////////////////////////////////
//[BRDF]MicrofacetTransmission BRDF Pdf()
//todo:reviewing if(wh.z<0)
static __device__ __inline__ float MicrofacetTransmission_InnerPdf(const float3 & wo_Local, const float3 & wi_Local, const ShaderParams & shaderParams, float3 &outWh_Local)
{
	if (SameHemisphere(wo_Local, wi_Local))
	{
		//rtPrintf("error %f %f %f %f %f %f\n", wo_Local.x, wo_Local.y, wo_Local.z, wi_Local.x, wi_Local.y, wi_Local.z);
		return 0.f;
	}

	float etaI = 1.f, etaT = shaderParams.ior;
	// Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
	float eta = TwUtil::BSDFMath::CosTheta(wo_Local) > 0 ? (etaT / etaI) : (etaI / etaT);
	float3 wh = safe_normalize(wo_Local + eta * wi_Local);
	if (wh.z < 0)
	{
		wh = -wh;//half-vector always points out of the surface(in the same hemisphere with normal)
	}

	outWh_Local = wh;

	// Compute change of variables _dwh\_dwi_ for microfacet transmission
	float sqrtDenom = dot(wo_Local, wh) + eta * dot(wi_Local, wh);
	float dwh_dwi =
		fabsf((eta * eta * dot(wi_Local, wh)) / (sqrtDenom * sqrtDenom));

	return SamplingMicrofacetNormal_Pdf(TwUtil::signum(BSDFMath::CosTheta(wo_Local))*wo_Local, wh, shaderParams) * dwh_dwi;
}

//////////////////////////////////////////////////////////////////////////
//[BRDF]MicrofacetTransmission BRDF Sample_f()
static __device__ __inline__ float4 MicrofacetTransmission_InnerSample_f(const float3 &wo_Local, float3 & outwi_Local, float2 & urand, float & outPdf, const ShaderParams & shaderParams)
{
	if (wo_Local.z == 0)
	{
		return make_float4(0.f);
	}

	float3 wh_Local = SamplingMicrofacetNormal_Sample_wh(
		TwUtil::signum(BSDFMath::CosTheta(wo_Local))*wo_Local, urand, shaderParams);

	float etaI = 1.f, etaT = shaderParams.ior;
	float eta = TwUtil::BSDFMath::CosTheta(wo_Local) > 0 ? (etaI / etaT) : (etaT / etaI);
	if (!TwUtil::refract(wo_Local, wh_Local, eta, outwi_Local))
		return make_float4(0.f);//total internal reflection

	//just call InnerPdf to evaluate pdf
	//todo:or use wh_Local we've known and inline by hand
	float3 tmp = make_float3(0.f);
	outPdf = MicrofacetTransmission_InnerPdf(wo_Local, outwi_Local, shaderParams, tmp);

	if (SameHemisphere(wo_Local, outwi_Local))
	{
		return make_float4(0.f);
	}

	return MicrofacetTransmission_InnerEval_f(wo_Local, outwi_Local, shaderParams);
}
#endif 

#endif // COLVILLEA_DEVICE_SHADER_MICROFACETBRDF_H_