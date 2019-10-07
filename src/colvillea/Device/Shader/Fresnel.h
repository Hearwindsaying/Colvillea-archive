#pragma once
#ifndef COLVILLEA_DEVICE_SHADER_FRESNEL_H_
#define COLVILLEA_DEVICE_SHADER_FRESNEL_H_
#include <optix_world.h>
#include <optix_device.h>
#include <optixu_math_namespace.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"

using namespace optix;
using namespace TwUtil;

static __device__ __inline__  float3 FresnelConductor(float cosi, const float3 & eta, const float3 & k)
{
	float3 tmp = (eta*eta + k * k)*cosi*cosi;
	float3 Rparl2 = (tmp - (2.f * eta * cosi) + 1.0f) /
					(tmp + (2.f * eta * cosi) + 1.0f);
	float3 tmp_f = eta * eta + k * k;
	float3 Rperp2 =
		(tmp_f - (2.f * eta * cosi) + cosi * cosi) /
		(tmp_f + (2.f * eta * cosi) + cosi * cosi);
	return (Rparl2 + Rperp2) / 2.f;
}

static __device__ __inline__ float3 FrCond_Evaluate(float cosi, const CommonStructs::ShaderParams & shaderParams)
{
	return FresnelConductor(fabsf(cosi), make_float3(shaderParams.FrCond_e), make_float3(shaderParams.FrCond_k));
}

//deprecated
//note that cosThetaI lies in shading coordinates(local)
static __device__ __inline__ float FrDiel_Evaluate(float cosThetaI, float etaI, float etaT)
{
	rtPrintf("deprecated!\n");
	cosThetaI = optix::clamp(cosThetaI, -1.f, 1.f);
	// Potentially swap indices of refraction
	bool entering = cosThetaI > 0.f;
	if (!entering) 
	{
		//swap(etaI, etaT);
		float tmp = etaI; etaI = etaT; etaT = tmp;
		cosThetaI = fabsf(cosThetaI);
	}

	// Compute _cosThetaT_ using Snell's law
	float sinThetaI = sqrtf(optix::fmaxf(0.f, 1 - cosThetaI * cosThetaI));
	float sinThetaT = etaI / etaT * sinThetaI;

	// Handle total internal reflection
	if (sinThetaT >= 1) 
		return 1.f;

	float cosThetaT = sqrtf(optix::fmaxf(0.f, 1 - sinThetaT * sinThetaT));
	float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
		((etaT * cosThetaI) + (etaI * cosThetaT));
	float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
		((etaI * cosThetaI) + (etaT * cosThetaT));
	return (Rparl * Rparl + Rperp * Rperp) / 2;
}

/**
 * @brief Calculate the fresnel dielectric coefficient at a planar interface between two dielectric medium(one side is usually air) and return the corresponding cosThetaT using Snell's law.
 * @param __cosThetaI indicates the incident direction, cosThetaI > 0.0 means the ray is incident
 * @param outCosThetaT the transmitted direction calculated using Snell's law, the value is 0.0f when total internal reflection incured and return 1.0(Fresnel eqaution is not needed at the very moment)
 * @param eta relative index of refraction, i.e. the ratio of transmitted(inner or etaB) media and incident(outer or etaA) media
 * @return the fresnel reflection using the dielectric and unpolarized Fresnel equation
 * @see FrDiel_Evaluate()(under construction)
 */
static __device__ __inline__ float FresnelDielectricExt(const float cosThetaI, float & outCosThetaT, const float eta)
{
	/*figure out the proper etaI/etaT according to the cosThetaI,
	 *note that the ray is incident when cosThetaI > 0.f, thus taking the reciprocal of eta(eta=etaT/etaI*/
	float etaI_etaT = (cosThetaI > 0.f) ? 1.f / eta : eta;

	/*compute the cosThetaTSqr using Snell's law*/
	float cosThetaTSqr = 1 - (etaI_etaT * etaI_etaT) * (1.f - cosThetaI * cosThetaI);

	/*check for the TIR*/
	if (cosThetaTSqr <= 0.f)
	{
		/*outCosThetaT gets 0.0 when TIR incured*/
		outCosThetaT = 0.f;
		return 1.f;
	}

	/*remember that use abosolute cosine while evaluating the Fresnel equation*/
	float absCosThetaI = fabsf(cosThetaI);
	float absCosThetaT = sqrtf(cosThetaTSqr);

	float Rs = (absCosThetaI - eta * absCosThetaT)
		/ (absCosThetaI + eta * absCosThetaT);
	float Rp = (eta * absCosThetaI - absCosThetaT)
		/ (eta * absCosThetaI + absCosThetaT);

	/*find out the correct outCosThetaT*/
	/* review:typo:outCosThetaT = (outCosThetaT > 0.f) ? -cosThetaT : cosThetaT;*/
	outCosThetaT = (cosThetaI > 0.f) ? -absCosThetaT : absCosThetaT;

	/* No polarization -- return the unpolarized reflectance */
	return 0.5f * (Rs * Rs + Rp * Rp);
}

#endif // COLVILLEA_DEVICE_SHADER_FRESNEL_H_