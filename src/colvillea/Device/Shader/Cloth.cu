#include <optix_world.h>
#include <optix_device.h>
#include <optixu_math_namespace.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"
#include "../Toolkit/NvRandom.h"

// For Cosine-weighted sampling
#include "LambertBRDF.h"

using namespace optix;
using namespace TwUtil;
using namespace CommonStructs;

// TODO:
// * Add more cloth distribution D
// * Add diffuse component
// * Importance sampling

static __device__ __inline__ float Ashikhmin_D(float roughness, float NoH)
{
	// Ref: Filament implementation
	// Ashikhmin 2007, "Distribution-based BRDFs"
	float a2 = roughness * roughness;
	float cos2h = NoH * NoH;
	float sin2h = fmaxf(1.0f - cos2h, 0.0078125f); // 2^(-14/2), so sin2h^2 > 0 in fp16
	float sin4h = sin2h * sin2h;
	float cot2 = -cos2h / (a2 * sin2h);
	return 1.0 / (M_PIf * (4.0 * a2 + 1.0f) * sin4h) * (4.0f * expf(cot2) + sin4h);
}

//////////////////////////////////////////////////////////////////////////
//[BSDF]Cloth BSDF Sample_f():
RT_CALLABLE_PROGRAM float4 Sample_f_Cloth(const float3& wo_World, float3& outwi_World, float2& urand, float& outPdf, float bsdfChoiceRand, const ShaderParams& shaderParams)
{
	/* Transform wo,wi to TBN. */
	float3 wo_Local = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
	float3 outwi_Local = make_float3(0.f);

	/* Single sided BRDF detection. */
	if (BSDFMath::CosTheta(wo_Local) <= 0.f)
	{
		return make_float4(0.f);
	}

	/* Cosine weighted sampling Wi. */
	outwi_Local = TwUtil::MonteCarlo::CosineSampleHemisphere(urand);
	if (outwi_Local.z < 0.) { rtPrintf("Assert failed at %s,%d wo_Local.z < 0.!\n", __FILE__, __LINE__); }
	outwi_World = TwUtil::BSDFMath::LocalToWorld(outwi_Local, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
	if (TwUtil::dot(outwi_World, shaderParams.nGeometry) * TwUtil::dot(wo_World, shaderParams.nGeometry) <= 0.f)
	{
		return make_float4(0.f);
	}

	/* Pdf. */
	outPdf = BSDFMath::CosTheta(outwi_Local) * M_1_PIf;

	/* Eval_f. */
	float3 wh = outwi_Local + wo_Local;
	if (wh.x == 0 && wh.y == 0 && wh.z == 0) return make_float4(0.f);
	wh = safe_normalize(wh);

	return make_float4(make_float3(Ashikhmin_D(shaderParams.alphax, BSDFMath::AbsCosTheta(wh))), 1.0f);
}

// [BSDF]Cloth Eval_f function:
// Ref: Michael Ashikhmin and Simon Premoze. 2007. Distribution-based BRDFs.
RT_CALLABLE_PROGRAM float4 Eval_f_Cloth(const float3& wo_World, const float3& wi_World, const ShaderParams& shaderParams)
{
    // Specular D only:
	if (TwUtil::dot(wi_World, shaderParams.nGeometry) * TwUtil::dot(wo_World, shaderParams.nGeometry) > 0)
	{
		float3 wo_Local = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
		float3 wi_Local = TwUtil::BSDFMath::WorldToLocal(wi_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

		/* Single sided BRDF detection. */
		if (BSDFMath::CosTheta(wo_Local) <= 0.f || BSDFMath::CosTheta(wi_Local) <= 0.f)
			return make_float4(0.f);

		/* Eval_f. */
		float3 wh = wi_Local + wo_Local;
		if (wh.x == 0 && wh.y == 0 && wh.z == 0) return make_float4(0.f);
		wh = safe_normalize(wh);

		return make_float4(make_float3(Ashikhmin_D(shaderParams.alphax, BSDFMath::AbsCosTheta(wh))), 1.0f);
	}
	else
	{
		return make_float4(0.f);
	}
}

//////////////////////////////////////////////////////////////////////////
// [BSDF]Cloth Pdf function:
RT_CALLABLE_PROGRAM float Pdf_Cloth(const float3& wo_World, const float3& wi_World, const ShaderParams& shaderParams)
{
	/* Light leak prevention. */
	if (TwUtil::dot(wi_World, shaderParams.nGeometry) * TwUtil::dot(wo_World, shaderParams.nGeometry) > 0)
	{
		float3 wo = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
		float3 wi = TwUtil::BSDFMath::WorldToLocal(wi_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

		/* Single sided BRDF detection. */
		if (BSDFMath::CosTheta(wo) <= 0.f || BSDFMath::CosTheta(wi) <= 0.f)
			return 0.f;

		/* Cosine weighted sample pdf. */
		return BSDFMath::CosTheta(wi) * M_1_PIf;
	}
	else
	{
		return 0.f;
	}
}