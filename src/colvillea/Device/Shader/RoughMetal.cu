#include "MicrofacetBRDF.h"

/*This is a double-sided material*/
using namespace CommonStructs;

//////////////////////////////////////////////////////////////////////////
//[BSDF]RoughMetal BSDF Pdf()
RT_CALLABLE_PROGRAM float Pdf_RoughMetal(const float3 & wo_World, const float3 & wi_World, const ShaderParams & shaderParams)
{
	//Revising light leak/light spot issue.
	if (TwUtil::dot(wi_World, shaderParams.nGeometry) * TwUtil::dot(wo_World, shaderParams.nGeometry) > 0)
	{
		//evaluate BRDF only
		float3 wo_Local = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
		float3 wi_Local = TwUtil::BSDFMath::WorldToLocal(wi_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

		/*double-sided material, flip the evaluated direction if necessary*/
		if (BSDFMath::CosTheta(wo_Local) <= 0.f)
		{
			wo_Local *= -1.f;
			wi_Local *= -1.f;
		}

		return MicrofacetReflection_InnerPdf(wo_Local, wi_Local, shaderParams);
	}
	return 0.f;
}

//////////////////////////////////////////////////////////////////////////
//[BSDF]RoughMetal BSDF Eval_f()
RT_CALLABLE_PROGRAM float4 Eval_f_RoughMetal(const float3 & wo_World, const float3 & wi_World, const ShaderParams & shaderParams)
{
	if (TwUtil::dot(wi_World, shaderParams.nGeometry) * TwUtil::dot(wo_World, shaderParams.nGeometry) > 0)
	{//ignore btdf and evaluate brdf only
		float3 wo_Local = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
		float3 wi_Local = TwUtil::BSDFMath::WorldToLocal(wi_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

		/*double-sided material, flip the evaluated direction if necessary*/
		if (BSDFMath::CosTheta(wo_Local) <= 0.f)
		{
			wo_Local *= -1.f;
			wi_Local *= -1.f;
		}

		return MicrofacetReflection_InnerEval_f(wo_Local, wi_Local, shaderParams, true);
	}
	//else ignore brdf and btdf doesn't exist here, so we just return 0.f:
	return make_float4(0.f);
}

//////////////////////////////////////////////////////////////////////////
//[BSDF]RoughMetal BSDF Sample_f():
RT_CALLABLE_PROGRAM float4 Sample_f_RoughMetal(const float3 &wo_World, float3 & outwi_World, float2 & urand, float & outPdf, float bsdfChoiceRand, const ShaderParams & shaderParams)
{
	float3 wo_Local = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
	float3 outwi_Local = make_float3(0.f);

	bool flipped = false;
    if (BSDFMath::CosTheta(wo_Local) <= 0.f)
    {
        wo_Local *= -1.f;
        flipped = true;
    }
		

	float4 f = MicrofacetReflection_InnerSample_f(wo_Local, outwi_Local, urand, outPdf, shaderParams, true);
	if (outPdf == 0.f)
		return make_float4(0.f);

	if (flipped)
		outwi_Local *= -1.f;

	outwi_World = TwUtil::BSDFMath::LocalToWorld(outwi_Local, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

	if (TwUtil::dot(outwi_World, shaderParams.nGeometry) * TwUtil::dot(wo_World, shaderParams.nGeometry) > 0)
	{//ignore btdf and evaluate brdf only
		return f;
	}
	return make_float4(0.f);
}