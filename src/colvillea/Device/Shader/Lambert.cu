#include "LambertBRDF.h"

using namespace optix;
using namespace CommonStructs;

//////////////////////////////////////////////////////////////////////////
//Forward declarations:
//system related:->Context
//rtDeclareVariable(DifferentialGeometry, dgShading, attribute dgShading, );
//rtDeclareVariable(optix::float4,		nGeometry, attribute nGeometry, );


//////////////////////////////////////////////////////////////////////////
//[BSDF]Lambert Pdf function:
RT_CALLABLE_PROGRAM float Lambert_Pdf(const float3 & wo_World, const float3 & wi_World, const ShaderParams & shaderParams)
{
	if (TwUtil::dot(wi_World, shaderParams.nGeometry) * TwUtil::dot(wo_World, shaderParams.nGeometry) > 0)
	{
		//evaluate BRDF only
		float3 wo = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
		float3 wi = TwUtil::BSDFMath::WorldToLocal(wi_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

		return Lambert_InnerPdf(wo, wi, shaderParams);
	}
	else
	{
		return 0.f;
	}
}

//[BSDF]Lambert Eval_f function:
RT_CALLABLE_PROGRAM float4 Lambert_Eval_f(const float3 & wo_World, const float3 & wi_World, const ShaderParams & shaderParams)
{
	//Technically, we need to transform wo_World and wi_World into Local Shading Coordinate System.For Lambertian Shader, this transformation is ommited(just constant).
	

	if (TwUtil::dot(wi_World, shaderParams.nGeometry) * TwUtil::dot(wo_World, shaderParams.nGeometry) > 0)
	{
		float3 wo_Local = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
		float3 wi_Local = TwUtil::BSDFMath::WorldToLocal(wi_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

		return Lambert_InnerEval_f(wo_Local, wi_Local, shaderParams);
	}
	else
	{
		return make_float4(0.f);
	}
}



//[BSDF]Lambert Sample_f function:
RT_CALLABLE_PROGRAM float4 Lambert_Sample_f(const float3 &wo_World, float3 & outwi_World, float2 & urand, float & outPdf, float bsdfChoiceRand, const ShaderParams & shaderParams)
{
	//Bug found when transplanting to OptiX, Shader::Sample_f() method should calculate after transform wo_World into Shading Coordinate System and transform back to World Coordinate System.
	//Flip wo_world was wrong in the original code.
	//Refer to Mirror.cu Sample_f()
	float3 wo = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
	float3 wi = make_float3(0.f);

	float4 f = Lambert_InnerSample_f(wo, wi, urand, outPdf, shaderParams);

	

	if (outPdf == 0.f)
	{
		return make_float4(0.f);
	}

	outwi_World = TwUtil::BSDFMath::LocalToWorld(wi, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

	

	if (TwUtil::dot(outwi_World, shaderParams.nGeometry) * TwUtil::dot(wo_World, shaderParams.nGeometry) > 0)
	{
		return Lambert_InnerEval_f(wo, wi, shaderParams);
	}
	else
	{
		return make_float4(0.f);
	}
}