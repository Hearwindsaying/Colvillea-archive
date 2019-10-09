#pragma once
#include "MicrofacetBRDF.h"
#include "LambertBRDF.h"

using namespace optix;
using namespace CommonStructs;

#define SAMPLE_VNDF

/********************************************************************************/
/*********************************Inner BRDF*************************************/
/********************************************************************************/
static __device__ __inline__ float4 FresnelBlend_InnerEval_f(const float3 & wo_Local, const float3 & wi_Local, const ShaderParams & shaderParams)
{
	/*one-sided BSDF detection*/
	if (BSDFMath::CosTheta(wo_Local) <= 0.f || BSDFMath::CosTheta(wi_Local) <= 0.f)
		return make_float4(0.f);

	auto pow5 = [](float v) { return (v * v) * (v * v) * v; };


	float4 Rs = shaderParams.Specular;
	if (shaderParams.SpecularID != RT_TEXTURE_ID_NULL)
	{
		Rs = rtTex2D<float4>(shaderParams.SpecularID, shaderParams.dgShading.uv.x, shaderParams.dgShading.uv.y);
	}
	float4 Rd = shaderParams.Reflectance;
	if (shaderParams.ReflectanceID != RT_TEXTURE_ID_NULL)
	{
		Rd = rtTex2D<float4>(shaderParams.ReflectanceID, shaderParams.dgShading.uv.x, shaderParams.dgShading.uv.y);
	}

	auto SchlickFresnel = [pow5, &Rs](float cosTheta)
	{
		return Rs + pow5(1 - cosTheta) * (make_float4(1.f) - Rs);
	};

	float4 diffuse = (28.f / (23.f * M_PIf)) * Rd * (make_float4(1.f) - Rs) *
		(1 - pow5(1 - .5f * BSDFMath::AbsCosTheta(wi_Local))) *
		(1 - pow5(1 - .5f * BSDFMath::AbsCosTheta(wo_Local)));
	float3 wh = wi_Local + wo_Local;
	if (wh.x == 0 && wh.y == 0 && wh.z == 0) return make_float4(0.f);
	wh = safe_normalize(wh);
	float4 specular =
		GGX_D(wh, shaderParams) /
		(4 * fabsf(dot(wi_Local, wh)) * fmaxf(BSDFMath::AbsCosTheta(wi_Local), BSDFMath::AbsCosTheta(wo_Local))) *
		SchlickFresnel(dot(wi_Local, wh));

	return diffuse + specular;
}


static __device__ __inline__ float FresnelBlend_InnerPdf(const float3 & wo_Local, const float3 & wi_Local, const ShaderParams & shaderParams, bool print=false)
{
	/*one-sided BSDF detection*/
	if (BSDFMath::CosTheta(wo_Local) <= 0.f || BSDFMath::CosTheta(wi_Local) <= 0.f)
		return 0.f;

	float3 wh = safe_normalize(wo_Local + wi_Local);
#ifdef SAMPLE_VNDF
	float pdf_wh = SamplingMicrofacetNormal_Pdf(TwUtil::signum(BSDFMath::CosTheta(wo_Local))*wo_Local, wh, shaderParams);
#else
	//float pdf_wh = SamplingMicrofacetNormal_Pdf(wh, shaderParams);
	float pdf_wh = GGX_D(wh, shaderParams) * BSDFMath::AbsCosTheta(wh);
	//if (print&&pdf_wh <= 20.f)
	//	rtPrintf("%f\n", pdf_wh);
#endif
	return .5f * (BSDFMath::AbsCosTheta(wi_Local) * M_1_PIf + pdf_wh / (4 * dot(wo_Local, wh)));
}

static __device__ __inline__ float4 FresnelBlend_InnerSample_f(const float3 & wo_Local, float3 & outwi_Local, const float2 &urand, const float & bsdfChoiceRand, float &outpdf, const ShaderParams & shaderParams)
{
	/*one-sided BSDF detection*/
	if (BSDFMath::CosTheta(wo_Local) <= 0.f)
		return make_float4(0.f);

	if (bsdfChoiceRand < .5f) 
	{
		// Cosine-sample the hemisphere
		outwi_Local = TwUtil::MonteCarlo::CosineSampleHemisphere(urand);
	}
	else 
	{
		// Sample microfacet orientation $\wh$ and reflected direction $\wi$
#ifdef SAMPLE_VNDF
		float3 wh_Local = SamplingMicrofacetNormal_Sample_wh(
			TwUtil::signum(BSDFMath::CosTheta(wo_Local))*wo_Local, urand, shaderParams);
#else
		float __pdf = 0.f;
		float3 wh_Local = SamplingMicrofacetNormal_Sample_wh(urand, shaderParams, __pdf);
#endif
		outwi_Local = TwUtil::reflect(wo_Local, wh_Local);//review reflect
		if (!SameHemisphere(wo_Local, outwi_Local))
		{
			return make_float4(0.f);
		}
	}

	outpdf = FresnelBlend_InnerPdf(wo_Local, outwi_Local, shaderParams);
	return FresnelBlend_InnerEval_f(wo_Local, outwi_Local, shaderParams);
}



/********************************************************************************/
/*********************************   BSDF   *************************************/
/********************************************************************************/
RT_CALLABLE_PROGRAM float4 Eval_f_Plastic(const float3 & wo_World, const float3 & wi_World, const ShaderParams & shaderParams)
{
#ifdef PLASTIC_ADDING
	float3 wi = TwUtil::BSDFMath::WorldToLocal(wi_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
	float3 wo = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

	if (dot(wi_World, shaderParams.nGeometry) * dot(wo_World, shaderParams.nGeometry) > 0)
	{
		//ignore BTDFs
		float4 f = MicrofacetReflection_InnerEval_f(wo, wi, shaderParams, false);
		f += Lambert_InnerEval_f(wo, wi, shaderParams);
		return f;
	}
	else
	{
		return make_float4(0.f);
	}
#else
	float3 wi = TwUtil::BSDFMath::WorldToLocal(wi_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
	float3 wo = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

	if (dot(wi_World, shaderParams.nGeometry) * dot(wo_World, shaderParams.nGeometry) > 0)
	{
		return FresnelBlend_InnerEval_f(wo, wi, shaderParams);
	}
	else
	{
		return make_float4(0.f);
	}
#endif
}


RT_CALLABLE_PROGRAM float4 Sample_f_Plastic(const float3 &wo_World, float3 & outwi_World, float2 & urand, float & outPdf, float bsdfChoiceRand, const ShaderParams & shaderParams)
{
#ifdef PLASTIC_ADDING
	float3 wo = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
	float3 wi = make_float3(0.f);

	outPdf = 0.f;

	float4 f = make_float4(0.f);
	if (bsdfChoiceRand > .5f)
	{
		f = Lambert_InnerSample_f(wo, wi, urand, outPdf, shaderParams);

		if (outPdf == 0.f)
			return make_float4(0.f);
	}
	else
	{
		f = MicrofacetReflection_InnerSample_f(wo, wi, urand, outPdf, shaderParams, false);

		if (outPdf == 0.f)
			return make_float4(0.f);
	}

	outwi_World = TwUtil::BSDFMath::LocalToWorld(wi, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

	//Compute overall Pdf
	if (bsdfChoiceRand > .5f)
		outPdf += MicrofacetReflection_InnerPdf(wo, wi, shaderParams);
	else
		outPdf += Lambert_InnerPdf(wo, wi, shaderParams);

	outPdf /= 2.f;

	//Compute value of BSDF for sampled direction
	if (dot(outwi_World, shaderParams.nGeometry) * dot(wo_World, shaderParams.nGeometry) > 0)
	{
		//ignore BTDFs
		f = MicrofacetReflection_InnerEval_f(wo, wi, shaderParams, false);
		f += Lambert_InnerEval_f(wo, wi, shaderParams);
	}
	else
	{
		f = make_float4(0.f);
	}

	return f;
#else
	float3 wo = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
	float3 wi = make_float3(0.f);

	float4 f = FresnelBlend_InnerSample_f(wo, wi, urand, bsdfChoiceRand, outPdf, shaderParams);

	if (outPdf == 0.f)
	{
		return make_float4(0.f);
	}

	outwi_World = TwUtil::BSDFMath::LocalToWorld(wi, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

	if (TwUtil::dot(outwi_World, shaderParams.nGeometry) * TwUtil::dot(wo_World, shaderParams.nGeometry) > 0)
	{
		return f;
	}
	else
	{
		return make_float4(0.f);
	}
#endif
}

RT_CALLABLE_PROGRAM float Pdf_Plastic(const float3 & wo_World, const float3 & wi_World, const ShaderParams & shaderParams)
{
#ifdef PLASTIC_ADDING
	//need to be corrected(consider light leak and light spot)
	float3 wo = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
	float3 wi = TwUtil::BSDFMath::WorldToLocal(wi_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

	float pdf = MicrofacetReflection_InnerPdf(wo, wi, shaderParams);
	pdf += Lambert_InnerPdf(wo, wi, shaderParams);

	pdf /= 2.f;
	return pdf;
#else
	float3 wo = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
	float3 wi = TwUtil::BSDFMath::WorldToLocal(wi_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
	return FresnelBlend_InnerPdf(wo, wi, shaderParams);
#endif
}