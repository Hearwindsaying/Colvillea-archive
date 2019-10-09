#include "MicrofacetBRDF.h"

rtDeclareVariable(uint2, sysLaunch_index, rtLaunchIndex, );
#define SAMPLING_VNDF

using namespace CommonStructs;

namespace TwUtil
{
	/**
	 * @brief Specularly refract the incident direction wi into transmitted direction and return.
	 * Note that this utility function requires a precomputed cosThetaT for transmitted direction(using FresnelDielectricExt()) and won't handle the total internal reflection.A forehand check of cosThetaT equal to 0.0 is necessary.
	 * @param wi incident direction
	 * @param n normal of the planar dielectric
	 * @param eta relative index of refraction, i.e. the ratio of transmitted(inner or etaB) media and incident(outer or etaA) media
	 * @return the transmitted direction
	 * @see FresnelDielectricExt()
	 */
	static __device__ __inline__ float3 Refract(const float3 &wi, const float3 &n,
		const float eta, const float cosThetaT)
	{
#if 1
		/*figure out the proper etaI/etaT according to the cosThetaT*/
		float etaI_etaT = (cosThetaT < 0.f) ? 1.f / eta : eta;
		/*if (cosThetaT < 0.f && dot(wi, n) < 0.f)
			rtPrintf("never happens!!!!!\n");*/

		/*(reviewed)compute the refracted direction, note that the sign is implicit in the cosThetaT which differs from pbrt's implementation of refract()*/
		return n * (dot(wi, n) * etaI_etaT + cosThetaT) - wi * etaI_etaT;
#endif
#if 0
		float etaI_etaT = (cosThetaT < 0.f) ? 1.f / eta : eta;
		float cosThetaI = dot(n, wi);
		float sin2ThetaI = fmaxf(0.f, 1.f - cosThetaI * cosThetaI);
		float sin2ThetaT = etaI_etaT * etaI_etaT * sin2ThetaI;
		if (sin2ThetaT >= 1.f)
		{
			rtPrintf("TIR never happens here!!!!!\n");
		}
		float __cosThetaT = sqrtf(1.f - sin2ThetaT);

		__cosThetaT = cosThetaT < 0.f ? -__cosThetaT : __cosThetaT;
		if (__cosThetaT != cosThetaT)
			rtPrintf("%f %f\n", __cosThetaT, cosThetaT);
		return n * (dot(wi, n) * etaI_etaT + cosThetaT) - wi * etaI_etaT;
#endif
	}
};

//////////////////////////////////////////////////////////////////////////
//[BSDF]RoughDielectric BSDF Pdf()
RT_CALLABLE_PROGRAM float Pdf_RoughDielectric(const float3 & wo_World, const float3 & wi_World, const ShaderParams & shaderParams)
{
	float3 wo_Local = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
	float3 wi_Local = TwUtil::BSDFMath::WorldToLocal(wi_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

	bool reflect = BSDFMath::CosTheta(wo_Local) * BSDFMath::CosTheta(wi_Local) > 0.f;

	float3 H = make_float3(0.f);
	float dwh_dwi = 0.f;

	if (reflect)
	{
		H = safe_normalize(wi_Local + wo_Local);
		dwh_dwi = 1.f / (4.f * fabsf(dot(wo_Local, H)));
	}
	else
	{
		float etaT_etaI = (BSDFMath::CosTheta(wo_Local) > 0.f) ? (shaderParams.ior / 1.f) : (1.f / shaderParams.ior);
		H = safe_normalize(etaT_etaI * wi_Local + wo_Local);

		/*jacobian determinant of half-direction transformation*/
		float sqrtDenom = dot(wo_Local, H) + etaT_etaI * dot(wi_Local, H);
		dwh_dwi = fabsf(etaT_etaI * etaT_etaI * dot(wi_Local, H)) / (sqrtDenom * sqrtDenom);
	}

	/*ensure the orientation be consistent with the macrosurface normal(always on the side of(0,0,1)*/
	if (BSDFMath::CosTheta(H) < 0.f)
		H *= -1.f;

	/*evaluate pdf of sampling microfacet normal*/
#ifdef SAMPLING_VNDF
	float prob = SamplingMicrofacetNormal_Pdf(TwUtil::signum(BSDFMath::CosTheta(wo_Local))*wo_Local, H, shaderParams);
#else
	float prob = SamplingMicrofacetNormal_Pdf(H, shaderParams);
#endif

	/*stretch fresnel term into pdf*/
	float __cosThetaT = 0.f;/*unused argument for FresnelDielectric input*/
	const float F = FresnelDielectricExt(dot(wo_Local, H), __cosThetaT, shaderParams.ior / 1.f);
	prob *= (reflect ? F : 1.f - F);

	/*involve with jacobian determinant*/
	prob *= fabsf(dwh_dwi);/*fabsf could be ignored*/

	return prob;
}


//////////////////////////////////////////////////////////////////////////
//[BSDF]RoughDielectric BSDF Eval_f()
RT_CALLABLE_PROGRAM float4 Eval_f_RoughDielectric(const float3 & wo_World, const float3 & wi_World, const ShaderParams & shaderParams)
{
	float3 wo_Local = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
	float3 wi_Local = TwUtil::BSDFMath::WorldToLocal(wi_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

	if (BSDFMath::CosTheta(wo_Local) == 0.f)
		return make_float4(0.f);

	bool reflect = BSDFMath::CosTheta(wo_Local) * BSDFMath::CosTheta(wi_Local) > 0.f;

	float3 H = make_float3(0.f);
	if (reflect)
	{
		H = safe_normalize(wi_Local + wo_Local);
	}
	else
	{
		float etaI_etaT = (BSDFMath::CosTheta(wo_Local) > 0.f) ? (1.f / shaderParams.ior) : (shaderParams.ior / 1.f);
		H = safe_normalize(wi_Local + etaI_etaT * wo_Local);
	}
	
	/*ensure the orientation be consistent with the macrosurface normal(always on the side of(0,0,1)*/
	if (BSDFMath::CosTheta(H) < 0.f)
		H *= -1.f;

	/*evaluate the fresnel term*/
	float __cosThetaT = 0.f;/*unused argument for FresnelDielectric input*/
	const float F = FresnelDielectricExt(dot(wo_Local, H), __cosThetaT, shaderParams.ior / 1.f);

	/*evaluate the GGX NDF*/
	const float D = GGX_D(H, shaderParams);
	if (D == 0.f)
		return make_float4(0.f);

	/*evaluate the Smith_G*/
	const float G = Smith_G_Sep(wo_Local, wi_Local, H, shaderParams);

	/*when G==0.0, microfacetTransmission BTDF becomes ill-defined,thus forcing value to be zero*/
	if(G == 0.f)
		return make_float4(0.f);

	if (reflect)
	{
		float value = F * D * G / 
			(4.f * BSDFMath::AbsCosTheta(wo_Local) * BSDFMath::AbsCosTheta(wo_Local));

		/*if (isnan(value) || isinf(value))
			rtPrintf("reflect--F:%f D:%f G:%f \n", F, D, G);*/
		return make_float4(value);
	}
	else
	{
		/*figure out the correct etaI / etaT*/
		float etaT_etaI = (BSDFMath::CosTheta(wo_Local) > 0.f) ? (shaderParams.ior / 1.f) : (1.f / shaderParams.ior);

		/*float sqrtDenom = (dot(wo_Local, H) + etaI_etaT * (dot(wi_Local, H))) *
						  (dot(wo_Local, H) + etaI_etaT * (dot(wi_Local, H)));
		float cosTerm = fabsf(dot(wi_Local, H)) * fabsf(dot(wo_Local, H)) /
			(BSDFMath::AbsCosTheta(wo_Local) * BSDFMath::AbsCosTheta(wi_Local));
		float value = etaI_etaT * etaI_etaT * (1.f - F) * G * D * cosTerm / (sqrtDenom * sqrtDenom);*/
		float sqrtDenom = etaT_etaI * dot(wi_Local, H) + dot(wo_Local, H);
		float cosTerm = fabsf(dot(wi_Local, H)) * fabsf(dot(wo_Local, H)) /
			(BSDFMath::AbsCosTheta(wo_Local) * BSDFMath::AbsCosTheta(wi_Local));

		/*todo: account for solid angle compression for bidirectional methods*/
		float value = (1.f - F) * G * D * cosTerm / (sqrtDenom * sqrtDenom);

		/*if (isnan(value) || isinf(value))
			rtPrintf("refract--1-F:%f D:%f G:%f etaI/T:%f cosTerm:%f sqrtDenom:%f\n", 1.f-F, D, Smith_G_Sep(wo_Local, wi_Local, H, shaderParams,true),
				etaT_etaI,cosTerm,sqrtDenom);*/
		return make_float4(value);
	}
}

//////////////////////////////////////////////////////////////////////////
//[BSDF]RoughDielectric BSDF Sample_f()
//todo:revise light spot/leak issue
RT_CALLABLE_PROGRAM float4 Sample_f_RoughDielectric(const float3 &wo_World, float3 & outwi_World, float2 & urand, float & outPdf, float bsdfChoiceRand, const ShaderParams & shaderParams)
{
	/*convert direction into local space*/
	float3 wo_Local = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
	float3 outwi_Local = make_float3(0.f);

	bool hasReflection = true, hasTransmission = true, sampleReflection = true;

	/*sample the microfacet normal, M */
	
#ifdef SAMPLING_VNDF
	float3 m = SamplingMicrofacetNormal_Sample_wh(
		TwUtil::signum(BSDFMath::CosTheta(wo_Local))*wo_Local, urand, shaderParams);
	float microfacetPDF = SamplingMicrofacetNormal_Pdf(TwUtil::signum(BSDFMath::CosTheta(wo_Local))*wo_Local, m, shaderParams);
#else
	float microfacetPDF = 0.f;
	float3 m = SamplingMicrofacetNormal_Sample_wh(urand, shaderParams, microfacetPDF);
#endif
	if (microfacetPDF == 0.f)
		return make_float4(0.f);

	/*update outPdf*/
	outPdf = microfacetPDF;/*outPdf = p(wm)*/

	float cosThetaT = 0.f;
	float F = FresnelDielectricExt(dot(wo_Local, m), cosThetaT, shaderParams.ior / 1.f);
	float4 bsdfValue = make_float4(0.f);

	//if(hasReflection && hasTransmission)
	if (bsdfChoiceRand > F)
	{
		/*sample refraction*/
		sampleReflection = false;
		outPdf *= (1.f - F);/* outPdf = p(wm) * (1.f - F) */
	}
	else
	{
		outPdf *= F;/* outPdf = p(wm) * F */
	}

	float dwh_dwo = 0.f;
	if (sampleReflection)
	{
		/*compute specular reflected direction for outwi_Local with respect to microfacet normal m*/
		outwi_Local = TwUtil::reflect(wo_Local, m);
		/*todo:update the bRec.eta so as to use in path tracing Russian Roulette*/

		/*check side of the reflected outwi_Local
		 * -- this could possibly happen when the outwi_Local lies in the lower hemisphere while wo_Local in the upper hemisphere*/
		if (!SameHemisphere(outwi_Local, wo_Local))
			return make_float4(0.f);

		/* todo:bsdfValue = R */

		/*jacobian determinant for half-direction transformation*/
		dwh_dwo = 1.0f / (4.0f * dot(outwi_Local, m));

		/*added code:*/
		bsdfValue = make_float4(
			F * Smith_G_Sep(wo_Local, outwi_Local, m, shaderParams) * GGX_D(m, shaderParams) / 
			(4.f * BSDFMath::CosTheta(wo_Local) * BSDFMath::CosTheta(outwi_Local)));

		/*if (bsdfValue.x == 0.f)
			rtPrintf("SamplefReflect--1-F:%f D:%f G:%f\n", 1.f - F, GGX_D(m, shaderParams), Smith_G_Sep(wo_Local, outwi_Local, m, shaderParams));*/
	}
	else
	{
		/*sample refraction*/

		/*Handle the total internal reflection
		  -- this could never happen because it will always sample
		  -- reflection when TIR incured.*/
		if (cosThetaT == 0.f)
			return make_float4(0.f);

		/*compute specular refracted direction*/
		outwi_Local = TwUtil::Refract(wo_Local, m, shaderParams.ior / 1.f, cosThetaT);

		/*todo:update the bRec.eta so as to use in path tracing Russian Roulette*/

		/*side check*/
		if (SameHemisphere(outwi_Local, wo_Local))
			return make_float4(0.f);

		/* todo:bsdfValue = T */

		/* todo:Radiance must be scaled to account for the solid angle compression
			   that occurs when crossing the interface. */
		//float factor = (cosThetaT < 0.f) ? (1.f / shaderParams.ior) : (shaderParams.ior / 1.f);

		/*added code:*/
		float etaT_etaI = (BSDFMath::CosTheta(wo_Local) > 0.f) ? (shaderParams.ior / 1.f) : (1.f / shaderParams.ior);
		float sqrtDenom = etaT_etaI * dot(outwi_Local, m) + dot(wo_Local, m);
		float cosTerm = fabsf(dot(outwi_Local, m)) * fabsf(dot(wo_Local, m)) / 
			            (BSDFMath::AbsCosTheta(wo_Local) * BSDFMath::AbsCosTheta(outwi_Local));
		bsdfValue = make_float4(
			GGX_D(m, shaderParams) * Smith_G_Sep(wo_Local, outwi_Local, m, shaderParams) * (1.f - F) * cosTerm / (sqrtDenom * sqrtDenom));

		/*jacobian determinant for half-direction transformation*/
		//float sqrtDenom = dot(wo_Local, m) + etaT_etaI * dot(outwi_Local, m);
		dwh_dwo = fabsf(etaT_etaI * etaT_etaI * dot(outwi_Local, m)) / (sqrtDenom * sqrtDenom);

		/*if (bsdfValue.x == 0.f)
			rtPrintf("SamplefRefract--1-F:%f D:%f G:%f etaI/T:%f cosTerm:%f sqrtDenom:%f\n", 1.f - F, GGX_D(m, shaderParams), Smith_G_Sep(wo_Local, outwi_Local, m, shaderParams),
				etaT_etaI, cosTerm, sqrtDenom);*/
	}

	outPdf *= fabsf(dwh_dwo);/* outPdf = p(wm) * F * dwh_dwo */

	/*convert wi to world space*/
	outwi_World = TwUtil::BSDFMath::LocalToWorld(outwi_Local, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

	return bsdfValue;
}