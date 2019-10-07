#include "MicrofacetBRDF.h"

using namespace optix;
using namespace CommonStructs;

/*This is a single-sided material*/
rtDeclareVariable(uint2, sysLaunch_index, rtLaunchIndex, );

rtDeclareVariable(float, eta0, , ) = 1.49f;
rtDeclareVariable(float, alpha0, , ) = 0.1f;
rtDeclareVariable(float, alpha1, , ) = 0.001f;//0.01/0.001?
rtDeclareVariable(float3, eta1, , );
rtDeclareVariable(float3, kappa1, , );


/*Reference: Efficient Rendering of Layered Materials
            using an Atomic Decomposition with Statistical Operators by Laurent Belcour,
            ACM Transactions on Graphics, Vol.37, No.4, Article 73.
  FrostedMetal BSDF is inspired by the paper and implements a simplified
 and effiecient layered-BSDF.
 
 ---Work in Progress---
 */

static __device__ __inline__ float averageRGBSpectrum(const float3 &r)
{
	return (1.f / 3.f) * (r.x + r.y + r.z);
}
static __device__ __inline__ bool isZero(float3 &r)
{
	return r.x == 0.f && r.y == 0.f && r.z == 0.f;
}
static __device__ __inline__ float4 float3Tofloat4(float3 r)
{
	return make_float4(r.x, r.y, r.z, 0.0f);
}

static __device__ __inline__ float roughnessToVariance(float a) 
{
#ifdef USE_BEST_FIT
	a = _clamp<float>(a, 0.0, 0.9999);
	float a3 = powf(a, 1.1);
	return a3 / (1.0f - a3);
#else
	return a / (1.0f - a);
#endif
}
static __device__ __inline__ float varianceToRoughness(float v) 
{
#ifdef USE_BEST_FIT
	return powf(v / (1.0f + v), 1.0f / 1.1f);
#else
	return v / (1.0f + v);
#endif
}

//static __device__ __inline__ float4 evalFGD(float3 &wi, float alpha)
//{
//	float3 wo;
//	float2 rand = make_float2(.5f, .5f);
//	float outPdf;
//	ShaderParams _shaderParams; _shaderParams.alphax = alpha;
//	return MicrofacetReflection_InnerSample_f(wi, wo, rand, outPdf, _shaderParams, true) * BSDFMath::AbsCosTheta(wi) / outPdf;
//}

/**
 * @brief Compute moment statistics for given input direction wi.
 * For frostedMetal BSDF, we assume it's two-layered. The top layer
 * is dielectric and the bottom layer is conductor.
 * @param wi input direction in local shading space
 * @param eta0,alpha0 top dielectric layer
 * @param eta1,kappa1,alpha1 bottom metal layer
 * @param energyCoeffs points to an array of size of two to store
 * energy coefficients
 * @param alphas points to an array of size of two to store alphas
 * @param cosThetas points to an array of size of two to store 
 * Not available now!
 * computed mean direction(which is different from original implementation)
 */
static __device__ __inline__ void computeAddingDoubling(float3 wi, float eta0, float alpha0, 
	                                                    float3 eta1, float3 kappa1, float alpha1,
	                                                    float3 *energyCoeffs, float *alphas,
	                                                    float *cosThetas)
{
	float cosThetaI = BSDFMath::AbsCosTheta(wi);//CosTheta(wi)

	float cosThetaT = 0.f;
	/*note that eta=eta2/eta1,eta2 = eta0(i.e. dielectric layer), eta1 = air*/
	float n12 = eta0 / 1.f;

	/*Evaluate off-specular transmission direction for the first layer*/
	float sinThetaI = sqrtf(optix::clamp(1.f - cosThetaI * cosThetaI, 0.0f, 1.0f));//prevent from numeric issue leading to NaN
	float sinThetaT = sinThetaI / n12;//Snell's law
	if (sinThetaT <= 1.f)
	{
		cosThetaT = sqrtf(1.f - sinThetaT * sinThetaT);
		cosThetas[0] = cosThetaI;
		cosThetas[1] = cosThetaT;
	}
	else
	{
		//Total Internal Reflection
		cosThetaT = -1.f;
	}

	/*Ray is not blocked by Total Internal Reflection -- undoubtedly*/
	const bool has_transimssion = cosThetaT > 0.f;
	if (!has_transimssion)
		rtPrintf("assert failed at has_transmission!\n");

	/*Evaluate variance for reflection operator*/
	float s_r12 = roughnessToVariance(alpha0);
	float s_r21 = s_r12;
	float s_r23 = roughnessToVariance(alpha1);

	/*Evaluate variance for refraction operator */
	float s_t12 = 0.f, s_t21 = 0.f, j21 = 0.f;
	//float j12 = 0.f;
	if (has_transimssion)
	{
		/*From author's original implementation, haven't figured out yet(Different from paper).*/
#if 0
		const float _ctt = 1.0f; // The scaling factor overblurs the BSDF at grazing
		const float _cti = 1.0f; // angles (we cannot account for the deformation of
								 // the lobe for those configurations.

		s_t12 = roughnessToVariance(alpha0 * 0.5f * fabs(_ctt*n12 - _cti) / (_ctt*n12));
		s_t21 = roughnessToVariance(alpha0 * 0.5f * fabs(_cti / n12 - _ctt) / (_cti / n12));
#else
		s_t12 = roughnessToVariance(alpha0 * 0.5f * fabs(cosThetaT*n12 - cosThetaI) / (cosThetaT*n12));
		s_t21 = roughnessToVariance(alpha0 * 0.5f * fabs(cosThetaI / n12 - cosThetaT) / (cosThetaI / n12));//id827_1616 seems good

		//s_t12 = roughnessToVariance(alpha0 * 0.5f * fabsf(1.f+(cosThetaI/cosThetaT)*n12));
		//s_t21 = roughnessToVariance(alpha0 * 0.5f * fabsf(1.f+(cosThetaT/cosThetaI)/n12));//id827_1619 over_blurs
#endif
		j21 = (cosThetaI / cosThetaT) / n12;
	}

	/*Evaluate reflection/transmission coefficient for energy(and variance) adding-doubling, todo:use FGD instead of Fresnel term only*/
	float _cosThetaT;
	float R12 = FresnelDielectricExt(cosThetaI, _cosThetaT, n12);
	float R21, T12, T21;
	float3 Eta1 = make_float3(1.f / eta0);
	float3 Kappa1 = kappa1 / eta0;
	//float3 R23 = FresnelConductor(cosThetaT, eta1, kappa1);
	float3 R23 = FresnelConductor(cosThetaT, Eta1, Kappa1);

	/*Asserts*/
	if (cosThetaT < 0.f)
	{
		rtPrintf("assert failed at cosThetaT\n");
	}
// 	if (fabsf(cosThetaT - _cosThetaT) > 1e-20f)
// 	{
// 		rtPrintf("assert failed at |cosThetaT-_cosThetaT| %f %f\n", cosThetaT, _cosThetaT);
// 	}

	if (has_transimssion)
	{
		R21 = R12;
		T21 = T12 = 1.f - R12;
	}
	else
	{
		T21 = R21 = T12 = 0.f;
	}

	/*Top dielectric layer is always left unchanged*/
	energyCoeffs[0] = make_float3(R12);
	alphas[0] = alpha0;

#if 0 //typo
	/*Lower metal layer's energy and alpha computed using adding-doubling method*/
	float3 multipleScatteringTerm = (R23 * R21) / (1.f - R23 * R21);
	energyCoeffs[1] = (T12 * multipleScatteringTerm);
	alphas[1] = varianceToRoughness(s_t12 + j21 * (s_t21 + s_r23 + (s_r21 + s_r23) * averageRGBSpectrum(multipleScatteringTerm)));//different from author's implementation, possibly fix the typo for s_r13 formula, see equation (38) and equation (51) of the paper.
#endif

	/*Lower metal layer's energy and alpha computed using adding-doubling method*/
	float3 multipleScatteringTerm = (R23) / (1.f - R23 * R21);
	energyCoeffs[1] = (T12 * multipleScatteringTerm * T21);
	alphas[1] = varianceToRoughness(s_t12 + j21 * (s_t21 + s_r23 + (s_r21 + s_r23) * averageRGBSpectrum(R21 * multipleScatteringTerm)));//different from author's implementation, possibly fix the typo for s_r13 formula, see equation (38) and equation (51) of the paper.
	//alphas[1] = varianceToRoughness(s_t21 + j12 * (s_t12 + s_r23 + (s_r21 + s_r23) * averageRGBSpectrum(R21 * multipleScatteringTerm)));//original impl
}


/**
 * @brief Evaluate Dielectric-Conductor layered BSDF given in/out direction
 * using adding-doubling method.
 * @param wo_Local out direction in local space
 * @param wi_Local in direction in local space
 * @param shaderParams ShaderParams relates to material
 * @return value of layered BSDF
 */
static __device__ __inline__ float3 FrostedMetal_InnerEval_f(const float3 & wo_Local, const float3 & wi_Local, const ShaderParams & shaderParams)
{
	/*one-sided BSDF detection*/
	if (BSDFMath::CosTheta(wo_Local) <= 0.f || BSDFMath::CosTheta(wi_Local) <= 0.f)
		return make_float3(0.f);

	/*Compute microfacet normal*/
	float3 H = safe_normalize(wo_Local + wi_Local);

	/*Result BSDF*/
	float3 f = make_float3(0.f);

	/*Trace statistics moment using adding-doubling*/
    float3 coeffs[2];
	float alphas[2];
	float cosThetas[2];
	computeAddingDoubling(wo_Local, eta0, alpha0, eta1, kappa1, alpha1, coeffs, alphas, cosThetas);

	/*Instantiate and evaluate BSDF lobes from moment statistics*/
	for (int i = 0; i < 2; ++i)
	{
		if(isZero(coeffs[i]))
			continue;

		const float alpha = alphas[i];
		ShaderParams _shaderParams; _shaderParams.alphax = alpha;

		const float D = GGX_D(H, _shaderParams);
		const float G = Smith_G_Sep(wo_Local, wi_Local, H, _shaderParams);//why not track mean and convert to wk?

		// Add to the contribution
		f += D * G * coeffs[i] / (4.0f * BSDFMath::CosTheta(wo_Local) * BSDFMath::CosTheta(wi_Local));
	}

	if (isnan(f.x)||isnan(f.y)||isnan(f.z))
		rtPrintf("assert failed at nan_f\n");
	return f;
}

static __device__ __inline__ float FrostedMetal_InnerPdf(const float3 & wo_Local, const float3 & wi_Local, const ShaderParams & shaderParams)
{
	/*one-sided BSDF detection*/
	if (BSDFMath::CosTheta(wo_Local) <= 0.f || BSDFMath::CosTheta(wi_Local) <= 0.f)
		return 0.f;

	/*Compute microfacet normal*/
	float3 H = safe_normalize(wo_Local + wi_Local);

	/*Trace statistics moment using adding-doubling*/
	float3 coeffs[2];
	float alphas[2];
	float cosThetas[2];
	computeAddingDoubling(wo_Local, eta0, alpha0, eta1, kappa1, alpha1, coeffs, alphas, cosThetas);

	/* Convert Spectral coefficients to float for pdf weighting */
	float pdf = 0.0;
	float cum_w = 0.0;
	for (int i = 0; i < 2; ++i) 
	{
		// Skip zero contributions
		if (isZero(coeffs[i])) 
			continue;

		/* Evaluate weight */
		auto weight = averageRGBSpectrum(coeffs[i]);
		cum_w += weight;

		const float alpha = alphas[i];
		ShaderParams _shaderParams; _shaderParams.alphax = alpha;

		/* Evaluate the pdf */
		float DG = GGX_D(H, _shaderParams) * Smith_G1(wo_Local, H, _shaderParams) / (4.0f * BSDFMath::CosTheta(wo_Local));//todo:review G1(?wo_Local)
		pdf += weight * DG;
	}

	if (cum_w > 0.0f) 
	{
		return pdf / cum_w;
	}
	else 
	{
		return 0.0f;
	}
}

RT_CALLABLE_PROGRAM float4 FrostedMetal_Eval_f(const float3 & wo_World, const float3 & wi_World, const ShaderParams & shaderParams)
{
	if (TwUtil::dot(wi_World, shaderParams.nGeometry) * TwUtil::dot(wo_World, shaderParams.nGeometry) > 0)
	{//ignore btdf and evaluate brdf only
		float3 wo_Local = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
		float3 wi_Local = TwUtil::BSDFMath::WorldToLocal(wi_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

		return float3Tofloat4(FrostedMetal_InnerEval_f(wo_Local, wi_Local, shaderParams));
	}
	//else ignore brdf and btdf doesn't exist here, so we just return 0.f:
	return make_float4(0.f);
}

RT_CALLABLE_PROGRAM float FrostedMetal_Pdf(const float3 & wo_World, const float3 & wi_World, const ShaderParams & shaderParams)
{
	//Revising light leak/light spot issue.
	if (TwUtil::dot(wi_World, shaderParams.nGeometry) * TwUtil::dot(wo_World, shaderParams.nGeometry) > 0)
	{
		//evaluate BRDF only
		float3 wo_Local = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
		float3 wi_Local = TwUtil::BSDFMath::WorldToLocal(wi_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

		return FrostedMetal_InnerPdf(wo_Local, wi_Local, shaderParams);
	}
	return 0.f;
}


RT_CALLABLE_PROGRAM float4 FrostedMetal_Sample_f(const float3 &wo_World, float3 & outwi_World, float2 & urand, float & outPdf, float bsdfChoiceRand, const ShaderParams & shaderParams)
{
	float3 wo = TwUtil::BSDFMath::WorldToLocal(wo_World, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);
	float3 wi = make_float3(0.f);

	/*one-sided BSDF detection*/
	if (BSDFMath::CosTheta(wo) <= 0.f)
		return make_float4(0.f);

	/*trace statistics moment using adding-doubling method*/
	float3 coeffs[2];
	float alphas[2];
	float cosThetas[2];
	computeAddingDoubling(wo, eta0, alpha0, eta1, kappa1, alpha1, coeffs, alphas, cosThetas);

	/* Convert Spectral coefficients to floats to select BRDF lobe to sample */
	float weights[2];
	float cum_w = 0.0;
	for (int i = 0; i < 2; ++i) 
	{
		weights[i] = averageRGBSpectrum(coeffs[i]);
		cum_w += weights[i];
	}

	/* Select a random BRDF lobe based on energy */
	float sel_w = bsdfChoiceRand * cum_w - weights[0];
	int   sel_i = 0;
	for (sel_i = 0; sel_w > 0.0 && sel_i < 2; ++sel_i) 
	{
		sel_w -= weights[sel_i + 1];
	}

	/* Sample a microfacet normal */
	const float alpha = alphas[sel_i];
	ShaderParams _shaderParams; _shaderParams.alphax = alpha;

	float3 m = SamplingMicrofacetNormal_Sample_wh(wo, urand, _shaderParams);
	outPdf = SamplingMicrofacetNormal_Pdf(TwUtil::signum(BSDFMath::CosTheta(wo))*wo, m, _shaderParams);

	/* Perfect specular reflection based on the microfacet normal */
	wi = TwUtil::reflect(wo, m);
	if (wi.z <= 0.0f || outPdf <= 0.0f)
	{
		return make_float4(0.0f);
	}

	/* Evaluate the MIS 'pdf' using the balance heuristic */
	outPdf = 0.f;//note that this is important for initializing the pdf's value
	for (int i = 0; i < 2; ++i) 
	{
		const float alpha = alphas[i];
		ShaderParams _shaderParams; _shaderParams.alphax = alpha;
		float DG = GGX_D(m, _shaderParams) * Smith_G1(wo, m, _shaderParams) / (4.0f * BSDFMath::CosTheta(wo));
		outPdf += (weights[i] / cum_w) * DG;
	}

	/* Evaluate the BRDF and the final weight */
	float4 R = float3Tofloat4(FrostedMetal_InnerEval_f(wo, wi, shaderParams));

	outwi_World = TwUtil::BSDFMath::LocalToWorld(wi, shaderParams.dgShading.dpdu, shaderParams.dgShading.tn, shaderParams.dgShading.nn);

	if (outPdf > 0.0f)
	{
		return R;
	}
	else 
	{
		return make_float4(0.0f);
	}
}