#pragma once
#ifndef COLVILLEA_DEVICE_SHADER_MICROSURFACE_H_
#define COLVILLEA_DEVICE_SHADER_MICROSURFACE_H_
#include <optix_world.h>
#include <optix_device.h>
#include <optixu_math_namespace.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"

using namespace optix;
using namespace TwUtil;

/**
 * @brief normal distribution function(NDF) for isotropic GGX
 * @param m microfacet normal
 * @return the value of evaluating D(m) for isotropic GGX
 */
static __device__ __inline__ float GGX_D(const float3 &m, const CommonStructs::ShaderParams &shaderParams)
{
	/*ensure the orientation is consistent with the macrosurface normal,
	 *enabling us to leave out the heaviside function in D(m)*/
	if (BSDFMath::CosTheta(m) <= 0.f)
		return 0.f;

	float CosThetaSqr = BSDFMath::Cos2Theta(m);
	float alphaSqr = shaderParams.alphax * shaderParams.alphax;
	float value = 0.f;

	float denom = (1 + BSDFMath::Tan2Theta(m) / alphaSqr);
	value = 1.f / (M_PIf * alphaSqr * CosThetaSqr * CosThetaSqr * denom * denom);

	/*potential numeric issues preventing*/
	if (value * BSDFMath::CosTheta(m) < 1e-20f)
		value = 0.f;
	return value;
}

#if 0


//deprecated
static __device__ __inline__ float GGX_Lambda(const float3 &w, const ShaderParams &shaderParams)
{
	/*float roughness = 0.0f;
	if (shaderParams.RoughnessID != RT_TEXTURE_ID_NULL)
	{
		roughness = rtTex2D<float4>(shaderParams.RoughnessID, shaderParams.dgShading.uv.x, shaderParams.dgShading.uv.y).x;
	}
	else*/
	float roughness = shaderParams.alphax;

	//todo:think we could ignore fabs here:
	const float absTanTheta = fabs(BSDFMath::TanTheta(w));
	//todo:review
	/*if (absTanTheta == 0.f)
		return 1.f;*/
	if (isinf(absTanTheta))
	{
#ifdef __CUDACC__
		rtPrintf("GGX_Lambda() get infinite absTanTheta value:%f\n", absTanTheta);
#endif
		return 0.f;
	}

	float tempSqrt = 1 + (roughness * absTanTheta) * (roughness * absTanTheta);
	return (-1 + sqrtf(tempSqrt)) / 2.f;
}

/*Height-Correlated Masking and Shadowing Smith_G:under construction*/
static __device__ __inline__ float Smith_G(const float3 &wo, const float3 &wi, const ShaderParams &shaderParams)
{
	rtPrintf("Smith_G");
	return 1 / (1 + GGX_Lambda(wo, shaderParams) + GGX_Lambda(wi, shaderParams));
}
#endif

/**
 * @brief Smith's masking and shadowing function G1(w,wh) for isotropic GGX
 * @param v view direction(use in heaviside function)
 * @param m microfacet normal
 * @return the value of evaluating G1(w,wh) for isotropic GGX
 * @see Smith_G_Sep()
 */
static __device__ __inline__ float Smith_G1(const float3 &v, const float3 &m, const CommonStructs::ShaderParams &shaderParams, bool print = false)
{
	//if (print)
	//{
	//	//print v=wi_local
	//	/*rtPrintf("wi:%f %f %f m:%f %f %f dot(wi,m):%f cosTheta(wi):%f\n",
	//		v.x, v.y, v.z, m.x, m.y, m.z, dot(v, m), BSDFMath::CosTheta(v));*/
	//}

	/*ensure consistent orientation(inspired from mitsuba)*/
	if (dot(v, m) * BSDFMath::CosTheta(v) <= 0)
		return 0.0f;

	/* Perpendicular incidence -- no shadowing/masking */
	float tanTheta = fabsf(BSDFMath::TanTheta(v));
	if (tanTheta == 0.0f)
		return 1.0f;

	float root = shaderParams.alphax * tanTheta;
	return 2.0f / (1.0f + sqrtf(1.f + root*root));
}

/*Separable Masking and Shadowing Smith_G function*/
static __device__ __inline__ float Smith_G_Sep(const float3 &wo, const float3 &wi, const float3 &m, const CommonStructs::ShaderParams &shaderParams, bool print = false)
{
	/*if(print)
		rtPrintf("%f %f woloc:%f %f %f m:%f %f %f wiloc:%f %f %f\n", Smith_G1(wo, m, shaderParams), Smith_G1(wi, m, shaderParams),
			wo.x,wo.y,wo.z,m.x,m.y,m.z,wi.x,wi.y,wi.z);*/
	return Smith_G1(wo, m, shaderParams) * Smith_G1(wi, m, shaderParams, print);
}

/**
 * @brief compute the probability density function for sampling all microfacet normals with GGX distribution
 * @param m microfacet normal
 * @param shaderParams ShaderParams of the intersection
 * @return the value of D(m)*abs(dot(m,n))
 */
static __device__ __inline__ float SamplingMicrofacetNormal_Pdf(const float3 &m, const CommonStructs::ShaderParams & shaderParams)
{
                            /*using BSDFMath::CosTheta(m) is okay*/
	return GGX_D(m, shaderParams) * BSDFMath::AbsCosTheta(m);
}

/**
 * @brief compute the probability density function for sampling visible normal of distribution function(VNDF) with GGX distribution
 * @param m microfacet normal
 * @param wo_Local viewing direction which lying on the upper hemisphere(flip forehand if necessary)
 * @param shaderParams ShaderParams of the intersection
 * @return the value of D(m)*abs(dot(m,n))
 */
static __device__ __inline__ float SamplingMicrofacetNormal_Pdf(const float3 &wo_Local, const float3 & m, const CommonStructs::ShaderParams & shaderParams)
{
	if (BSDFMath::CosTheta(wo_Local) == 0.f)
		return 0.0f;
	return Smith_G1(wo_Local, m, shaderParams) * fabsf(dot(wo_Local, m)) * GGX_D(m, shaderParams) / BSDFMath::AbsCosTheta(wo_Local);
}

//deprecated
//static __device__ __inline__ float SamplingMicrofacetNormal_Pdf(const float3 & wo_Local, const float3 & wh_Local, const ShaderParams & shaderParams)
//{
//	rtPrintf("SamplingMicrofacetNormal_Pdf");
//#ifdef USE_SAMPLING_NORMAL
//	//Importance sampling microfacet normal, i.e. half-direction vector wh--X wrong??
//	// 
//	//p_m(m) = D(m) * dot(m,n)
//	return GGX_D(wh_Local, shaderParams) * BSDFMath::AbsCosTheta(wh_Local);
//#endif
//#ifdef USE_SAMPLING_VISIBLE_NORMAL
//	//deleteme:and review
//	/*if (BSDFMath::CosTheta(wo_Local) == 0.f)
//		return 0.f;
//	if (fabsf(dot(wh_Local, wo_Local)) * BSDFMath::CosTheta(wo_Local) <= 0.f)
//		return 0.f;*/
//	float G1 = 1.f / (1 + GGX_Lambda(wo_Local, shaderParams));
//
//	return GGX_D(wh_Local, shaderParams) * G1 * fabsf(dot(wo_Local, wh_Local)) / BSDFMath::AbsCosTheta(wo_Local);
//#endif
//}


__inline__ __device__ void TrowbridgeReitzSample11(float cosTheta, float U1, float U2,
	float *slope_x, float *slope_y) {
	// special case (normal incidence)
	if (cosTheta > .9999) {
		float r = sqrt(U1 / (1 - U1));
		float phi = 6.28318530718 * U2;
		*slope_x = r * cos(phi);
		*slope_y = r * sin(phi);
		return;
	}

	float sinTheta =
		sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));
	float tanTheta = sinTheta / cosTheta;
	float a = 1 / tanTheta;
	float G1 = 2 / (1 + std::sqrt(1.f + 1.f / (a * a)));

	// sample slope_x
	float A = 2 * U1 / G1 - 1;
	float tmp = 1.f / (A * A - 1.f);
	if (tmp > 1e10) tmp = 1e10;
	float B = tanTheta;
	float D = sqrtf(
		fmaxf(float(B * B * tmp * tmp - (A * A - B * B) * tmp), float(0)));
	float slope_x_1 = B * tmp - D;
	float slope_x_2 = B * tmp + D;
	*slope_x = (A < 0 || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

	// sample slope_y
	float S;
	if (U2 > 0.5f) {
		S = 1.f;
		U2 = 2.f * (U2 - .5f);
	}
	else {
		S = -1.f;
		U2 = 2.f * (.5f - U2);
	}
	float z =
		(U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
		(U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
	*slope_y = S * z * std::sqrt(1.f + *slope_x * *slope_x);

	assert(!isinf(*slope_y));
	assert(!isnan(*slope_y));
}


__device__ __inline__ float3 TrowbridgeReitzSample(const float3 &wi, float alpha_x,
	float alpha_y, float U1, float U2) {
	// 1. stretch wi
	float3 wiStretched =
		safe_normalize(make_float3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

	// 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
	float slope_x, slope_y;
	TrowbridgeReitzSample11(BSDFMath::CosTheta(wiStretched), U1, U2, &slope_x, &slope_y);

	// 3. rotate
	float tmp = BSDFMath::CosPhi(wiStretched) * slope_x - BSDFMath::SinPhi(wiStretched) * slope_y;
	slope_y = BSDFMath::SinPhi(wiStretched) * slope_x + BSDFMath::CosPhi(wiStretched) * slope_y;
	slope_x = tmp;

	// 4. unstretch
	slope_x = alpha_x * slope_x;
	slope_y = alpha_y * slope_y;

	// 5. compute normal
	return safe_normalize(make_float3(-slope_x, -slope_y, 1.));
}

//static __device__ __inline__ float3 SamplingMicrofacetNormal_Sample_wh(const float3 &wo_Local,
//	const float2 & urand, const ShaderParams & shaderParams)
//{
//	//deleteme
//	rtPrintf("SamplingMicrofacetNormal_Sample_wh");
//#ifdef USE_SAMPLING_NORMAL
//	float phi = 2 * M_PIf * urand.y;
//	float tanTheta2 = shaderParams.alphax * shaderParams.alphax * urand.x / (1.f - urand.x);
//
//	float cosTheta = 1 / sqrtf(1 + tanTheta2);
//	float sinTheta = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));
//
//	//convert theta,phi to cartesian coordinates
//	float3 wh_Local = make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
//
//	//in consist with wo_Local
//	if (!SameHemisphere(wo_Local, wh_Local))
//		wh_Local = -wh_Local;
//
//	return wh_Local;
//#endif
//#ifdef USE_SAMPLING_VISIBLE_NORMAL
//	/*[Heitz18]Sampling the GGX Distribution of Visible Normals*/
//	/*float roughness = 0.0f;
//	if (shaderParams.RoughnessID != RT_TEXTURE_ID_NULL)
//	{
//		roughness = rtTex2D<float4>(shaderParams.RoughnessID, shaderParams.dgShading.uv.x, shaderParams.dgShading.uv.y).x;
//	}
//	else*/
//	float roughness = shaderParams.alphax;
//
//	//todo:fix the possibly wo_Local lying on the lower hemisphere
//	bool flip = wo_Local.z < 0;
//	float3 wo_Local_flipped = flip ? -wo_Local : wo_Local;
//
//	 //Section 3.2: transforming the view direction to the hemisphere configuration
//	float3 Vh = safe_normalize(make_float3(roughness * wo_Local_flipped.x, roughness * wo_Local_flipped.y, wo_Local_flipped.z));
//	// Section 4.1: orthonormal basis
//	float3 T1 = (Vh.z < 0.9999) ? safe_normalize(cross(make_float3(0, 0, 1.f), Vh)) : make_float3(1, 0, 0);
//	float3 T2 = cross(Vh, T1);
//	// Section 4.2: parameterization of the projected area
//	float r = sqrtf(urand.x);
//	float phi = 2.0 * M_PIf * urand.y;
//	float t1 = r * cosf(phi);
//	float t2 = r * sinf(phi);
//	float s = 0.5 * (1.0 + Vh.z);
//	t2 = (1.0 - s)*sqrtf(1.0 - t1 * t1) + s * t2;
//
//	// Section 4.3: reprojection onto hemisphere
//	float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2))*Vh;
//
//	// Section 3.4: transforming the normal back to the ellipsoid configuration
//	float3 Ne = safe_normalize(make_float3(roughness * Nh.x, roughness * Nh.y, fmaxf(0.0, Nh.z)));
//
//	//deleteme:
//	if (flip)
//		Ne = -Ne;
//	
//	return Ne;
//
////	bool flip = wo_Local.z < 0;
//// 	if (flip)
//// 	{
//// 		if (sysLaunch_index == make_uint2(384))
//// 			rtPrintf("%f\n", wo_Local.z);
//// 	}
////	float3 wh = TrowbridgeReitzSample(flip ? -wo_Local : wo_Local, shaderParams.alphax, shaderParams.alphax, urand.x, urand.y);
////	if (flip) wh = -wh;
//// 	if (sysLaunch_index == make_uint2(384))
//// 		rtPrintf("%f %f %f\n", wh.x, wh.y, wh.z);
//	
////	return wh;
//#endif
//}


/**
 * @brief Sample microfacet normal with isotropic GGX distribution, note that this version sample all the microfacet normal without accounting for the visiblity of microfacet normal with respect to the viewing direction.
 * @param urand random numbers in [0,1]
 * @param shaderParams ShaderParam of the intersection
 * @param outPdf pdf of the sampled direction
 * @return M, the microfacet normal sampled from D(m)*dot(m,n) which is always on the upper hemisphere.
 */
static __device__ __inline__ float3 SamplingMicrofacetNormal_Sample_wh(const float2 & urand, const CommonStructs::ShaderParams & shaderParams, float & outPdf)
{
	float cosThetaM = 0.0f;
	float sinPhiM = 0.0f, cosPhiM = 0.0f;
	float alphaSqr = shaderParams.alphax * shaderParams.alphax;

	/*isotropic sampling from GGX*/

	/*sample phi component*/
	sinPhiM = sinf(2.f * M_PIf * urand.y);
	cosPhiM = cosf(2.f * M_PIf * urand.y);

	/*sample theta component*/
	float tanThetaMSqr = alphaSqr * urand.x / (1.f - urand.x);
	cosThetaM = 1.f / sqrtf(1.f + tanThetaMSqr);

	/*compute the pdf of the sampled direction*/
	float tmp = 1.f + tanThetaMSqr / alphaSqr;
	/*note that the absoloute cosThetaMSqr ensures the dot(wm,wg) > 0, enabling us to leave out the heaviside term in pdf evaluation*/
	outPdf = 1.f / (M_PIf * alphaSqr * cosThetaM * cosThetaM * cosThetaM * tmp * tmp);

	/*prevent numerical issues*/
	if (outPdf < 1e-20f)
		outPdf = 0.f;

	float sinThetaM = sqrtf(fmaxf(0.f, 1.f - cosThetaM * cosThetaM));
	/*convert to spherical direction*/
	return make_float3(sinThetaM * cosPhiM, sinThetaM * sinPhiM, cosThetaM);
}


/**
 * @brief Sample microfacet normal with isotropic GGX distribution, this version samples visible normal of distribution function(VNDF) and reduces firfly samples greatly.
 * @param urand random numbers in [0,1]
 * @param wo_Local the viewing direction
 * @param shaderParams ShaderParam of the intersection
 * @return M, the microfacet normal sampled from G1(wh)*fmaxf(dot(wh,wo))*D(wh)/wo.z which is always on the upper hemisphere.
 */
static __device__ __inline__ float3 SamplingMicrofacetNormal_Sample_wh(const float3 &wo_Local, const float2 & urand, const CommonStructs::ShaderParams & shaderParams)
{
	/*Reference:
	 *Journal of Computer Graphics Techniques vol.7, no.4,2018
	 *Sampling the GGX Distribution of Visible Normals*/

	float roughness = shaderParams.alphax;

	//Section 3.2: transforming the view direction to the hemisphere configuration
	float3 Vh = safe_normalize(make_float3(roughness * wo_Local.x, roughness * wo_Local.y, wo_Local.z));
	// Section 4.1: orthonormal basis
	float3 T1 = (Vh.z < 0.9999) ? safe_normalize(cross(make_float3(0, 0, 1.f), Vh)) : make_float3(1, 0, 0);
	float3 T2 = cross(Vh, T1);
	// Section 4.2: parameterization of the projected area
	float r = sqrtf(urand.x);
	float phi = 2.0 * M_PIf * urand.y;
	float t1 = r * cosf(phi);
	float t2 = r * sinf(phi);
	float s = 0.5 * (1.0 + Vh.z);
	t2 = (1.0 - s)*sqrtf(1.0 - t1 * t1) + s * t2;

	// Section 4.3: reprojection onto hemisphere
	float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2))*Vh;

	// Section 3.4: transforming the normal back to the ellipsoid configuration
	float3 Ne = safe_normalize(make_float3(roughness * Nh.x, roughness * Nh.y, fmaxf(0.0, Nh.z)));
	return Ne;
}

#endif // COLVILLEA_DEVICE_SHADER_MICROSURFACE_H_