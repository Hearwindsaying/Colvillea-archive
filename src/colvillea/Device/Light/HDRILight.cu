#include <optix_world.h>/*for uint16_t support*/
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"
#include "LightUtil.h"

using namespace optix;
using namespace TwUtil;

//////////////////////////////////////////////////////////////////////////
//Forward declarations:
// HDRILight:->Context
#ifndef TWRT_DELCARE_LIGHTBUFFER
#define TWRT_DELCARE_LIGHTBUFFER
rtDeclareVariable(CommonStructs::LightBuffers, sysLightBuffers, , );
#endif

#ifndef TWRT_DECLARE_SYSLAUNCH
#define TWRT_DECLARE_SYSLAUNCH
rtDeclareVariable(uint2, sysLaunch_Dim,   rtLaunchDim, );
rtDeclareVariable(uint2, sysLaunch_index, rtLaunchIndex, );
#endif

//prefiltering HDRIMap:->Context
rtDeclareVariable(CommonStructs::HDRIEnvmapLuminanceBufferWrapper, sysHDRIEnvmapLuminanceBufferWrapper, , );     


//////////////////////////////////////////////////////////////////////////
//Util Functions:
static __device__ __inline__ float SampleContinuous1D_pMarginal(float urand, float &outPdf, int &outOffsetpMarginal)
{
    CommonStructs::Distribution1D &pMarginal = sysLightBuffers.hdriLight.distributionHDRI.pMarginal;
	rtBufferId<float, 1> &cdf = pMarginal.cdf;


	int offset = TwUtil::FindInterval(cdf.size(), [&](int index) {return cdf[index] <= urand; });
	outOffsetpMarginal = offset;

	float du = urand - cdf[offset];
	if ((cdf[offset + 1] - cdf[offset]) > 0)
		du /= (cdf[offset + 1] - cdf[offset]);

	if (isnan(du))
		rtPrintf("SampleContinuous1D_pMarignal has got an nan:du %f\n", du);

	outPdf = pMarginal.func[offset] / pMarginal.funcIntegral;

	if (pMarginal.funcIntegral <= 0)
		rtPrintf("SampleContinuous1D_pMarignal has got a exceptional number:funcIntegral %f\n", pMarginal.funcIntegral);

	return (offset + du) / pMarginal.func.size();
}

static __device__ __inline__ float SampleContinuous1D_pCondV(int rowInPCondV, float urand, float &outPdf)
{
	rtBufferId<float, 2> &pCondVcdf = sysLightBuffers.hdriLight.distributionHDRI.pConditionalV_cdf;
	rtBufferId<float, 2> &pCondVfunc = sysLightBuffers.hdriLight.distributionHDRI.pConditionalV_func;
	rtBufferId<float, 1> &pCondVfuncIntegral = sysLightBuffers.hdriLight.distributionHDRI.pConditionalV_funcIntegral;
	size_t pCondVcdfSize = pCondVcdf.size().x;
	size_t pCondVfuncSize = pCondVfunc.size().x;

#define pCondVcdf(index) pCondVcdf[make_uint2(index, rowInPCondV)]
#define pCondVfunc(index) pCondVfunc[make_uint2(index, rowInPCondV)]

	int offset = TwUtil::FindInterval(pCondVcdfSize, [&](int index) {return pCondVcdf(index) <= urand; });

	float du = urand - pCondVcdf(offset);
	if ((pCondVcdf(offset + 1) - pCondVcdf(offset)) > 0)
		du /= (pCondVcdf(offset + 1) - pCondVcdf(offset));

	if (isnan(du))
		rtPrintf("SampleContinuous1D_pCondV has got an nan:du %f\n", du);

	outPdf = pCondVfunc(offset) / pCondVfuncIntegral[rowInPCondV];

	if (pCondVfuncIntegral[rowInPCondV] <= 0)
		rtPrintf("SampleContinuous1D_pCondV has got a exceptional number:funcIntegral %f\n", pCondVfuncIntegral[rowInPCondV]);

	return (offset + du) / pCondVfuncSize;
#undef pCondVcdf
#undef pCondVfunc
}

static __device__ __inline__ float2 SampleContinuous2D(const float2 &urand, float &outPdf)
{
	float2 pdfs = make_float2(0.f);
	int v = 0;
	//sample con1D
	float2 sampledP = make_float2(0.f);
	sampledP.y = SampleContinuous1D_pMarginal(urand.y, pdfs.y, v);
	sampledP.x = SampleContinuous1D_pCondV(v, urand.x, pdfs.x);
	outPdf = pdfs.x * pdfs.y;
	return sampledP;
}

static __device__ __inline__ float SampleContinuous2D_Pdf(const float2 &p)
{
	//convert continuous coordinate p into corresponding discrete coordinate (u,v)
	const int maxU = sysLightBuffers.hdriLight.distributionHDRI.pConditionalV_func.size().x;
	const int maxV = sysLightBuffers.hdriLight.distributionHDRI.pMarginal.func.size();
	int iu = optix::clamp(static_cast<int>(p.x * maxU), 0, maxU - 1);
	int iv = optix::clamp(static_cast<int>(p.y * maxV), 0, maxV - 1);

	return sysLightBuffers.hdriLight.distributionHDRI.pConditionalV_func[make_uint2(iu, iv)] / sysLightBuffers.hdriLight.distributionHDRI.pMarginal.funcIntegral;
}


//////////////////////////////////////////////////////////////////////////
//HDRILight SampleLd function:
RT_CALLABLE_PROGRAM float4 Sample_Ld_HDRI(const float3 &point, const float & rayEpsilon, float3 & outwi, float & outpdf, float2 lightSample, uint lightBufferIndex, Ray & outShadowRay)
{
	float mapPdf = 0.f;
	float2 uv = SampleContinuous2D(lightSample, mapPdf);
	if (mapPdf == 0)//outPdf?
		return make_float4(0.f);

	float theta = uv.y * M_PIf;
	float phi   = uv.x * 2 * M_PIf;

	float cosTheta = cosf(theta);	float sinTheta = sinf(theta);
	float cosPhi   = cosf(phi);		float sinPhi   = sinf(phi);

	outwi = xfmVector(make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta), sysLightBuffers.hdriLight.lightToWorld);

	outpdf = mapPdf / (2 * M_PIf * M_PIf * sinTheta);
	if (sinTheta == 0.f)
		outpdf = 0.f;

	outShadowRay = MakeShadowRay(point, rayEpsilon, outwi);
	return rtTex2D<float4>(sysLightBuffers.hdriLight.hdriEnvmap, uv.x, uv.y);
}

//Pdf function
RT_CALLABLE_PROGRAM float LightPdf_HDRI(const float3 & p, const float3 & wi, const int lightId, Ray &shadowRay)//todo:add shadowray
{
	//ignore WorldToLight
	float3 w_Light = xfmVector(wi, sysLightBuffers.hdriLight.worldToLight);
	float theta = TwUtil::sphericalTheta(w_Light), phi = TwUtil::sphericalPhi(w_Light);
	float sinTheta = sinf(theta);
	if (sinTheta == 0.f)
		return 0.f;
	return SampleContinuous2D_Pdf(make_float2(phi * M_1_PIf / 2.f, theta * M_1_PIf)) / (2.f * M_PIf * M_PIf * sinTheta);
}

//////////////////////////////////////////////////////////////////////////
//Prefiltering program for HDRILight:
RT_PROGRAM void RayGeneration_PrefilterHDRILight()
{
    float vp = static_cast<float>(sysLaunch_index.y) / sysLaunch_Dim.y;
    float up = static_cast<float>(sysLaunch_index.x) / sysLaunch_Dim.x;
    float sinTheta = sinf(M_PIf * (vp + .5f) / sysLaunch_Dim.y);

    sysHDRIEnvmapLuminanceBufferWrapper.HDRIEnvmapLuminanceBuffer[sysLaunch_index] =
        TwUtil::luminance(make_float3(rtTex2D<float4>(sysLightBuffers.hdriLight.hdriEnvmap, up, vp))) * sinTheta;

}