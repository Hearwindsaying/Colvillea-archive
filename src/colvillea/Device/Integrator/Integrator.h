#pragma once
#ifndef COLVILLEA_DEVICE_INTEGRATOR_INTEGRATOR_H_
#define COLVILLEA_DEVICE_INTEGRATOR_INTEGRATOR_H_
/* This file is device only. */
#include <optix_world.h>
#include <optix_device.h>
#include <optixu_math_namespace.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"
#include "../Toolkit/NvRandom.h"
#include "../Light/LightUtil.h"

#include "../Sampler/Sampler.h"




//////////////////////////////////////////////////////////////////////////
//Forward declarations:

//system variables:->Context
rtDeclareVariable(rtObject,				sysTopShadower, , );
rtDeclareVariable(rtObject,				sysTopObject, , );
rtDeclareVariable(float,                sysSceneEpsilon, , );

#ifndef TWRT_DELCARE_SAMPLERTYPE
#define TWRT_DELCARE_SAMPLERTYPE
rtDeclareVariable(int, sysSamplerType, , );         /* Sampler type chosen in GPU program.
                                                       --need to fetch underlying value from
                                                         CommonStructs::SamplerType. */
#endif

//lights:
#ifndef TWRT_DELCARE_POINTLIGHT
#define TWRT_DELCARE_POINTLIGHT
rtBuffer<CommonStructs::PointLight> pointLightBuffer;
#endif
#ifndef TWRT_DELCARE_QUADLIGHT
#define TWRT_DELCARE_QUADLIGHT
rtBuffer<CommonStructs::QuadLight> quadLightBuffer;
#endif

rtDeclareVariable(int, 					hdriEnvmap, ,);

rtDeclareVariable(CommonStructs::PerRayData_radiance,  prdRadiance,     rtPayload, );
rtDeclareVariable(CommonStructs::PerRayData_shadow,	prdShadow,	     rtPayload, );
rtDeclareVariable(Ray,					ray,		     rtCurrentRay, );
rtDeclareVariable(float,				tHit,		     rtIntersectionDistance, );

#ifndef TWRT_DECLARE_HDRILIGHT
#define TWRT_DECLARE_HDRILIGHT
rtDeclareVariable(Distribution2D, hdriLightDistribution, , );
rtDeclareVariable(HDRILight, hdriLight, , );
#endif

#ifndef TWRT_DECLARE_SYS_ITERATION_INDEX
#define TWRT_DECLARE_SYS_ITERATION_INDEX
rtDeclareVariable(uint, sysIterationIndex, , ) = 0;
#endif

#ifndef TWRT_DECLARE_SYSLAUNCH
#define TWRT_DECLARE_SYSLAUNCH
rtDeclareVariable(uint2, sysLaunch_Dim, rtLaunchDim, );
rtDeclareVariable(uint2, sysLaunch_index, rtLaunchIndex, );
#endif


//Bindless BSDF callable programs:->Context
rtBuffer< rtCallableProgramId<
    float4(const float3 & wo_World, const float3 & wi_World, const CommonStructs::ShaderParams & shaderParams)> > Eval_f;
rtBuffer< rtCallableProgramId<
    float4(const float3 &wo_World, float3 & outwi_World, float2 & urand, float & outPdf, float bsdfChoiceRand, const CommonStructs::ShaderParams & shaderParams)> >
    Sample_f;
rtBuffer< rtCallableProgramId<
    float(const float3 & wo_World, const float3 & wi_World, const CommonStructs::ShaderParams & shaderParams)> >
    Pdf;

//Bindless Light callable programs:->Context
rtBuffer< rtCallableProgramId<
    float4(const float3 &point, const float & rayEpsilon, float3 & outwi, float & outpdf, float2 lightSample, uint lightBufferIndex, Ray & outShadowRay)> > 
    Sample_Ld;
rtBuffer< rtCallableProgramId<
    float(const float3 & p, const float3 & wi, const int lightId, Ray &shadowRay)> >
    LightPdf;


//////////////////////////////////////////////////////////////////////////
//Common integrator utility functions:

/**
 * @brief Estimate direct lighting contribution to the surface
 * interaction. For performance and implementation consideration,
 * supported light types in Colvillea should provide partial 
 * template specification in accordance with individual 
 * CommonStructs::LightType. It's also recommended that divide
 * light type into HDRILight, AreaLight and Dirac (Delta) Light
 * for more generic programming. 
 * 
 * @param[in] lightId index to light buffer, only avaliable when
 * lightType is not HDRILight
 * @param[in] shaderParams material and differential geometry information
 * @param[in] isectP       intersection point in world space
 * @param[in] isectDir     incoming direction in world space, pointing out of the surface
 * @param[in] localSampler local sampler binded to current launch
 * 
 * @return Return direct lighting contribution due to one specific
 * light.
 */
template<CommonStructs::LightType lightType>
static __device__ __inline__ float4 EstimateDirectLighting(
    int lightId,
    const CommonStructs::ShaderParams & shaderParams,
    const float3 & isectP, const float3 & isectDir,
    GPUSampler &localSampler);

template<>
static __device__ __inline__ float4 EstimateDirectLighting<CommonStructs::LightType::HDRILight>(
    int lightId,
    const CommonStructs::ShaderParams & shaderParams,
    const float3 & isectP, const float3 & isectDir,
    GPUSampler &localSampler)
{
    float4 Ld = make_float4(0.f);

    float3 p = isectP;
    float3 wo_world = isectDir;
    float sceneEpsilon = sysSceneEpsilon;

    float3 outWi = make_float3(0.f);
    float lightPdf = 0.f, bsdfPdf = 0.f;
    Ray shadowRay;

    /* Sample light source with Mulitple Importance Sampling. */
    float2 randSamples = Get2D(&localSampler);

    float4 Li = Sample_Ld[toUnderlyingValue(CommonStructs::LightType::HDRILight)](p, sceneEpsilon, outWi, lightPdf, randSamples, lightId, shadowRay);
    if (lightPdf > 0.f && !isBlack(Li))
    {
        /* Compute BSDF value using sampled outWi from sampling light source. */
        float4 f = Eval_f[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, shaderParams);

        if (!isBlack(f))
        {
            /* Trace shadow ray and find out its visibility. */
            CommonStructs::PerRayData_shadow shadow_prd;
            shadow_prd.blocked = 0;

            rtTrace<CommonStructs::PerRayData_shadow>(sysTopShadower, shadowRay, shadow_prd);

            if (!shadow_prd.blocked)
            {
                /* Compute Ld using MIS weight. */

                bsdfPdf = Pdf[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, shaderParams);
                float weight = TwUtil::MonteCarlo::PowerHeuristic(1, lightPdf, 1, bsdfPdf);
                Ld += f * Li * (fabs(dot(outWi, shaderParams.dgShading.nn)) * weight / lightPdf);

                /* Check exceptional value. */
                if (isnan(Ld.x) || isnan(Ld.y) || isnan(Ld.z))
                    rtPrintf("l107f:%f %f %f Li:%f %f %f cosTheta:%f weight:%f lightpdf:%f\n", f.x, f.y, f.z,
                        Li.x, Li.y, Li.z, (fabs(dot(outWi, shaderParams.dgShading.nn)), weight, lightPdf));
                if (weight == 0.f && lightPdf == 0.f)
                    rtPrintf("weirdBSDFPDF %f lightPdf %f weight %f!!!", bsdfPdf, lightPdf, weight);

                if (Ld.x >= 300.f || Ld.y >= 300.f || Ld.z >= 300.f)
                {
                    rtPrintf("LightSampling:f)%f %f %f Li:%f %f %f cosTheta:%f weight:%.7f lightpdf:%.7f bsdfPdf:%.7f Ld+:%f\n", f.x, f.y, f.z,
                        Li.x, Li.y, Li.z, fabs(dot(outWi, shaderParams.dgShading.nn)),
                        weight, lightPdf, bsdfPdf,
                        (f * Li * (fabs(dot(outWi, shaderParams.dgShading.nn)) * weight / lightPdf)).x);
                }
            }
        }
    }


    /* Sample BSDF with Multiple Importance Sampling. */
    randSamples = Get2D(&localSampler);

    float4 f = Sample_f[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, randSamples, bsdfPdf, Get1D(&localSampler), shaderParams);
    if (!isBlack(f) && bsdfPdf > 0.)
    {
        float weight = 1.f;

        //TODO:should use sampledType while involving with compound BSDF type
        if (shaderParams.bsdfType != CommonStructs::BSDFType::SmoothGlass && shaderParams.bsdfType != CommonStructs::BSDFType::SmoothMirror)
        {
            lightPdf = LightPdf[toUnderlyingValue(CommonStructs::LightType::HDRILight)](p, outWi, lightId, shadowRay);
            if (lightPdf == 0.f)
                return Ld;
            weight = TwUtil::MonteCarlo::PowerHeuristic(1, bsdfPdf, 1, lightPdf);
        }


        /* Trace shadow ray to find out whether it's blocked down by object. */
        CommonStructs::PerRayData_shadow shadow_prd;
        shadow_prd.blocked = 0;
        shadowRay = TwUtil::MakeShadowRay(p, sceneEpsilon, outWi);
        rtTrace<CommonStructs::PerRayData_shadow>(sysTopShadower, shadowRay, shadow_prd);

        if (!shadow_prd.blocked)
        {
            Li = TwUtil::Le_HDRILight(outWi, hdriEnvmap, hdriLight.worldToLight);
            if(!isBlack(Li))
                Ld += f * Li * fabsf(dot(outWi, shaderParams.dgShading.nn)) * weight / bsdfPdf;

            /* Check exceptional values. */
            if (isnan(Ld.x) || isnan(Ld.y) || isnan(Ld.z))
            {
                rtPrintf("costheta:%f\n", fabsf(dot(outWi, shaderParams.dgShading.nn)));
                rtPrintf("%f %f %f Li:%f %f %f weight:%f bsdfPdf:%f,outwi:%f %f %f",
                    f.x, f.y, f.z,
                    Li.x, Li.y, Li.z,
                    weight,
                    bsdfPdf,
                    outWi.x, outWi.y, outWi.z);
                rtPrintf("PdfBack:%f\n", Pdf[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, shaderParams));
            }
        }
    }

    return Ld;
}

template<>
static __device__ __inline__ float4 EstimateDirectLighting<CommonStructs::LightType::PointLight>(
    int lightId,
    const CommonStructs::ShaderParams & shaderParams,
    const float3 & isectP, const float3 & isectDir,
    GPUSampler &localSampler)
{
	float4 Ld = make_float4(0.f);

	float3 p = isectP;
	float3 wo_world = isectDir;
    float sceneEpsilon = sysSceneEpsilon;
	
	float3 outWi = make_float3(0.f);
	float lightPdf = 0.f;
	Ray shadowRay;

	//Sample point light, no MIS performed here
	float4 Li = Sample_Ld[toUnderlyingValue(CommonStructs::LightType::PointLight)](p, sceneEpsilon, outWi, lightPdf, make_float2(0.f), lightId, shadowRay);

	if (lightPdf > 0.f && !isBlack(Li))
	{
		float4 f = Eval_f[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, shaderParams);
		if (!isBlack(f))
		{
			//Trace ShadowRay
            CommonStructs::PerRayData_shadow shadow_prd;
			shadow_prd.blocked = 0;

			rtTrace<CommonStructs::PerRayData_shadow>(sysTopShadower, shadowRay, shadow_prd);

			if (!shadow_prd.blocked)
				Ld += f * Li * fabs(dot(outWi, shaderParams.dgShading.nn)) / lightPdf;
		}
	}

	return Ld;
}

template<>
static __device__ __inline__ float4 EstimateDirectLighting<CommonStructs::LightType::QuadLight>(
    int lightId,
    const CommonStructs::ShaderParams & shaderParams,
    const float3 & isectP, const float3 & isectDir,
    GPUSampler &localSampler)
{
	float4 Ld = make_float4(0.f);

	float3 p = isectP;
	float3 wo_world = isectDir;
    float sceneEpsilon = sysSceneEpsilon;

	float3 outWi = make_float3(0.f);
	float lightPdf = 0.f, bsdfPdf = 0.f;
    Ray shadowRay;

    /* Sample light source with Mulitple Importance Sampling. */
    float2 randSamples = Get2D(&localSampler);
	float4 Li = Sample_Ld[toUnderlyingValue(CommonStructs::LightType::QuadLight)](p, sceneEpsilon, outWi, lightPdf, randSamples, lightId, shadowRay);
	if (lightPdf > 0.f && !isBlack(Li))
	{
        /* Compute BSDF value using sampled outWi from sampling light source. */
		float4 f = Eval_f[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, shaderParams);
		if (!isBlack(f))
		{
            /* Trace shadow ray and find out its visibility. */
            CommonStructs::PerRayData_shadow shadow_prd;
			shadow_prd.blocked = 0;

			rtTrace<CommonStructs::PerRayData_shadow>(sysTopShadower, shadowRay, shadow_prd);
			if (!shadow_prd.blocked)
			{
                /* Compute Ld using MIS weight. */

				bsdfPdf = Pdf[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, shaderParams);
				float weight = TwUtil::MonteCarlo::PowerHeuristic(1, lightPdf, 1, bsdfPdf);
				Ld += f * Li * (fabs(dot(outWi, shaderParams.dgShading.nn)) * weight / lightPdf);
                /*rtPrintf("%f %f %f Li:%f %f %f weight:%f bsdfPdf:%f,outwi:%f %f %f\n",
                    f.x, f.y, f.z,
                    Li.x, Li.y, Li.z,
                    weight,
                    bsdfPdf,
                    outWi.x, outWi.y, outWi.z);
                rtPrintf("Ld %f %f %f\n", Ld.x, Ld.y, Ld.z);*/
			}
		}
	}


    /* Sample BSDF with Multiple Importance Sampling. */
    randSamples = Get2D(&localSampler);

	float4 f = Sample_f[toUnderlyingValue(shaderParams.bsdfType)](wo_world, outWi, randSamples, bsdfPdf, Get1D(&localSampler), shaderParams);
	if (!isBlack(f) && bsdfPdf > 0.)
	{
		float weight = 1.f;

		//todo:should use sampledType while we use overall bsdf type:
		if (shaderParams.bsdfType != BSDFType::SmoothGlass && shaderParams.bsdfType != BSDFType::SmoothMirror)
		{
			lightPdf = LightPdf[toUnderlyingValue(CommonStructs::LightType::QuadLight)](p, outWi, lightId, shadowRay);

			if (lightPdf == 0.f)
				return Ld;

			weight = TwUtil::MonteCarlo::PowerHeuristic(1, bsdfPdf, 1, lightPdf);
		}

        
        /* Trace shadow ray to find out whether it's blocked down by object. */
        CommonStructs::PerRayData_shadow shadow_prd;
        shadow_prd.blocked = 0;
        rtTrace<CommonStructs::PerRayData_shadow>(sysTopShadower, shadowRay, shadow_prd);

        /* In LightPdf we have confirmed that ray is able to intersect the light
         * -- surface from |p| towards |outWi| if there are no objects between
         * them. So now we need to make sure that hypothesis using shadow ray. 
         * Given |shadowRay| from LightPdf, ray's |tmax| is narrowed down and
         * could lead to further optimization when tracing shadow ray. */
		if (!shadow_prd.blocked)
		{
            Li = TwUtil::Le_QuadLight(quadLightBuffer[lightId], -outWi);
            if (!isBlack(Li))
                Ld += f * Li * fabsf(dot(outWi, shaderParams.dgShading.nn)) * weight / bsdfPdf;
		}
	}

	return Ld;
}

/**
 * @brief Uniformly sample all lights available in the scene and
 * compute direct lighting contribution to the surface interaction.
 *
 * @param[in] shaderParams material and differential geometry information
 * @param[in] isectP       intersection point in world space
 * @param[in] isectDir     incoming direction in world space, pointing out of the surface
 * @param[in] localSampler local sampler binded to current launch
 *
 * @see EstimateDirectLighting()
 *
 * @return Return direct lighting contribution due to one specific
 * light.
 */
static __device__ __inline__ float4 SampleLightsAggregate(const CommonStructs::ShaderParams & shaderParams, const float3 & isectP, const float3 & isectDir, GPUSampler &localSampler)
{
	float4 L = make_float4(0.f);

     if (hdriEnvmap != RT_TEXTURE_ID_NULL)
     {
         L += EstimateDirectLighting<CommonStructs::LightType::HDRILight>(-1, shaderParams, isectP, isectDir, localSampler);
     }
        
    for (int i = 0; i < pointLightBuffer.size(); ++i)//todo:use BufferId and RT_BUFFER_ID_NULL
    {
        L += EstimateDirectLighting<CommonStructs::LightType::PointLight>(i, shaderParams, isectP, isectDir, localSampler);
    }

    for (int i = 0; i < quadLightBuffer.size(); ++i)
    {
        L += EstimateDirectLighting<CommonStructs::LightType::QuadLight>(i, shaderParams, isectP, isectDir, localSampler);
    }
    

	return L;
}

#endif // COLVILLEA_DEVICE_INTEGRATOR_INTEGRATOR_H_