#include <optix_world.h>
#include <optix_device.h>

#include "Integrator.h"


using namespace optix;

//////////////////////////////////////////////////////////////////////////
//Forward declarations:
// Light buffer:->Context
#ifndef TWRT_DELCARE_LIGHTBUFFER
#define TWRT_DELCARE_LIGHTBUFFER
rtDeclareVariable(CommonStructs::LightBuffers, sysLightBuffers, , );
#endif

// Material buffer:->Context
rtBuffer<ShaderParams, 1> shaderBuffer;

// Material related:->GeometryInstance
rtDeclareVariable(int, materialIndex, , );
// Avaliable for Quad Area Light
rtDeclareVariable(int, quadLightIndex, , ) = -1; /* potential area light binded to the geometryInstance */
// Avaliable for Sphere Area Light
rtDeclareVariable(int, sphereLightIndex, , ) = -1;

//differential geometry:->Attribute
rtDeclareVariable(optix::float4,		nGeometry, attribute nGeometry, );
rtDeclareVariable(DifferentialGeometry, dgShading, attribute dgShading, );

//Path Tracing Integrator:->Context /*todo:change into variables*/
rtDeclareVariable(PerRayData_pt, prdPT,					rtPayload, );
rtDeclareVariable(int,			 ptIntegrator_maxDepth, , );
rtDeclareVariable(int,           ptIntegrator_disableRR, ,);

//Statistics:->Context
//rtBuffer<uint2> testBounces;

//////////////////////////////////////////////////////////////////////////
//Closest Hit program for PT Ray:
RT_PROGRAM void ClosestHit_PTRay_PathTracing()
{
    /* Confirm that detective ray has hit an object. */
	prdPT.validPT = 1;

    /* Copy shaderParams and ray attributes to per-ray data for transferring. */
	prdPT.shaderParams           = shaderBuffer[materialIndex];
	prdPT.shaderParams.dgShading = dgShading;
	prdPT.shaderParams.nGeometry = nGeometry;

	prdPT.isectP = ray.origin + tHit * ray.direction;
	prdPT.isectDir = ray.direction;

	prdPT.bsdfType = shaderBuffer[materialIndex].bsdfType;
    if (prdPT.bsdfType == CommonStructs::BSDFType::Emissive)
    {
        if(quadLightIndex != -1)
            prdPT.emittedRadiance = TwUtil::Le_QuadLight(sysLightBuffers.quadLightBuffer[quadLightIndex], -ray.direction);
        else
            prdPT.emittedRadiance = (TwUtil::dot(prdPT.isectP - sysLightBuffers.sphereLightBuffer[sphereLightIndex].center, nGeometry) > 0.f
                                     ? sysLightBuffers.sphereLightBuffer[sphereLightIndex].intensity
                                     : make_float4(0.f));
    }
}

//////////////////////////////////////////////////////////////////////////
//Closest Hit program for Radiance Ray:
RT_PROGRAM void ClosestHit_PathTracing()
{
	/* bug: we might added specular light twice possibly because in pbrt's convention, SamplingLightsAggregate ignores perfect Specular BSDF which is compensated in PathTracing integrator(here)
	/* however, the origin version doesn't obey the convention and fatal errors in image could happen.
	*/

    /* Declare sampler. */
    GPUSampler localSampler; 
    makeSampler(RayTracingPipelinePhase::ClosestHit, localSampler);

    /* Declare data for path tracing. */
	float4 L = make_float4(0.f);
	float4 beta = make_float4(1.f);
	bool specularBounce = false;
	int bounces = 0; /* Seperate definition for statistics counting. */

    /* Copy data directly from attribute for the first ray. */
	Ray nextRay;
	ShaderParams currentShaderParams           = shaderBuffer[materialIndex];
	             currentShaderParams.dgShading = dgShading;
	             currentShaderParams.nGeometry = nGeometry;

    float3 isectP = ray.origin + tHit * ray.direction;
	float3 isectDir = ray.direction;

    //float4 emittedRadiance = (currentShaderParams.bsdfType == CommonStructs::BSDFType::Emissive ?
    //    TwUtil::Le_QuadLight(sysLightBuffers.quadLightBuffer[quadLightIndex], -ray.direction) :
    //    make_float4(0.f)); /* Emitted radiance from area light. */
    float4 emittedRadiance = (currentShaderParams.bsdfType == CommonStructs::BSDFType::Emissive ?
                                (quadLightIndex == -1 ?
                                     (TwUtil::dot(isectP - sysLightBuffers.sphereLightBuffer[sphereLightIndex].center, nGeometry) > 0.f ? sysLightBuffers.sphereLightBuffer[sphereLightIndex].intensity
                                      : make_float4(0.f))
                                 : TwUtil::Le_QuadLight(sysLightBuffers.quadLightBuffer[quadLightIndex], -ray.direction))
                             : make_float4(0.f));

    /* Always found intersection for the first ray. */
    bool foundIntersection = true;

	for (;; ++bounces)
	{
        /* No need to detect for the first ray. */
		if (bounces != 0)
		{
			PerRayData_pt prdPathTracing;
			prdPathTracing.validPT = 0; /* Nothing is hit initially. */

			rtTrace<PerRayData_pt>(sysTopObject, nextRay, prdPathTracing, RT_VISIBILITY_ALL, RT_RAY_FLAG_DISABLE_ANYHIT);

			/* Copy shaderParams and intersection data back. */
			currentShaderParams           = prdPathTracing.shaderParams;
			currentShaderParams.dgShading = prdPathTracing.shaderParams.dgShading;
			currentShaderParams.nGeometry = prdPathTracing.shaderParams.nGeometry;
            emittedRadiance = prdPathTracing.emittedRadiance;
			isectP = prdPathTracing.isectP;
			isectDir = nextRay.direction; //isectDir = prdPathTracing.isectDir;

			foundIntersection = prdPathTracing.validPT != 0;
		}

        /* We have to add contribution from area light and HDRILight for the first ray.
         * -- This is the only opportunity to account for direct lighting
         * from area light for the ray started from camera.
         * Another case is that when last bounce is due to specular component,
         * we need to add light contribution as well. */
		if (bounces == 0 || specularBounce)
		{
			/* Possibly intersect an area light. */
			if (foundIntersection) 
			{
                L += beta * emittedRadiance;
			}
			else /* Ray has escaped from scene, adding light contribution from HDRI envinronment. */
			{
				L += beta * TwUtil::Le_HDRILight(isectDir, sysLightBuffers.hdriLight.hdriEnvmap, sysLightBuffers.hdriLight.worldToLight);

				if (L.x >= 300.f || L.y >= 300.f || L.z >= 300.f)
				{
					//rtPrintf("L:%f %f %f,bounces:%d,beta:%f\n", L.x, L.y, L.z, bounces, beta.x);
				}
			}
		}

		/* Terminate ray if necessary. */
		if (!foundIntersection || bounces >= ptIntegrator_maxDepth) 
            break;

		/* Explicitly evaluate direct lighting at each path vertex.
		 * -- Skip for Smooth BSDF. */
		if (currentShaderParams.bsdfType != CommonStructs::BSDFType::SmoothGlass &&  
            currentShaderParams.bsdfType != CommonStructs::BSDFType::SmoothMirror)
		{         
            float4 SA = SampleLightsAggregate(currentShaderParams, isectP, -isectDir, localSampler);
			float4 Ld = beta * SA;
			L += Ld;

			if (isnan(L.x) || isnan(L.y) || isnan(L.z))
				rtPrintf("l126 bounces:%d,L:%f %f %f,beta:%f %f %f,SA:%f %f %f\n", bounces, L.x, L.y, L.z, beta.x, beta.y, beta.z,
					SA.x, SA.y, SA.z);

			if (L.x >= 300.f || L.y >= 300.f || L.z >= 300.f)
			{
				rtPrintf("L:%f %f %f,bounces:%d,beta:%f SA:%f %f %f\n", L.x, L.y, L.z, bounces, beta.x, SA.x, SA.y, SA.z);
			}
		}

		/* Sample BSDF to get next ray. */
		float3 wo = -isectDir, wi = make_float3(0.f);
		float pdf = 0.f;

        float4 f = Sample_f[toUnderlyingValue(currentShaderParams.bsdfType)](wo, wi, Get2D(&localSampler), pdf, Get1D(&localSampler), currentShaderParams);

		if (isBlack(f) || pdf == 0.f)
			break;
			
		beta *= f * fabsf(dot(wi, currentShaderParams.dgShading.nn)) / pdf;


		if (L.x >= 300.f || L.y >= 300.f || L.z >= 300.f)
		{
			/*rtPrintf("last L:%f bounces:%d curSA:%f curL:%f %f %f,pt%f,"
				"UpComingPT:%f=%f*cosTheta:%f/pdf:%f\n", 
				stL.x, bounces, stSA.x, L.x, L.y, L.z, pathThroughput.x, 
				(f * fabsf(dot(wi_World, currentMaterialParams.dgShading.nn)) / pdf).x,
				f.x,
				fabsf(dot(wi_World, currentMaterialParams.dgShading.nn)),
				pdf);*/
            rtPrintf("L:%f %f %f,bounces:%d,beta:%f\n", L.x, L.y, L.z, bounces, beta.x);
		}

		specularBounce = (currentShaderParams.bsdfType == CommonStructs::BSDFType::SmoothGlass || 
                          currentShaderParams.bsdfType == CommonStructs::BSDFType::SmoothMirror);

		nextRay = make_Ray(isectP, wi, toUnderlyingValue(CommonStructs::RayType::Detection), sysSceneEpsilon, RT_DEFAULT_MAX);

        /* Roussian Roulette. */
		if (fmaxf(make_float3(beta.x, beta.y, beta.z)) < 1.f && 
            bounces > 3 && 
            ptIntegrator_disableRR == 0)
		{
			float q = fmaxf(0.05f, 1 - fmaxf(make_float3(beta.x, beta.y, beta.z)));
			//float q = fmaxf(0.05f, 1 - TwUtil::luminance(make_float3(pathThroughput.x, pathThroughput.y, pathThroughput.z)));
            if (Get1D(&localSampler) < q)
				break;
			beta /= 1.f - q;

			if (isinf(TwUtil::luminance(make_float3(beta.x, beta.y, beta.z))))
				rtPrintf("inf radiance got in Russian Roulette.");
		}
	}

	prdRadiance.radiance = L;
	if (isinf(L.x) || isinf(L.y) || isinf(L.z))
		rtPrintf("[error]inf radiance got in path tracing integrator.\n");
	else if (isnan(L.x) || isnan(L.y) || isnan(L.z))
		rtPrintf("[error]nan radiance got in path tracing integrator.\n");


	//Statistics:
	//atomicAdd(&(testBounces[0].x), (uint)bounces);
	//atomicAdd(&(testBounces[0].y), (uint)1);
}