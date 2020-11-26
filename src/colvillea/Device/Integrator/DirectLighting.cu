#include <optix_world.h>
#include <optix_device.h>

#include "Integrator.h"
#include "../Shader/MicrofacetBRDF.h"

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
rtDeclareVariable(int, reverseOrientation, , );
rtDeclareVariable(int, quadLightIndex, , ); /* potential area light binded to the geometryInstance */


//differential geometry:->Attribute
rtDeclareVariable(optix::float4,        nGeometry, attribute nGeometry, );
rtDeclareVariable(DifferentialGeometry, dgShading, attribute dgShading, );

// Visualize BRDF distribution
rtDeclareVariable(float, wo_theta, , ) = 0.0f;
rtDeclareVariable(float, wo_phi, , ) = 0.0f;

static __device__ __inline__ float3 sphericalToCartesian(const float theta, const float phi)
{
    return make_float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
};

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
//ClosestHit program:
RT_PROGRAM void ClosestHit_DirectLighting(void)
{
#if 1
    GPUSampler localSampler; 
    makeSampler(RayTracingPipelinePhase::ClosestHit, localSampler);

	ShaderParams shaderParams           = shaderBuffer[materialIndex];
	             shaderParams.nGeometry = nGeometry;
	             shaderParams.dgShading = dgShading;

    /* Include emitted radiance from surface. 
     * -- SampleLightsAggregate() does not account for that. */
    
    float4 Ld = (shaderParams.bsdfType == CommonStructs::BSDFType::Emissive ?
        TwUtil::Le_QuadLight(sysLightBuffers.quadLightBuffer[quadLightIndex], -ray.direction) :
        make_float4(0.f)); /* Emitted radiance from area light. */

	Ld += SampleLightsAggregate(shaderParams, ray.origin + tHit * ray.direction, -ray.direction, localSampler);

    if (sysLaunch_index == make_uint2(971, 720 - 384))
        rtPrintf("ld :%f %f %f\n", Ld.x, Ld.y, Ld.z);

	prdRadiance.radiance = Ld;
#else
    float3 wo = sphericalToCartesian(wo_theta, wo_phi);
    float3 wi = safe_normalize(ray.origin + tHit * ray.direction);

    ShaderParams shaderParams = shaderBuffer[materialIndex];
    shaderParams.nGeometry = nGeometry;
    shaderParams.dgShading = dgShading;

    float3 wh = wo + wi;
    wh = safe_normalize(wh);

    //prdRadiance.radiance = MicrofacetReflection_InnerEval_f(wo, wi, shaderParams, true) * fabsf(wi.z);
    prdRadiance.radiance = make_float4(Ashikhmin_D(shaderParams.alphax, BSDFMath::AbsCosTheta(wh)) * fabsf(wi.z));
#endif
}
