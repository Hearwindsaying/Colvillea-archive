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
rtDeclareVariable(int, reverseOrientation, , ); 
rtDeclareVariable(int, quadLightIndex, , )=-1; /* potential area light binded to the geometryInstance */
// Avaliable for Sphere Area Light
rtDeclareVariable(int, sphereLightIndex, , )=-1; // TODO: use light type


//differential geometry:->Attribute
rtDeclareVariable(optix::float4,        nGeometry, attribute nGeometry, );
rtDeclareVariable(DifferentialGeometry, dgShading, attribute dgShading, );

//////////////////////////////////////////////////////////////////////////
//ClosestHit program:
RT_PROGRAM void ClosestHit_DirectLighting(void)
{
    GPUSampler localSampler; 
    makeSampler(RayTracingPipelinePhase::ClosestHit, localSampler);

	ShaderParams shaderParams           = shaderBuffer[materialIndex];
	             shaderParams.nGeometry = nGeometry;
	             shaderParams.dgShading = dgShading;

    const optix::float3 isectP = ray.origin + tHit * ray.direction;

    /* Include emitted radiance from surface. 
     * -- SampleLightsAggregate() does not account for that. */
    
    float4 Ld = (shaderParams.bsdfType == CommonStructs::BSDFType::Emissive ?
                   (quadLightIndex == -1 ? 
                       (TwUtil::dot(isectP - sysLightBuffers.sphereLightBuffer[sphereLightIndex].center, nGeometry)>0.f ? sysLightBuffers.sphereLightBuffer[sphereLightIndex].intensity
                        : make_float4(0.f)) 
                    : TwUtil::Le_QuadLight(sysLightBuffers.quadLightBuffer[quadLightIndex], -ray.direction)) 
                : make_float4(0.f)); /* Emitted radiance from area light. */
    if (shaderParams.bsdfType != CommonStructs::BSDFType::Emissive)
        Ld += SampleLightsAggregate(shaderParams, isectP, -ray.direction, localSampler);

	prdRadiance.radiance = Ld;
}
