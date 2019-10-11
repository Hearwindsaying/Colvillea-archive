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
rtDeclareVariable(int, quadLightIndex, , ); /* potential area light binded to the geometryInstance */

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

    /* Include emitted radiance from surface. 
     * -- SampleLightsAggregate() does not account for that. */
    float4 Ld = (shaderParams.bsdfType == CommonStructs::BSDFType::Emissive ?
        TwUtil::Le_QuadLight(sysLightBuffers.quadLightBuffer[quadLightIndex], -ray.direction) :
        make_float4(0.f)); /* Emitted radiance from area light. */

	Ld += SampleLightsAggregate(shaderParams, ray.origin + tHit * ray.direction, -ray.direction, localSampler);

	prdRadiance.radiance = Ld;
}
