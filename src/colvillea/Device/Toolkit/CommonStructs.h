#pragma once

#ifndef COLVILLEA_DEVICE_TOOLKIT_COMMONSTRUCTS_H_
#define COLVILLEA_DEVICE_TOOLKIT_COMMONSTRUCTS_H_
#include <optixu/optixu_vector_types.h>

/** CommonStructs.h stores structs and scoped enums
 * that could be used in both host and device side
 * or device side only, according to annotation.
 * But none of them could be only used in host side,
 * which is defined in GlobalDefs.h.
 * 
 * Attention must be paid that some structs used for
 * containing data employed by GPU programs could 
 * possibly have name collision with those serving
 * only in host side which actually are of class
 * type. Better using namespace explicitly (don't
 * use 'using xxx;') on host side.
 * 
 * 
 */

namespace CommonStructs
{
    enum class RayTracingPipelinePhase : unsigned int
    {
        RayGeneration,
        Intersection,
        AnyHit,
        Miss,
        ClosestHit,
        Exception,

        CountOfType
    };

    enum class RayType
    {
        Radiance  = 0,
        Shadow    = 1,
        Detection = 2, /* for area light tracing detection or path tracing */

        CountOfType
    };


    enum class LightType : unsigned int
    {
        PointLight,
        HDRILight,
        QuadLight,

        CountOfType
    };

    struct Distribution1D
    {
        rtBufferId<float, 1> func;
        rtBufferId<float, 1> cdf;
        float funcIntegral;
    };

    struct Distribution2D
    {
        rtBufferId<float, 1> pConditionalV_funcIntegral;
        rtBufferId<float, 2> pConditionalV_func;
        rtBufferId<float, 2> pConditionalV_cdf;
        Distribution1D pMarginal;
    };

    struct HDRILight
    {
        optix::Matrix4x4 lightToWorld;
        optix::Matrix4x4 worldToLight;

        Distribution2D distributionHDRI;

        int hdriEnvmap;
        LightType lightType;
    };

    struct PointLight
    {
        optix::float4 intensity;
        optix::float3 lightPos; // Matrix4x4 is too hevay for PointLight!

        LightType lightType;
    };

    struct QuadLight
    {
        optix::float4 intensity;

        /* Transform matrices and |reverseOrientation| is the same as 
         * -- quadLight's underlying shape's one. */
        optix::Matrix4x4 lightToWorld;       /* corresponding to objectToWorld */
        optix::Matrix4x4 worldToLight;       /* corresponding to worldToObject */
        int              reverseOrientation; /* corresponding to reverseOrientation */
        float            invSurfaceArea;     /* corresponding to (not exists yet) invSurfaceArea
                                                -- useful in light sampling */

        LightType lightType;
        
        //int nSamples;
        //optix::float2 scaleXY;
    };

    struct LightBuffers
    {
        rtBufferId<CommonStructs::PointLight> pointLightBuffer;
        rtBufferId<CommonStructs::QuadLight>  quadLightBuffer;
        CommonStructs::HDRILight              hdriLight;
    };

    struct HDRIEnvmapLuminanceBufferWrapper
    {
        rtBufferId<float, 2> HDRIEnvmapLuminanceBuffer;
    };


    struct GlobalSobolSampler
    {
        int resolution, log2Resolution; /* Film resolution round up to power of two */

        rtBufferId<uint32_t, 1> sobolMatrices32;    /* 32bits Sobol Matrix */
        rtBufferId<uint64_t, 2> vdCSobolMatrices;   /* part of 64-bit Sobol Matrix */
        rtBufferId<uint64_t, 2> vdCSobolMatricesInv;/* inverse Matrix of "A" */
    };

    struct SobolSampler
    {
        int dimension;                 /* dimension of sampled vector, [0,1000) */
        int64_t intervalSampleIndex;   /* intermediate result
                                        --offset(of the i_th sample for current pixel) in global sequence */   
    };

    struct GlobalHaltonSampler
    {
        rtBufferId<uint32_t, 1>       fastPermutationTable; /* Permutation table with radical inverse to
                                                               support efficient implementation of
                                                               ScrambledRadicalInverse */
        rtBufferId<uint32_t, 2>       hs_offsetForCurrentPixelBuffer;
    };

    struct HaltonSampler
    {
        //int64_t offsetForCurrentPixel;/* offset(of the first sample) in global halton sequence for current                                pixel, only computed once when the iteration is 0 */
        //todo:next two variables could be optimized away.
        int dimension;                /* dimension of sampled vector, [0,1000) */
        uint32_t intervalSampleIndex;  /* intermediate result
									      --offset(of the i_th sample for current pixel) in global sequence */
    };

    union GPUSampler
    {
        SobolSampler  sobolSampler;
        HaltonSampler haltonSampler;
    };


    enum class SamplerType : unsigned int
    {
        HaltonQMCSampler,  // low discrepancy sampler: Halton sequence
        SobolQMCSampler,   // low discrepancy sampler: Sobol sequence

        CountOfType
    };



    enum BSDFType : unsigned int
    {
        Lambert         = 0,
        RoughMetal      = 1,
        RoughDielectric = 2,
        SmoothGlass     = 3,
        Plastic         = 4,
        Emissive        = 5,
        SmoothMirror    = 6, //todo:categorize by lobe attributes
        FrostedMetal    = 7,

        CountOfType
    };


    struct DifferentialGeometry
    {
        optix::float4 nn; /* normalized normal = normalize(cross(dpdu,dpdv)) 
                             -- where dpdu is not normalized
                             -- also use float4 for padding and sync with nGeometry 
                             to avoid make_float conversion	*/
        optix::float3 tn; /* tangent = cross(dpdu, nn) where dpdu,nn are both normalized */

        optix::float2 uv;
        optix::float3 dpdu; /* secondary tangent(i.e. sn), need to be normalized while initializing */
        optix::float3 dpdv;
    };

    struct ShaderParams
    {
        DifferentialGeometry dgShading;
        optix::float4        nGeometry;

        optix::float4 Reflectance;
        optix::float4 Specular;
        optix::float4 FrCond_e;
        optix::float4 FrCond_k;

        float alphax;
        float ior;
        int   ReflectanceID;
        int   SpecularID;
        int   RoughnessID;

        CommonStructs::BSDFType bsdfType;
    };


    //////////////////////////////////////////////////////////////////////////
    //Per-ray Data
    struct /*__align__(16) */PerRayData_radiance
    {
        optix::float4 radiance;
    };

    struct PerRayData_shadow
    {
        int blocked;

        optix::int3 padding;
    };

    struct PerRayData_pt
    {
        ShaderParams shaderParams;
        optix::float4 emittedRadiance;
        optix::float3 isectP;
        optix::float3 isectDir;

        int validPT;

        CommonStructs::BSDFType bsdfType;
    };
}


#if 0
/**@struct SphQuad
 * @brief SphericalQuad struct for the implementation of "An Area-Preserving Parametrization for Spherical Rectangles"
 * 
 */
struct SphQuad
{
	optix::float3 o, x, y, z;
	float z0, z0sq;
	float x0, y0, y0sq;
	float x1, y1, y1sq;
	float b0, b1, b0sq, k;
	float S;

	//added s indicating quadlight's origin
	optix::float3 s;
};
#endif

#endif // COLVILLEA_DEVICE_TOOLKIT_COMMONSTRUCTS_H_