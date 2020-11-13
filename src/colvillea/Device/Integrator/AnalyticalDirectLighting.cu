#include <optix_world.h>
#include <optix_device.h>
#include <optixu_math_namespace.h>

#include "Integrator.h"

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"
#include "../Toolkit/NvRandom.h"
#include "../Light/LightUtil.h"

#include "../Sampler/Sampler.h"

using namespace optix;

//////////////////////////////////////////////////////////////////////////
//Forward declarations:
// Light buffer:->Context
#ifndef TWRT_DELCARE_LIGHTBUFFER
#define TWRT_DELCARE_LIGHTBUFFER
rtDeclareVariable(CommonStructs::LightBuffers, sysLightBuffers, , );
#endif

#ifndef TWRT_DELCARE_SAMPLERTYPE
#define TWRT_DELCARE_SAMPLERTYPE
rtDeclareVariable(int, sysSamplerType, , );         /* Sampler type chosen in GPU program.
                                                       --need to fetch underlying value from
                                                         CommonStructs::SamplerType. */
#endif

//rtDeclareVariable(Ray,                                ray,         rtCurrentRay, );
//rtDeclareVariable(float,                              tHit,        rtIntersectionDistance, );

#ifndef TWRT_DECLARE_SYS_ITERATION_INDEX
#define TWRT_DECLARE_SYS_ITERATION_INDEX
rtDeclareVariable(uint, sysIterationIndex, , ) = 0;
#endif

#ifndef TWRT_DECLARE_SYSLAUNCH
#define TWRT_DECLARE_SYSLAUNCH
rtDeclareVariable(uint2, sysLaunch_Dim, rtLaunchDim, );
rtDeclareVariable(uint2, sysLaunch_index, rtLaunchIndex, );
#endif

// Material buffer:->Context
rtBuffer<ShaderParams, 1> shaderBuffer;

// Material related:->GeometryInstance

rtDeclareVariable(int, materialIndex, , );
rtDeclareVariable(int, reverseOrientation, , );
rtDeclareVariable(int, quadLightIndex, , ); /* potential area light binded to the geometryInstance */


//differential geometry:->Attribute
rtDeclareVariable(optix::float4, nGeometry, attribute nGeometry, );
rtDeclareVariable(DifferentialGeometry, dgShading, attribute dgShading, );

// LTC LUTs
rtDeclareVariable(LTCBuffers, ltcBuffers, , );

#define LUT_SIZE 64.0
#define LUT_SCALE (LUT_SIZE - 1.0) / LUT_SIZE
#define LUT_BIAS 0.5 / LUT_SIZE

using namespace optix;


static __device__ __inline__ void ClipQuadToHorizon(float3 L[5], int& n)
{
    // detect clipping config
    int config = 0;
    if (L[0].z > 0.0) config += 1;
    if (L[1].z > 0.0) config += 2;
    if (L[2].z > 0.0) config += 4;
    if (L[3].z > 0.0) config += 8;

    // clip
    n = 0;

    if (config == 0)
    {
        // clip all
    }
    else if (config == 1) // V1 clip V2 V3 V4
    {
        n = 3;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 2) // V2 clip V1 V3 V4
    {
        n = 3;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 3) // V1 V2 clip V3 V4
    {
        n = 4;
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
        L[3] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 4) // V3 clip V1 V2 V4
    {
        n = 3;
        L[0] = -L[3].z * L[2] + L[2].z * L[3];
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
    }
    else if (config == 5) // V1 V3 clip V2 V4) impossible
    {
        n = 0;
    }
    else if (config == 6) // V2 V3 clip V1 V4
    {
        n = 4;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 7) // V1 V2 V3 clip V4
    {
        n = 5;
        L[4] = -L[3].z * L[0] + L[0].z * L[3];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 8) // V4 clip V1 V2 V3
    {
        n = 3;
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
        L[1] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = L[3];
    }
    else if (config == 9) // V1 V4 clip V2 V3
    {
        n = 4;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[2].z * L[3] + L[3].z * L[2];
    }
    else if (config == 10) // V2 V4 clip V1 V3) impossible
    {
        n = 0;
    }
    else if (config == 11) // V1 V2 V4 clip V3
    {
        n = 5;
        L[4] = L[3];
        L[3] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 12) // V3 V4 clip V1 V2
    {
        n = 4;
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
    }
    else if (config == 13) // V1 V3 V4 clip V2
    {
        n = 5;
        L[4] = L[3];
        L[3] = L[2];
        L[2] = -L[1].z * L[2] + L[2].z * L[1];
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
    }
    else if (config == 14) // V2 V3 V4 clip V1
    {
        n = 5;
        L[4] = -L[0].z * L[3] + L[3].z * L[0];
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
    }
    else if (config == 15) // V1 V2 V3 V4
    {
        n = 4;
    }

    if (n == 3)
        L[3] = L[0];
    if (n == 4)
        L[4] = L[0];
}

static __device__ __inline__ float inversesqrt(float x)
{
    assert(x >= 0.f);
    return 1.f / sqrtf(x);
}

static __device__ __inline__ float3 IntegrateEdgeVec(const float3 &v1, const float3& v2)
{
    float x = dot(v1, v2);
    float y = fabsf(x);

    float a = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
    float b = 3.4175940 + (4.1616724 + y) * y;
    float v = a / b;

    float theta_sintheta = (x > 0.0) ? v : 0.5 * inversesqrt(fmaxf(1.0 - x * x, 1e-7f)) - v;

    return cross(v1, v2) * theta_sintheta;
}

static __device__ __inline__ float IntegrateEdge(const float3& v1, const float3& v2)
{
    return IntegrateEdgeVec(v1, v2).z;
}

static __device__ __inline__ float4 LTC_Evaluate(
    const float3 &N, const float3& V, const float3& P, const Matrix3x3& Minv, const float3 points[4], bool twoSided)
{
    // construct orthonormal basis around N
    float3 T1, T2;
    T1 = safe_normalize(V - N * dot(V, N));
    T2 = cross(N, T1);

    // rotate area light in (T1, T2, N) basis
    const float T1T2NMatrixData[9]{ T1.x,T1.y,T1.z,T2.x,T2.y,T2.z,N.x,N.y,N.z };
    const Matrix3x3 tmpMat = Minv * Matrix3x3{ T1T2NMatrixData };

    // polygon (allocate 5 vertices for clipping)
    float3 L[5];
    L[0] = tmpMat * (points[0] - P);
    L[1] = tmpMat * (points[1] - P);
    L[2] = tmpMat * (points[2] - P);
    L[3] = tmpMat * (points[3] - P);

    // integrate
    float sum = 0.0;

    int n;
    ClipQuadToHorizon(L, n);

    if (n == 0)
        return make_float4(0.f);
    // project onto sphere
    L[0] = safe_normalize(L[0]);
    L[1] = safe_normalize(L[1]);
    L[2] = safe_normalize(L[2]);
    L[3] = safe_normalize(L[3]);
    L[4] = safe_normalize(L[4]);

    // integrate
    sum += IntegrateEdge(L[0], L[1]);
    sum += IntegrateEdge(L[1], L[2]);
    sum += IntegrateEdge(L[2], L[3]);
    if (n >= 4)
        sum += IntegrateEdge(L[3], L[4]);
    if (n == 5)
        sum += IntegrateEdge(L[4], L[0]);

    sum = twoSided ? fabsf(sum) : fmaxf(0.0, sum);

    return make_float4(sum, sum, sum, 1.0f);
}


template<CommonStructs::LightType lightType>
static __device__ __inline__ float4 AnalyticalEstimateDirectLighting(
    int lightId,
    const CommonStructs::ShaderParams& shaderParams,
    const float3& isectP, const float3& isectDir,
    GPUSampler& localSampler);

template<>
static __device__ __inline__ float4 AnalyticalEstimateDirectLighting<CommonStructs::LightType::QuadLight>(
    int lightId,
    const CommonStructs::ShaderParams& shaderParams,
    const float3& isectP, const float3& isectDir,
    GPUSampler& localSampler)
{
    float4 Ld = make_float4(0.f);

    float3 wo_world = isectDir;
    //float sceneEpsilon = sysSceneEpsilon;

    //float3 outWi = make_float3(0.f);
    //float lightPdf = 0.f, bsdfPdf = 0.f;
    //Ray shadowRay;

    float3 quadShape[4];
    const CommonStructs::QuadLight& quadLight = sysLightBuffers.quadLightBuffer[lightId];

    // Order in ClockWise, consistent with Eric's code.
    // Still in WorldSpace
    quadShape[0] = TwUtil::xfmPoint(make_float3(-1.f, -1.f, 0.f), quadLight.lightToWorld);
    quadShape[1] = TwUtil::xfmPoint(make_float3(1.f, -1.f, 0.f), quadLight.lightToWorld);
    quadShape[2] = TwUtil::xfmPoint(make_float3(1.f, 1.f, 0.f), quadLight.lightToWorld);
    quadShape[3] = TwUtil::xfmPoint(make_float3(-1.f, 1.f, 0.f), quadLight.lightToWorld);

    float ndotv = optix::clamp(TwUtil::dot(wo_world, shaderParams.dgShading.nn), 0.0f, 1.0f);
    float2 uv = make_float2(shaderParams.alphax, sqrtf(1.0f - ndotv));
    uv = uv * LUT_SCALE + LUT_BIAS;

    assert(ltcBuffers.ltc1 != RT_TEXTURE_ID_NULL);
    float4 t1 = rtTex2D<float4>(ltcBuffers.ltc1, uv.x, uv.y);
    
    const float mInvData[3 * 3]{ t1.x, 0, t1.z,
                                 0,    1,    0, 
                                 t1.y, 0, t1.w };
    Matrix3x3 mInv{ mInvData };
    float4 spec = LTC_Evaluate(make_float3(shaderParams.dgShading.nn), wo_world, isectP, mInv, quadShape, false);

    return spec * quadLight.intensity;
}

static __device__ __inline__ float4 AnalyticalQuadLights(const CommonStructs::ShaderParams& shaderParams, const float3& isectP, const float3& isectDir, GPUSampler& localSampler)
{
    float4 L = make_float4(0.f);

    if (sysLightBuffers.quadLightBuffer != RT_BUFFER_ID_NULL)
    {
        for (int i = 0; i < sysLightBuffers.quadLightBuffer.size(); ++i)
        {
            L += AnalyticalEstimateDirectLighting<CommonStructs::LightType::QuadLight>(i, shaderParams, isectP, isectDir, localSampler);
        }
    }

    return L;
}

// Visualize BRDF distribution
rtDeclareVariable(float, wo_theta, , ) = 0.0f;
rtDeclareVariable(float, wo_phi, , ) = 0.0f;

static __device__ __inline__ float3 sphericalToCartesian(const float theta, const float phi)
{
    return make_float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
};

static __device__ __inline__ float Eval_Do(const float3& w)
{
    return M_1_PIf * fmaxf(0, w.z);
}

//////////////////////////////////////////////////////////////////////////
//ClosestHit program:
RT_PROGRAM void ClosestHit_AnalyticalDirectLighting(void)
{
#if 1
    GPUSampler localSampler;
    makeSampler(RayTracingPipelinePhase::ClosestHit, localSampler);

    ShaderParams shaderParams = shaderBuffer[materialIndex];
    shaderParams.nGeometry = nGeometry;
    shaderParams.dgShading = dgShading;

    /* Include emitted radiance from surface.
     * -- SampleLightsAggregate() does not account for that. */

    float4 Ld = (shaderParams.bsdfType == CommonStructs::BSDFType::Emissive ?
        TwUtil::Le_QuadLight(sysLightBuffers.quadLightBuffer[quadLightIndex], -ray.direction) :
        make_float4(0.f)); /* Emitted radiance from area light. */

    Ld += AnalyticalQuadLights(shaderParams, ray.origin + tHit * ray.direction, -ray.direction, localSampler);

    prdRadiance.radiance = Ld;
#else
    float3 wo = sphericalToCartesian(wo_theta, wo_phi);
    float3 wi = safe_normalize(ray.origin + tHit * ray.direction);

    ShaderParams shaderParams = shaderBuffer[materialIndex];
    shaderParams.nGeometry = nGeometry;
    shaderParams.dgShading = dgShading;

    float ndotv = optix::clamp(TwUtil::dot(wo, shaderParams.dgShading.nn), 0.0f, 1.0f);
    float2 uv = make_float2(shaderParams.alphax, sqrtf(1.0f - ndotv));
    uv = uv * LUT_SCALE + LUT_BIAS;
    assert(ltcBuffers.ltc1 != RT_TEXTURE_ID_NULL);
    float4 t1 = rtTex2D<float4>(ltcBuffers.ltc1, uv.x, uv.y);

    const float mInvData[3 * 3]{ t1.x, 0, t1.z,
                                 0,    1,    0,
                                 t1.y, 0, t1.w };
    Matrix3x3 mInv{ mInvData };

    float Do = Eval_Do(safe_normalize(mInv * wi));
    Do *= mInv.det() / (length(mInv * wi) * length(mInv * wi) * length(mInv * wi));
    prdRadiance.radiance = make_float4(Do);
#endif
}
