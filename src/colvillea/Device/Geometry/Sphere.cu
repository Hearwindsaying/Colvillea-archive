#include <optix_world.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include "../Toolkit/Utility.h"
#include "../Toolkit/CommonStructs.h"

#include "Sphere.h"

using namespace optix;
using namespace CommonStructs; 


//////////////////////////////////////////////////////////////////////////
//Forward declarations:
//global variables:->Context
rtDeclareVariable(optix::Ray,           ray,       rtCurrentRay, );
rtDeclareVariable(DifferentialGeometry, dgShading, attribute dgShading, );
rtDeclareVariable(optix::float4,        nGeometry, attribute nGeometry, );

//Geometry related variables:->GeometryInstance
rtDeclareVariable(Matrix4x4, objectToWorld, , );
rtDeclareVariable(Matrix4x4, worldToObject, , );
rtDeclareVariable(float,     radius, ,);
rtDeclareVariable(float3,    center, ,);

#ifndef TWRT_DECLARE_SYSLAUNCH
#define TWRT_DECLARE_SYSLAUNCH
rtDeclareVariable(uint2, sysLaunch_Dim, rtLaunchDim, );
rtDeclareVariable(uint2, sysLaunch_index, rtLaunchIndex, );
#endif

RT_PROGRAM void BoundingBox_Sphere(int primIdx, float result[6])
{
    Aabb *aabb = reinterpret_cast<Aabb *>(result);

    aabb->include(center - radius);
    aabb->include(center + radius);
}

__device__ __inline__ void fillAttributes_Sphere(float t)
{
    float3 p = ray.origin + t * ray.direction;
    p = center + TwUtil::safe_normalize(p - center)*radius;

    float3 local = TwUtil::xfmVector(p - center, worldToObject);
    float theta = TwUtil::safe_acos(local.z / radius);
    float phi = atan2f(local.y, local.x);

    if (phi < 0)
        phi += 2 * M_PIf;

    dgShading.uv = make_float2(phi*(0.5f*M_1_PIf), theta*M_1_PIf);

    dgShading.dpdu = TwUtil::safe_normalize(TwUtil::xfmVector(make_float3(-local.y, local.x, 0)*(2 * M_PIf), objectToWorld));
    nGeometry = make_float4(TwUtil::safe_normalize(p - center));
    
    /* dpdv is removed. */
    //float zrad = sqrtf(local.x*local.x + local.y*local.y);

    //if (zrad > 0.f)
    //{
    //    float invZRad = 1.0f / zrad;
    //    float cosPhi = local.x * invZRad;
    //    float sinPhi = local.y * invZRad;
    //    //dgShading.dpdv = TwUtil::xfmVector(make_float3(local.z*cosPhi, local.z*sinPhi, -sin(theta)*radius)*M_PIf, objectToWorld);
    //    //dgShading.dpdu = TwUtil::safe_normalize(dgShading.dpdu);
    //    //dgShading.dpdv = TwUtil::safe_normalize(dgShading.dpdv);
    //}
    //else
    //{
    //    float cosPhi = 0.f;
    //    float sinPhi = 1.f;
    //    dgShading.dpdv = TwUtil::xfmVector(make_float3(local.z*cosPhi, local.z*sinPhi, -sinf(theta)*radius)*M_PIf, objectToWorld);
    //    TwUtil::CoordinateSystem(make_float3(nGeometry.x, nGeometry.y, nGeometry.z), dgShading.dpdu, dgShading.dpdv);
    //}

    dgShading.nn = nGeometry;
    dgShading.tn = cross(make_float3(dgShading.nn), dgShading.dpdu);
}

RT_PROGRAM void Intersect_Sphere(int primIdx)
{
    float3 o = ray.origin - center;
    float3 d = ray.direction;

    double A = TwUtil::sqr_length(d);
    double B = 2 * dot(o, d);
    double C = TwUtil::sqr_length(o) - radius * radius;

    double nearT, farT;
    if (TwUtil::solveQuadraticDouble(A, B, C, nearT, farT))
    {
        if (nearT <= ray.tmax && farT >= ray.tmin)
        {
            if (nearT < ray.tmin)
            {
                if (farT < ray.tmax)
                {
                    // Pick farT
                    if (rtPotentialIntersection(farT))
                    {
                        fillAttributes_Sphere(farT);
                        rtReportIntersection(0);
                    }
                }
            }
            else
            {
                // Pick nearT
                if (rtPotentialIntersection(nearT))
                {
                    fillAttributes_Sphere(nearT);
                    rtReportIntersection(0);
                }
            }
        }
    }
}