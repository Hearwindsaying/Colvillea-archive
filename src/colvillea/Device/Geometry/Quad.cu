#include <optix_world.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include "../Toolkit/Utility.h"
#include "../Toolkit/CommonStructs.h"

using namespace optix;
using namespace CommonStructs;


//////////////////////////////////////////////////////////////////////////
//Forward declarations:
//global variables:->Context
rtDeclareVariable(optix::Ray,           ray,       rtCurrentRay, );
rtDeclareVariable(DifferentialGeometry, dgShading, attribute dgShading, );
rtDeclareVariable(optix::float4,        nGeometry, attribute nGeometry, );

//Geometry related variables:->GeometryInstance
rtDeclareVariable(int,       reverseOrientation, , ) = 0;
rtDeclareVariable(Matrix4x4, objectToWorld, , );
rtDeclareVariable(Matrix4x4, worldToObject, , );


RT_PROGRAM void BoundingBox_Quad(int primIdx, float result[6])
{
    /* By convention, local space quad occpuies on [-1,-1]*[1,1] on x-y plane. */
    Aabb *aabb = reinterpret_cast<Aabb *>(result);

    aabb->include(TwUtil::xfmPoint(make_float3(-1.f, -1.f, 0.f), objectToWorld));
    aabb->include(TwUtil::xfmPoint(make_float3( 1.f, -1.f, 0.f), objectToWorld));
    aabb->include(TwUtil::xfmPoint(make_float3( 1.f,  1.f, 0.f), objectToWorld));
    aabb->include(TwUtil::xfmPoint(make_float3(-1.f,  1.f, 0.f), objectToWorld));
}

RT_PROGRAM void Intersect_Quad(int primIdx)
{
    /* Transform world scope ray into local space to perform intersecting. 
     * -- Do not normalize ray. */
    Ray localRay = TwUtil::xfmRay(ray, worldToObject);

    /* Compute parameter [t] for intersection. */
    float tHit = -localRay.origin.z / localRay.direction.z; /* If ray and plane is parallel, tHit is infinity and would never be a valid intersection, which will simply ignored by rtPotentialIntersection. */

    float3 localHitPoint = localRay.origin + tHit * localRay.direction;
    /* Validate area for quad. */
    if (fabsf(localHitPoint.x) <= 1 && fabsf(localHitPoint.y) <= 1)
    {
        if (rtPotentialIntersection(tHit)) /* Not necessarily ignoring tHit ourselves. */
        {
            /* Fill in DifferentialGeomery and [nGeometry]. */
            dgShading.rayEpsilon = 1e-3f * tHit;
            dgShading.uv = make_float2((localHitPoint.x + 1.f) / 2.f, (localHitPoint.y + 1.f) / 2.f);
            dgShading.dpdu = TwUtil::xfmVector(make_float3(2.f, 0.f, 0.f), objectToWorld);
            dgShading.dpdv = TwUtil::xfmVector(make_float3(0.f, 2.f, 0.f), objectToWorld);
            nGeometry = make_float4(TwUtil::safe_normalize(cross(dgShading.dpdu, dgShading.dpdv))); /* todo:encapsulate differentialgeomtery maker and delete dpdv. better name rules. */

            if (reverseOrientation)
                nGeometry *= -1.f;
            dgShading.nn = nGeometry;

            dgShading.dpdu = TwUtil::safe_normalize(dgShading.dpdu);
            dgShading.tn = cross(make_float3(dgShading.nn), dgShading.dpdu);

            rtReportIntersection(0);
        }
    }
}