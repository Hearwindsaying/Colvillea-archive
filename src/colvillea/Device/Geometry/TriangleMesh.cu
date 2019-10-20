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
rtDeclareVariable(optix::Ray,			ray,	   rtCurrentRay, );
rtDeclareVariable(DifferentialGeometry, dgShading, attribute dgShading, );
rtDeclareVariable(optix::float4,        nGeometry, attribute nGeometry, );//use float4 for padding

//trianglemesh buffers:->Geometry
rtBuffer<float3> vertexBuffer;
rtBuffer<float3> normalBuffer;
rtBuffer<float2> texcoordBuffer;

rtBuffer<int3>   vertexIndexBuffer;
rtBuffer<int3>   texcoordIndexBuffer;
rtBuffer<int3>   normalIndexBuffer;


RT_PROGRAM void Attributes_TriangleMesh()
{
    /* Get primitive index to |vertexIndexBuffer| . */
    const int primIdx = rtGetPrimitiveIndex();
    
    /* Get vertex index to |vertexBuffer|. */
    const int3 v_idx = vertexIndexBuffer[primIdx];

    const float3 p0 = vertexBuffer[v_idx.x];
    const float3 p1 = vertexBuffer[v_idx.y];
    const float3 p2 = vertexBuffer[v_idx.z];

    /* Fill in Differential Geometry |dgShading|. */

    /* Default uv values, if there are no texcoords from obj available. */
    float2 uv0 = make_float2(0.f, 0.f);
    float2 uv1 = make_float2(1.f, 0.f);
    float2 uv2 = make_float2(1.f, 1.f);
    if (texcoordBuffer.size())
    {
        /* Get vertex index to |texcoordBuffer|. */
        const int3 vt_idx = texcoordIndexBuffer[primIdx];
        uv0 = texcoordBuffer[vt_idx.x];
        uv1 = texcoordBuffer[vt_idx.y];
        uv2 = texcoordBuffer[vt_idx.z];
    }

    /* Compute deltas for uvs. */
    float du0_2 = uv0.x - uv2.x;
    float du1_2 = uv1.x - uv2.x;
    float dv0_2 = uv0.y - uv2.y;
    float dv1_2 = uv1.y - uv2.y;

    /* Compute deltas for position. */
    float3 dp0_2 = p0 - p2, dp1_2 = p1 - p2;
    float determinant = du0_2 * dv1_2 - dv0_2 * du1_2;
    if (determinant == 0.0f)
    {
        TwUtil::CoordinateSystem(TwUtil::safe_normalize(cross(p2-p0, p1-p0)), dgShading.dpdu, dgShading.dpdv);
    }
    else
    {
        float invdet = 1.0f / determinant;
        dgShading.dpdu = ( dv1_2 * dp0_2 - dv0_2 * dp1_2) * invdet;
        dgShading.dpdv = (-du1_2 * dp0_2 + du0_2 * dp1_2) * invdet;
    }

    /* Interpolate parametric coordinates uv for triangle. */
    float2 barycentrics = rtGetTriangleBarycentrics();
    float b0 = 1.f - barycentrics.x - barycentrics.y;
    dgShading.uv = b0 * uv0 + barycentrics.x * uv1 + barycentrics.y * uv2;

    /* Compute normalized normal for triangle. 
     * -- We use |dp0_2|,|dp1_2| to compute normal which relies on 
     * the winding order of triangle rather than its parameterizations
     * (texture coordinates). */
    dgShading.nn = nGeometry = make_float4(TwUtil::safe_normalize(cross(dp0_2, dp1_2)));

    /* Compute shading differential geometry for shading normal, . */
    if (normalBuffer.size())
    {
        float3 ns = make_float3(0.f);
        float3 &ss = dgShading.dpdu;
        float3 ts = make_float3(0.f);

        /* Get vertex index to |normalBuffer|. */
        const int3 vn_idx = normalIndexBuffer[primIdx];

        /* Interpolate shading normal. */
        ns = b0 * normalBuffer[vn_idx.x] + barycentrics.x * normalBuffer[vn_idx.y] + barycentrics.y * normalBuffer[vn_idx.z];    
        ns = TwUtil::safe_normalize(ns);

        /* Compute shading bitangent. */
        ts = cross(ss, ns);
        if (TwUtil::sqr_length(ts) > 0.f)
        {
            ts = TwUtil::safe_normalize(ts);
            ss = cross(ts, ns);
        }
        else
            TwUtil::CoordinateSystem(ns, ss, ts);

        /* ss is a reference to dgShading.dpdu.
         * -- dgShading.dpdu = ss; */
        dgShading.dpdv = ts;
        dgShading.nn = make_float4(cross(dgShading.dpdu, dgShading.dpdv));
    }

    /* Keep geometric normal on the same side of shading normal. */
    auto dot3f = [](const float4& a, const float4& b) { return a.x * b.x + a.y * b.y + a.z * b.z; };
    nGeometry = (dot3f(nGeometry, dgShading.nn) < 0.f) ? -nGeometry : nGeometry;

    dgShading.dpdu = TwUtil::safe_normalize(dgShading.dpdu);
    dgShading.tn = cross(make_float3(dgShading.nn), dgShading.dpdu);
}

#ifdef OLDCODE_FOR_TRIANGLEMESH
RT_PROGRAM void BoundingBox_TriangleMesh(int primIdx, float result[6])
{
	//note that bouding box program always consider that aabb is specified in object space.
	//never transform aabb into world space
	const int3 v_idx = vertexIndexBuffer[primIdx];

	const float3 v0 = vertexBuffer[v_idx.x];
	const float3 v1 = vertexBuffer[v_idx.y];
	const float3 v2 = vertexBuffer[v_idx.z];
	const float  area = length(cross(v1 - v0, v2 - v0));

	optix::Aabb* aabb = (optix::Aabb*)result;

	if (area > 0.0f && !isinf(area))
	{
		aabb->m_min = fminf(fminf(v0, v1), v2);
		aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
	}
	else
	{
		aabb->invalidate();//Make infinite bounding box
	}
}

RT_PROGRAM void Intersect_TriangleMesh(int primIdx)
{
	const int3 v_idx = vertexIndexBuffer[primIdx];

	const float3 p1 = vertexBuffer[v_idx.x];
	const float3 p2 = vertexBuffer[v_idx.y];
	const float3 p3 = vertexBuffer[v_idx.z];

	float3 e1 = p2 - p1;
	float3 e2 = p3 - p1;
	float3 s1 = optix::cross(ray.direction, e2);
	float divisor = optix::dot(s1, e1);

	if (divisor == 0.0f)
		return;
	float invDivisor = 1.0f / divisor;

	// Compute first barycentric coordinate
	float3 s = ray.origin - p1;
	float b1 = optix::dot(s, s1) * invDivisor;
	if (b1 < 0.0f || b1 > 1.0f)
		return;

	// Compute second barycentric coordinate
	float3 s2 = optix::cross(s, e1);
	float b2 = optix::dot(ray.direction, s2) * invDivisor;
	if (b2 < 0.0f || b1 + b2 > 1.0f)
		return;

	// Compute _t_ to intersection point
	float t = optix::dot(e2, s2) * invDivisor;
	if (rtPotentialIntersection(t))
	{
		float3 dpdu = make_float3(0.f), dpdv = make_float3(0.f);

		float2 uv0 = make_float2(0.f), uv1 = make_float2(0.f), uv2 = make_float2(0.f);
		if (texcoordBuffer.size() == 0)
		{//No UV found
			uv0.x = 0.f; uv0.y = 0.f;
			uv1.x = 1.f; uv1.y = 0.f;
			uv2.x = 1.f; uv2.y = 1.f;
		}
		else
		{
			const int3 vt_idx = texcoordIndexBuffer[primIdx];
			uv0 = texcoordBuffer[vt_idx.x];
			uv1 = texcoordBuffer[vt_idx.y];
			uv2 = texcoordBuffer[vt_idx.z];
		}

		float du1 = uv0.x - uv2.x;
		float du2 = uv1.x - uv2.x;
		float dv1 = uv0.y - uv2.y;
		float dv2 = uv1.y - uv2.y;
		float3 dp1 = p1 - p3, dp2 = p2 - p3;
		float determinant = du1 * dv2 - dv1 * du2;
		if (determinant == 0.0f)
		{
			// Handle zero determinant for triangle partial derivative matrix
			//todo:a minor problem occured when the trianglemesh doesn't own its UV(all gets 0.0)
			//rtPrintf("Degenerate Triangles\n");
			TwUtil::CoordinateSystem(TwUtil::safe_normalize(cross(e2, e1)), dpdu, dpdv);
		}
		else
		{
			float invdet = 1.0f / determinant;
			dpdu = (dv2 * dp1 - dv1 * dp2) * invdet;
			dpdv = (-du2 * dp1 + du1 * dp2) * invdet;
		}

		//interpolate u,v triangle parametric coordinates
		float b0 = 1 - b1 - b2;
		float tu = b0 * uv0.x + b1 * uv1.x + b2 * uv2.x;
		float tv = b0 * uv0.y + b1 * uv1.y + b2 * uv2.y;

		//Fill in Differential Geometry [dgShading] info of invariable information
		//dgShading.rayEpsilon = 1e-3f * t;//org
		dgShading.rayEpsilon = 1e-3f * t;
		dgShading.uv.x = tu;
		dgShading.uv.y = tv;

		dgShading.dpdu = dpdu;
		dgShading.dpdv = dpdv;
		nGeometry = make_float4(TwUtil::safe_normalize(cross(dgShading.dpdu, dgShading.dpdv)));

		/*flip computed geometry normal if necessary
		 *-- a consistent of geometry normal and shading
		 *normal is indispensible. Detect by yourself.
		 *
		 *In current implementation, the flipping geometry
		 *normal operation would be done automatically if
		 *shading normal avaliable or otherwise, a reverseOrientation
		 *flag is exposed to user, enabling the manually flipping
		 *if necessary.*/
		//if (reverseOrientation)
		//	nGeometry *= -1.f;

		//Tangents are not supported yet.
		if (normalBuffer.size() == 0)
		{
			/*no shading normal provided so as to assist
			 *reverseing normal orientation*/
			if (reverseOrientation)
				nGeometry *= -1.f;

			dgShading.nn = nGeometry;
		}
		else
		{
			//We've calculated barycentric coordinates b0 b1 b2 above
			float3 ns = make_float3(0.f);
			float3 &ss = dgShading.dpdu;//not supported interpolate tangent
			float3 ts = make_float3(0.f);

			/*float3 normalMapSample = 2.0f * make_float3(tex2D(NormalMap, diffGeom.uv.x, diffGeom.uv.y)) - 1.0f;
			float3 worldNormal = normalize(twilight::LocalToWorld(normalMapSample, dgShading.dpdu, dgShading.tn, dgShading.nn));*/

			const int3 vn_idx = normalIndexBuffer[primIdx];
			float3 interpolatedNormal = b0 * normalBuffer[vn_idx.x] + b1 * normalBuffer[vn_idx.y] + b2 * normalBuffer[vn_idx.z];
			ns = TwUtil::safe_normalize(interpolatedNormal);

			/*flip shading normal in accordance with
			 *shading normal -- performance penalty*/
			if (TwUtil::dot(ns, nGeometry) < 0.f)
				nGeometry *= -1.f;

			ts = cross(ss, ns);
			if (TwUtil::sqr_length(ts) > 0.f)
			{
				ts = TwUtil::safe_normalize(ts);
				ss = cross(ts, ns);
			}
			else
				TwUtil::CoordinateSystem(ns, ss, ts);

			dgShading.dpdu = ss;
			dgShading.dpdv = ts;
			dgShading.nn = make_float4(cross(dgShading.dpdu, dgShading.dpdv));
		}
		// Compute shading coordinates
		dgShading.dpdu = TwUtil::safe_normalize(dgShading.dpdu);
		dgShading.tn = cross(make_float3(dgShading.nn), dgShading.dpdu);

		rtReportIntersection(/*material_buffer[primIdx]*/0);
	}
}
#endif // OLDCODE_FOR_TRIANGLEMESH


