#include <optix_world.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"
#include "../Geometry/Quad.h"
#include "LightUtil.h"

using namespace optix;
using namespace TwUtil;

#ifndef TWRT_DELCARE_QUADLIGHT
#define TWRT_DELCARE_QUADLIGHT
rtBuffer<CommonStructs::QuadLight> quadLightBuffer;
#endif

rtDeclareVariable(float, sysSceneEpsilon, , );




/**
 * @brief Given a shading point |point|, evaluate direct lighting
 * contribution from the specific light implied by |lightBufferIndex|,
 * assuming that there is no occlusive object between the point
 * and light.
 *
 * @param[in]  point            shading point
 * @param[in]  rayEpsilon       epsilon parameter used to make shadow ray
 * @param[in]  lightSample      uniformly distributed light samples
 * @param[in]  lightBufferIndex index to quadlight buffer
 *
 * @param[out] outwi            contribution direction from |point| to the sampled point on light
 * @param[out] outpdf           sampled pdf in solid angle measure
 * @param[out] outShadowRay     shadow ray that could be used for testing occlusion
 *
 * @return contribution radiance from light
 */
RT_CALLABLE_PROGRAM float4 Sample_Ld_Quad(const float3 &point, const float & rayEpsilon, float3 & outwi, float & outpdf, float2 lightSample, uint lightBufferIndex, Ray & outShadowRay)
{
    const CommonStructs::QuadLight &quadLight = quadLightBuffer[lightBufferIndex];
    float3 sampledPoint = Quad_Sample(quadLight, lightSample, &outpdf); /* outpdf for temporary storage, it's with respect to area now! */
    outwi = sampledPoint - point; /* outwi for temporary storage, it's unnormalized now! */
    if (dot(outwi, outwi) == 0.f) /* use dot to fetch sqrlength to avoid sqrt operation */
    {
        outpdf = 0.f;
        return make_float4(0.f);
    }
    else
    {
        outwi = safe_normalize(outwi);
        optix::float3 lightNormal = TwUtil::safe_normalize(TwUtil::xfmNormal(
            optix::make_float3(0.f, 0.f, (quadLight.reverseOrientation ? -1.f : 1.f)), quadLight.worldToLight));

        /* Convert pdf from area measure to solid angle measure. */
        outpdf *= sqr_length(sampledPoint - point) / fabs(dot(lightNormal, -outwi));

        if (isinf(outpdf))
            outpdf = 0.f;

        outShadowRay = MakeShadowRay(point, rayEpsilon, sampledPoint, 1e-3f);

        return Le_QuadLight(quadLight, -outwi);
    }

//Old code:
#if 0
#define USE_OLDIMPL_SAMPLEQUADLIGHT
#ifdef USE_APPSR_SAMPLEQUADLIGHT
    SphQuad curThreadSphQuad = sphQuad[0];/*copy old data first*/


    sphQuadInit(curThreadSphQuad, point);/*init sphquad*/

    outLightSamplePoint = sphQuadSample(curThreadSphQuad, lightSample, point);/*todo:try using local light space*/
    /*if ((fabsf(fabsf(point.x)) <= 1e-3f) && (fabsf(fabsf(point.y)) <= 1e-3f) &&(fabsf(0.466544f - fabsf(point.z)) <= 1e-3f))
        rtPrintf("%f %f %f\n", outLightSamplePoint.x, outLightSamplePoint.y, outLightSamplePoint.z);*/
    outwi = safe_normalize(outLightSamplePoint - point);
    outpdf = 1.f / curThreadSphQuad.S;
    outShadowRay = MakeShadowRay(point, rayEpsilon, outLightSamplePoint, 1e-3f);

    float3 lightNormal = xfmNormal(make_float3(0.f, 0.f, -1.f),
        quadLight.lightToWorld);
    return TwUtil::L_QuadLight(lightNormal, -outwi, quadLight);
#endif // USE_APPSR_SAMPLEQUADLIGHT


#ifdef USE_OLDIMPL_SAMPLEQUADLIGHT
//     /*--The old implementation which samples uniformly from area and transform to solid angle measure*/
//     ///ShapeSet Sample begins///
//     /*Calculate the sampled point on the quad*/
//     /*Calculation performs in Local Light Space*/
//     float3 p = make_float3((lightSample.x - .5f) * quadLight.scaleXY.x, (lightSample.y - .5f) * quadLight.scaleXY.y, 0.f);
// 
//     p = xfmPoint(p, quadLight.lightToWorld);
//     ///ShapeSet Sample ends///
// 
//     outLightSamplePoint = p;
// 
//     /*Get wi direction*/
//     float dist = distance(p, point);
//     outwi = safe_normalize(p - point);//outwi is relative to shading coordinate system;not light.
// 
//     //use quadLight's normal to calculate cosTheta
//     float3 lightNormal = xfmNormal(make_float3(0.f, 0.f, -1.f),
//         quadLight.lightToWorld);
//     outpdf = sqr_length(p - point) / (fabs(dot(lightNormal, -outwi)) * (quadLight.scaleXY.x * quadLight.scaleXY.y));
//     outShadowRay = MakeShadowRay(point, rayEpsilon, outLightSamplePoint, 1e-3f);
// 
//     return TwUtil::L_QuadLight(lightNormal, -outwi, quadLight);
//     /*--The old implementation which samples uniformly from area and transform to solid angle measure*/
#endif

#endif
}

/**
 * @brief Compute pdf of sampling a particular point on quadlight.
 * 
 * @param[in]  p         reference point 
 * @param[in]  wi        sampling direction from |p| to sampled point on quadlight surface
 * @param[in]  lightId   index to the specific quadlight
 * @param[out] shadowRay shadow ray for detect the occlusion from p to sampled point on light surface
 * 
 * @return pdf in solid angle measure
 */
RT_CALLABLE_PROGRAM float LightPdf_Quad(const float3 & p, const float3 & wi, const int lightId, Ray &shadowRay)
{
    const CommonStructs::QuadLight &quadLight = quadLightBuffer[lightId];

    /* Spawn ray from reference point. */
    optix::Ray detectedRay = optix::make_Ray(p, wi, toUnderlyingValue(CommonStructs::RayType::Detection), 1e-3f, RT_DEFAULT_MAX); /* note that RayType::Detection is unnecessarily needed. */

    /* Perform detective intersection test manually. 
      -- It's really simple to intersect one single
         specific quad given a ray so we do not turn
         to OptiX for rtTrace. Meanwhile, it's not
         possible for OptiX to store a buffer of
         rtObject for intersection. */
    float tHit = 0.f;
    float3 nn = make_float3(0.f), sampledPoint = make_float3(0.f);
    if (!Quad_FastIntersect(quadLight, detectedRay, &tHit, &nn, &sampledPoint))
    {/* todo: use orientation test (dot) to avoid _FastIntersect. */
        return 0.f;
    }

    /* Make shadow ray for later detection. */
    shadowRay = optix::make_Ray(p, wi, toUnderlyingValue(CommonStructs::RayType::Shadow), sysSceneEpsilon, tHit); //todo:review epsilon

    /* Convert pdf in area measure to solid angle measure. */
    float pdf = sqr_length(p - sampledPoint) * quadLight.invSurfaceArea / fabsf(dot(nn, -wi));
    if (isinf(pdf))
        return 0.f;
    return pdf;
}


#if 0
//rtDeclareVariable(rtObject, quadLightObject, , );
//
////rtDeclareVariable(SphQuad, sphQuad, , );
//rtBuffer<SphQuad, 1> sphQuad;
//rtDeclareVariable(uint2, sysLaunch_index, rtLaunchIndex, );//Current launch index
///**
// * @brief initialize sphQuad used to sample quadlight
// * @param o local reference system origin which could be passed in current shading point
// */
//static __device__ __inline__ void sphQuadInit(SphQuad &outSphQuad, const float3 &o)
//{
//	outSphQuad.o = o;
//	
//	float3 d = outSphQuad.s - outSphQuad.o;
//	outSphQuad.z0 = dot(d, outSphQuad.z);
//
//	/*flip 'z' to make it point against 'Q'*/
//	if (outSphQuad.z0 > 0.f)
//	{
//		outSphQuad.z *= -1.f;
//		outSphQuad.z0 *= -1.f;
//	}
//	outSphQuad.z0sq = outSphQuad.z0 * outSphQuad.z0;
//	outSphQuad.x0 = dot(d, outSphQuad.x);
//	outSphQuad.y0 = dot(d, outSphQuad.y);
//	outSphQuad.x1 = outSphQuad.x0 + quadLight.scaleXY.x;
//	outSphQuad.y1 = outSphQuad.y0 + quadLight.scaleXY.y;
//	outSphQuad.y0sq = outSphQuad.y0 * outSphQuad.y0;
//	outSphQuad.y1sq = outSphQuad.y1 * outSphQuad.y1;
//
//	
//
//	/*create vectors to four vertices*/
//	float3 v00 = make_float3(outSphQuad.x0, outSphQuad.y0, outSphQuad.z0);
//	float3 v01 = make_float3(outSphQuad.x0, outSphQuad.y1, outSphQuad.z0);
//	float3 v10 = make_float3(outSphQuad.x1, outSphQuad.y0, outSphQuad.z0);
//	float3 v11 = make_float3(outSphQuad.x1, outSphQuad.y1, outSphQuad.z0);
//
//	/*compute normals to edges*/
//	float3 n0 = safe_normalize(cross(v00, v10));
//	float3 n1 = safe_normalize(cross(v10, v11));
//	float3 n2 = safe_normalize(cross(v11, v01));
//	float3 n3 = safe_normalize(cross(v01, v00));
//
//	/*compute internal angles (gamma_i)*/
//	float g0 = acosf(-dot(n0, n1));
//	float g1 = acosf(-dot(n1, n2));
//	float g2 = acosf(-dot(n2, n3));
//	float g3 = acosf(-dot(n3, n0));
//
//	/*compute predefined constants*/
//	outSphQuad.b0 = n0.z;
//	outSphQuad.b1 = n2.z;
//	outSphQuad.b0sq = outSphQuad.b0 * outSphQuad.b0;
//	outSphQuad.k = 2 * M_PIf - g2 - g3;
//
//	/*compute solid angle from internal angles*/
//	outSphQuad.S = g0 + g1 - outSphQuad.k;
//
//	//if (sysLaunch_index == make_uint2(384, 384))
//	//	rtPrintf("%f %f %f %f %f\n", outSphQuad.x0, outSphQuad.x1, outSphQuad.y0, outSphQuad.y1, outSphQuad.S);
//}
//
///**
// * @brief sample quadlight using precomputed sphQuad
// * @param o local reference system origin which could be passed in current shading point
// * @return sample point on the quadlight in global world coorindates
// */
//static __device__ __inline__ float3 sphQuadSample(SphQuad &outSphQuad, const float2 &urand, const float3 &point)
//{
//#define EPSILON 1e-6f
//	/*1.compute 'cu'*/
//	float au = urand.x * outSphQuad.S + outSphQuad.k;
//	float fu = (cosf(au) * outSphQuad.b0 - outSphQuad.b1) / sinf(au);
//	float cu = 1.f / sqrtf(fu*fu + outSphQuad.b0sq) * (fu > 0.f ? +1.f : -1.f);
//	cu = clamp(cu, -1.f, 1.f);
//	
//	/*2.compute 'xu'*/
//	float xu = -(cu * outSphQuad.z0) / sqrtf(1.f - cu*cu);
//	xu = clamp(xu, outSphQuad.x0, outSphQuad.x1);
//
//	/*3.compute 'yv'*/
//	float d = sqrtf(xu*xu + outSphQuad.z0sq);
//	float h0 = outSphQuad.y0 / sqrtf(d*d + outSphQuad.y0sq);
//	float h1 = outSphQuad.y1 / sqrtf(d*d + outSphQuad.y1sq);
//	float hv = h0 + urand.y * (h1 - h0), hv2 = hv * hv;
//	float yv = (hv2 < 1 - EPSILON) ? (hv*d) / sqrtf(1 - hv2) : outSphQuad.y1;
//
//	/*if ((fabsf(fabsf(point.x)) <= 1e-3f) && (fabsf(fabsf(point.y)) <= 1e-3f) && (fabsf(0.466544f - fabsf(point.z)) <= 1e-3f))
//		rtPrintf("%f %f %f %f %f %f|o:%f %f %f|po:%f %f %f\n", xu,yv, outSphQuad.z0, outSphQuad.S,outSphQuad.x0,outSphQuad.x1,outSphQuad.o.x,outSphQuad.o.y,outSphQuad.o.z,point.x,point.y,point.z);*/
//
//	/*4.transform (xu,yv,z0) to world coordiantes*/
//	return (outSphQuad.o + xu * outSphQuad.x + yv * outSphQuad.y + outSphQuad.z0 * outSphQuad.z);
//	
//#undef EPSILON
//}




//lightId doesn't work currently.
//RT_CALLABLE_PROGRAM float LightPdf_Quad(const float3 & p, const float3 & wi, const int lightId)
//{
//#ifdef USE_APPSR_SAMPLEQUADLIGHT
//	optix::Ray ray = optix::make_Ray(p, wi, toUnderlyingValue(CommonStructs::RayType::Detection), 1e-3f, RT_DEFAULT_MAX);
//
//	PerRayData_pt prd;
//	prd.validPT = 0;//using validPT as shadow ray
//
//	rtTrace(quadLightObject, ray, prd);
//
//	if (prd.validPT == 1)
//	{//Hit the emissive plane(quadLight) itself:
//		//if (sphQuad[0].S == -1.f)//incorrect,will be changed to non-1,deleteme:review this issue
//		//{
//		//	sphQuadInit(p);/*init sphquad to get S*/
//		//}
//		//if (sysLaunch_index == make_uint2(384, 384))
//		//	rtPrintf("[Info]solid angle:%f\n", sphQuad[0].S);
//		//sphQuadInit(p);
//		//return 1.f / sphQuad[0].S;
//	}
//
//	return 0.f;
//#endif //USE_APPSR_SAMPLEQUADLIGHT
//
//#ifdef USE_OLDIMPL_SAMPLEQUADLIGHT
//	optix::Ray ray = optix::make_Ray(p, wi, toUnderlyingValue(CommonStructs::RayType::Detection), 1e-3f, RT_DEFAULT_MAX);
//
//	PerRayData_pt prd;
//	prd.validPT = 0;//using validPT as shadow ray
//
//	rtTrace(quadLightObject, ray, prd);
//
//	if (prd.validPT == 1)
//	{//Hit the emissive plane(quadLight) itself:
//		float area = quadLight.scaleXY.x * quadLight.scaleXY.y;
//		float pdf = sqr_length(p - prd.isectP) / (fabs(dot(make_float3(prd.shaderParams.nGeometry.x,prd.shaderParams.nGeometry.y,prd.shaderParams.nGeometry.z), -wi)) * (area));//cosTheta's calculation could be further optimized:prd.nn is unnecessary
//
//		return pdf;
//	}
//	
//	return 0.f;
//#endif
//}
#endif