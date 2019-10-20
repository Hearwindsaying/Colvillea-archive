#include <optix_world.h>
#include <optix_device.h>
#include <optixu_math_namespace.h>

#include "../Toolkit/CommonStructs.h"
#include "../Toolkit/Utility.h"

using namespace optix;
using namespace CommonStructs;

//////////////////////////////////////////////////////////////////////////
//Forward declarations:
//system variables:->Context
rtDeclareVariable(PerRayData_shadow, prdShadow, rtPayload, );

//alphaTexture:->Material
rtDeclareVariable(int, alphaTexture, , ) = RT_TEXTURE_ID_NULL;//todo:review the initialize value
rtDeclareVariable(DifferentialGeometry, dgShading, attribute dgShading, );

//Deprecated since OptiX 6.0
//RT_PROGRAM void AnyHit_ShadowRay_Geometry(void)
//{
//    prdShadow.blocked = 1;
//    rtTerminateRay();
//}

/**
 * @brief ClosestHit program for shadow ray. Starting
 * from OptiX 6.0, it's recommended using 
 * ClosestHit & rtTrace for shadow-like ray and built-in
 * triangles. We can disable anyhit completely if there
 * is no alpha texture for cutout material in scene.
 * 
 * @note When RTrayflags::RT_RAY_FLAG_TERMINATE_ON_FIRST_HIT
 * is set, ray is terminated after first hit, which is reported
 * as closest hit as well and call the corresponding ClosestHit
 * program.
 */
RT_PROGRAM void ClosestHit_ShadowRay_GeometryTriangles()
{
    prdShadow.blocked = 1;
}

//////////////////////////////////////////////////////////////////////////
//AnyHit program for shadow ray with cutout opacity support
RT_PROGRAM void AnyHit_ShadowRay_TriangleMesh_Cutout()
{
	//assert that opacity mask texture has been correctly loaded
	if (alphaTexture != RT_TEXTURE_ID_NULL)
	{
		if (rtTex2D<float4>(alphaTexture, dgShading.uv.x, dgShading.uv.y).x == 0.f)
		{
			rtIgnoreIntersection();
		}
		else
		{
			prdShadow.blocked = 1;
			rtTerminateRay();
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//AnyHit program for radiance ray/pt ray for cutout opacity support
RT_PROGRAM void AnyHit_RDPT_TriangleMesh_Cutout()
{
	//assert that opacity mask texture has been correctly loaded
	if (alphaTexture != RT_TEXTURE_ID_NULL)
	{
		/*rtPrintf("%f uv%f %f\n", rtTex2D<float4>(alphaTexture, dgShading.uv.x, dgShading.uv.y).x,dgShading.uv.x,dgShading.uv.y);*/
		if (rtTex2D<float4>(alphaTexture, dgShading.uv.x, dgShading.uv.y).x == 0.f)
		{
			//rtPrintf("ignored!");
			rtIgnoreIntersection();
		}
		//else this is a valid intersection
	}
}

