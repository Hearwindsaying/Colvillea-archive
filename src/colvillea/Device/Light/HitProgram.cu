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

//////////////////////////////////////////////////////////////////////////
//AnyHit program for shadow ray
RT_PROGRAM void AnyHit_ShadowRay_Shape(void)
{
	prdShadow.blocked = 1;
	rtTerminateRay();
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

