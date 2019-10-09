#include "Camera.h"
#include "../../Device/Toolkit/Utility.h"

#include "../../Application/Application.h"


#include <iostream>
#include <optixu_math_namespace.h>

using namespace optix;
using namespace TwUtil;


void Camera::updateCameraMatrices()
{
    this->m_rayGenerationProgram["RasterToCamera"]->setMatrix4x4fv(false, this->m_rasterToCamera.getData());//todo:this is redundant and should be done once.
    this->m_rayGenerationProgram["CameraToWorld"]->setMatrix4x4fv(false, this->m_cameraToWorld.getData());

    /* Do some reset rendering parameters stuff. */
    /*this->m_resetRenderParameters();*/ // use observer pattern.
    //todo:delete this
    this->m_application->resetRenderParams();
}

Camera::Camera(const optix::Matrix4x4 & cam2world, float fov, float filmWidth, float filmHeight):m_cameraToWorld(cam2world),m_filmWidth(filmWidth),m_filmHeight(filmHeight)
{
	if (hasScale(cam2world))
		std::cout << "[Warning] " << "Scaling detected in cameraToWorld transformation!" << std::endl;

	/* Calculate projection matrix. */
	this->m_cameraToScreen = GetPerspectiveMatrix(fov, 1e-2f, 1000.f);


    float pminx, pminy, pmaxx, pmaxy;
	float aspectRatio = this->m_filmWidth / this->m_filmHeight;
    if (aspectRatio > 1.f)
    {
        pminx = -aspectRatio; pminy = -1.f;
        pmaxx = aspectRatio;  pmaxy = 1.f;
    }
    else
    {
        pminx = -1.f, pminy = -1.f / aspectRatio;
        pmaxx = 1.f,  pmaxy = 1.f  / aspectRatio;
    }

    /*Remain Y's normal due to the launch's coordiante system*/

    this->m_screenToRaster = Matrix4x4::scale(make_float3(this->m_filmWidth, this->m_filmHeight, 1.f)) *
                             Matrix4x4::scale(make_float3(1.f / (pmaxx - pminx), 1.f / (pmaxy - pminy), 1.f)) *
                             Matrix4x4::translate(make_float3(-pminx, pmaxy, 0.f));
    this->m_rasterToScreen = this->m_screenToRaster.inverse();
    this->m_rasterToCamera = Matrix4x4(this->m_cameraToScreen.inverse()) * this->m_rasterToScreen;
}



Matrix4x4 Camera::createCameraToWorldMatrix(const optix::float3 & pos, const optix::float3 & look, const optix::float3 & up)
{
	float m[16];
	m[3] = pos.x; m[7] = pos.y; m[11] = pos.z; m[15] = 1.0f;

	float3 dir = safe_normalize(look - pos);

	// Carefully deal with the situation when cross(up,dir) is degenerated.
	// Initialize first three columns of viewing matrix.
	if (optix::length(optix::cross(safe_normalize(up), dir)) == 0)
	{
        std::cout << "[Info] Up vector (" << up.x << "," << up.y << "," << up.z << ") and direction vector (" << dir.x << "," << dir.y << "," << dir.z << ")points to the same direction! Reset to identity matrix." << std::endl;
		return Matrix4x4::identity();
	}
	float3 left = safe_normalize(optix::cross(safe_normalize(up), dir));
	float3 newup = optix::cross(dir, left);

	m[0] = left.x;  m[4] = left.y;  m[8] = left.z;  m[12] = 0;
	m[1] = newup.x; m[5] = newup.y; m[9] = newup.z; m[13] = 0;
	m[2] = dir.x;   m[6] = dir.y;   m[10] = dir.z;  m[14] = 0;

	Matrix4x4 cam2world(m);
	return cam2world;
}


