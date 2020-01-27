#pragma once

#define  CL_CHECK_MEMORY_LEAKS
#ifdef CL_CHECK_MEMORY_LEAKS
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#define CL_CHECK_MEMORY_LEAKS_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new CL_CHECK_MEMORY_LEAKS_NEW
#endif

#include <optixu_math_namespace.h>
#include <optixu_matrix_namespace.h>
#include <optix_world.h>
#include <optix_host.h>

#include <functional>
#include <map>

#include "colvillea/Application/GlobalDefs.h"
#include "colvillea/Application/TWAssert.h"


class Application;

/**
 * @brief Camera class provides interface to prepare
 * statitcs, helping generate ray in RayGeneration
 * program. It's also responsible for interaction
 * with CameraControl via getter and setter methods
 * for camear matrix.
 * 
 * It's served as PerspectiveCamera with pinhole style.
 */
class Camera
{
public:
    /**
     * @brief Constructor for creating Camera. 
     * @param cam2world CameraToWorld matrix could be
     * found by utility function Camera::createCameraToWorldMatrix()
     * @see Camera::createCameraToWorldMatrix()
     */
 	Camera(optix::Context context, const std::map<std::string, optix::Program> &programsMap,
        /*std::function<void()> resetRenderParam,*/ Application * application,
		   const optix::Matrix4x4 & cam2world, float fov, float filmWidth, float filmHeight,
           float focalDistance, float lensRadius) :
		Camera(context, cam2world, fov, filmWidth, filmHeight, focalDistance, lensRadius)
 	{
        this->m_application = application;
        TW_ASSERT(this->m_application);
        /*this->m_resetRenderParameters = resetRenderParam;*/

 		/* Setup RayGeneration program for OptiX. */
 		auto programItr = programsMap.find("RayGeneration_PinholeCamera");
        this->m_rayGenerationProgram = programItr->second;
 
 		TW_ASSERT(programItr != programsMap.end());
 		context->setRayGenerationProgram(toUnderlyingValue(RayGenerationEntryType::Render), this->m_rayGenerationProgram);

        context["focalDistance"]->setFloat(this->m_focalDistance);
        context["lensRadius"]->setFloat(this->m_lensRadius);
 	}

    /**
     * @brief Helper utlity for converting lookAt directions to 
     * CameraToWorld matrix.
     */
    static optix::Matrix4x4 createCameraToWorldMatrix(const optix::float3 & pos, const optix::float3 & look, const optix::float3 & up);

    /**
     * @brief Get |left| vector from cameraToWorld matrix.
     */
    optix::float3 GetLeftVector() const
    {
        const float *m = this->m_cameraToWorld.getData();
        return optix::make_float3(m[0], m[4], m[8]);
    }
    /**
     * @brief Get |up| vector from cameraToWorld matrix.
     */
    optix::float3 GetUpVector() const
    {
        const float *m = this->m_cameraToWorld.getData();
        return optix::make_float3(m[1], m[5], m[9]);
    }
    /**
     * @brief Get |direction| vector from cameraToWorld matrix.
     */
    optix::float3 GetDirectionVector() const 
    {
        const float *m = this->m_cameraToWorld.getData();
        return optix::make_float3(m[2], m[6], m[10]);
    }

    /**
     * @brief Setter for cameraToWorld.
     */
    void setCameraToWorld(const optix::Matrix4x4 &cam2world)
    {
        this->m_cameraToWorld = cam2world;
        updateCameraMatrices();
    }

    /**
     * @brief Convinent setter using (position,lookAt,up) vector
     * to create cameraToWorld matrix.
     */
    void setCameraToWorld(const optix::float3 & pos, const optix::float3 & look, const optix::float3 & up = optix::make_float3(0.f, 0.f, 1.f))
    {
        this->m_cameraToWorld = Camera::createCameraToWorldMatrix(pos, look, up);
        updateCameraMatrices();
    }

    float getLensRadius() const
    {
        return this->m_lensRadius;
    }
    
    void setLensRadius(float lensRadius)
    {
        TW_ASSERT(lensRadius >= 0.f);
        this->m_lensRadius = lensRadius;
        this->m_context["lensRadius"]->setFloat(this->m_lensRadius);
    }

    float getFocalDistance() const
    {
        return this->m_focalDistance;
    }

    void setFocalDistance(float focalDistance)
    {
        TW_ASSERT(focalDistance >= 0.f);
        this->m_focalDistance = focalDistance;
        this->m_context["focalDistance"]->setFloat(this->m_focalDistance);
    }

    void updateCameraMatrices(); //todo:make inline

private:
    /**
     * @brief Private constructor for initializing camera matrices.
     */
	Camera(optix::Context context, const optix::Matrix4x4 & cam2world, float fov, float filmWidth, float filmHeight, float focalDistance, float lensRadius);

private:
    Application *m_application;
    optix::Context m_context;

	//Matrix group
	optix::Matrix4x4 m_cameraToWorld;
	optix::Matrix4x4 m_cameraToScreen, m_rasterToCamera;
	optix::Matrix4x4 m_screenToRaster, m_rasterToScreen;

	float m_filmWidth, m_filmHeight;

    /// Depth of Field parameters
    float m_lensRadius, m_focalDistance;

    optix::Program m_rayGenerationProgram;

    /** Stores a callable object for invoking after
     * updating Camera matrices. This could be useful
     * when we want to reset some states for rendering,
     * such as resetting |sysIterationIndex| and launch
     * initFilter.
     */
    /*std::function<void()> m_resetRenderParameters;*/
};



