#pragma once

#include <memory>

#include "colvillea/Module/Camera/Camera.h"
#include "colvillea/Application/GlobalDefs.h"
#include "colvillea/Application/TWAssert.h"
#include "colvillea/Device/Toolkit/Utility.h"

#include <optixu_math_namespace.h>


/**
 * @brief CameraController class provides storage to
 * CameraInfo and help camera interaction with Input 
 * GUI event by correctly updating camera's lookat
 * vectors.
 */
class CameraController
{
private:
    struct CameraInfo
    {
        // Information of mouse position and change delta on screen
        optix::int2 basePosition;
        optix::int2 deltaValue;

        // Depict camera coordinates in spherical coordinates (Phi, Theta, Radius)
        float phi, theta;     // in radian
        float radialDistance; // or equivalently in spherical coordinates, radius
        
        // Depict camera coordinates in vector (Eye position, LookAt direction)
        optix::float3 eye;    // todo:delete |eye| property, which is redundant and could be computed by converting spherical coordinates of camera position to cartesian coordinates.
        optix::float3 lookAtDestination; //todo:pick a more appropriate name.

        // Feedback vector from cameraToWorld matrix
        optix::float3 left;
        optix::float3 up;

        // Parameters for interaction
        float trackSpeed;
        float minimumRadialDistance = 1e-4f;

    public:
        CameraInfo() : 
            basePosition(optix::make_int2(0,0)), deltaValue(optix::make_int2(0,0)),
            phi(.88f), theta(.64f), trackSpeed(25.f), radialDistance(5.f)
        {
            this->eye = optix::make_float3(1.436f, -1.304f, -0.030f);
            this->lookAtDestination = optix::make_float3(0.326f, -0.244f, -0.482f);
            this->left = this->up = optix::make_float3(0.f);
        }
    };

public:
    /** Scoped enum specifying camera motion type.
     *
     */
    enum class CameraMotionType
    {
        None,
        Pan,
        Orbit,
        Dolly
    };

    /** Scoped enum specifying mouse input action type.
     *
     */
    enum class InputMouseActionType
    {
        LeftMouse    = 0x001,
        MiddleMouse  = 0x002,
        RightMouse   = 0x004,

        Down         = 0x008,
        Release      = 0x010,

        MouseDown    = LeftMouse | MiddleMouse | RightMouse | Down,
        MouseRelease = LeftMouse | MiddleMouse | RightMouse | Release
    };

public:
    CameraController() = default;

    CameraController(std::shared_ptr<Camera> camera, const int filmWidth, const int filmHeight):
        m_cameraMotionType(CameraMotionType::None), 
        m_filmWidth(filmWidth), m_filmHeight(filmHeight),
        m_camera(camera)
    {
        /* Setup camera before launch. */
        this->updateCameraInfo();
    }

    CameraInfo getCameraInfo() const
    {
        return this->m_cameraInfo;
    }
    void setCameraInfo(CameraInfo &cameraInfo)
    {
        /* Update |eye| and |lookAtDestination| from cameraInfo paramter. */
        this->m_cameraInfo.eye               = cameraInfo.eye;
        this->m_cameraInfo.lookAtDestination = cameraInfo.lookAtDestination;

        /* Compute the corresponding |phi|, |theta| and |radialDistance| from |eye| and |lookAtDestination|. */
        this->m_cameraInfo.radialDistance = TwUtil::distance(this->m_cameraInfo.eye, this->m_cameraInfo.lookAtDestination);
        this->m_cameraInfo.phi = TwUtil::sphericalPhi(TwUtil::safe_normalize(this->m_cameraInfo.eye - this->m_cameraInfo.lookAtDestination));
        this->m_cameraInfo.theta = TwUtil::sphericalTheta(TwUtil::safe_normalize(this->m_cameraInfo.eye - this->m_cameraInfo.lookAtDestination));


        //--
        const float cosPhi = cosf(this->m_cameraInfo.phi);
        const float sinPhi = sinf(this->m_cameraInfo.phi);
        const float cosTheta = cosf(this->m_cameraInfo.theta);
        const float sinTheta = sinf(this->m_cameraInfo.theta);

        this->m_cameraInfo.phi   /= 2.0f * M_PIf;
        this->m_cameraInfo.theta /= M_PIf;

        optix::float3 normal = optix::make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
        auto lenNormal = length(normal);
        //optix::float3 normal = optix::make_float3(cosPhi * sinTheta, cosTheta, sinPhi * sinTheta); // "normal", unit vector from origin to spherical coordinates (phi, theta)
        this->m_cameraInfo.eye = this->m_cameraInfo.lookAtDestination + this->m_cameraInfo.radialDistance * normal;
        TW_ASSERT(fabsf(this->m_cameraInfo.eye.x - cameraInfo.eye.x) < 1e-3f);
        //-- 
        //auto xx = this->m_cameraInfo.lookAtDestination + this->m_cameraInfo.radialDistance * TwUtil::safe_normalize(this->m_cameraInfo.eye - this->m_cameraInfo.lookAtDestination);

        /* Update camera transformation using |eye| and |lookAtDestination|. */
        this->m_camera->setCameraToWorld(this->m_cameraInfo.eye, this->m_cameraInfo.lookAtDestination);

        /* Fetching feedback vectors after setting cameraToWorld. */
        this->m_cameraInfo.left = this->m_camera->GetLeftVector();
        this->m_cameraInfo.up = this->m_camera->GetUpVector();
    }
    

    /**
     * @brief Handle mouse input from ImGui so as to
     * adjust camera motion and invoke corresponding 
     * motion function to respond.
     * 
     * @see orbit(),dolly(),pan()
     */
    void handleInputGUIEvent(const InputMouseActionType mouseAction, const optix::int2 screenPos);

private:
    /**
     * @brief Update mouse position and delta value
     * relative to the last position given |screenPos|.
     * @return Whether the mouse position has changed.
     */
	inline bool updateMouseInfo(optix::int2 screenPos);

    /**
     * @brief Mouse motion group functions.
     */
    inline void orbit(optix::int2 screenPos);
    inline void dolly(optix::int2 screenPos);
    inline void pan(optix::int2 screenPos);

    inline void updateCameraInfo();

private:
    CameraInfo m_cameraInfo;
    CameraMotionType m_cameraMotionType;
    std::shared_ptr<Camera> m_camera;

    

    // todo:get a pointer to Application and fetch film resolution from it
    int m_filmWidth, m_filmHeight;
};

inline bool CameraController::updateMouseInfo(optix::int2 screenPos)
{
    if (this->m_cameraInfo.basePosition != screenPos)
	{
        this->m_cameraInfo.deltaValue = screenPos - this->m_cameraInfo.basePosition;
        this->m_cameraInfo.basePosition = screenPos;
		return true; 
	}
	return false;
}

inline void CameraController::orbit(optix::int2 screenPos)
{
	if (updateMouseInfo(screenPos))
	{
        /* Figure out the new value of phi and theta. */
        this->m_cameraInfo.phi += static_cast<float>(this->m_cameraInfo.deltaValue.x) / this->m_filmWidth;

        /* Clamp phi. */
		if (this->m_cameraInfo.phi < 0.0f)
		{
            this->m_cameraInfo.phi += 1.0f;
		}
		else if (1.0f < this->m_cameraInfo.phi)
		{
            this->m_cameraInfo.phi -= 1.0f;
		}

        this->m_cameraInfo.theta -= static_cast<float>(this->m_cameraInfo.deltaValue.y) / this->m_filmHeight;

        /* Clamp theta. */
		if (this->m_cameraInfo.theta < 0.0f)
		{
            this->m_cameraInfo.theta = std::numeric_limits<float>::epsilon();
		}
		else if (1.0f - std::numeric_limits<float>::epsilon() < this->m_cameraInfo.theta)
		{
			/* Solved the problem that when theta becomes 1.f, LookAt() could be wrong, leading to swap side of view. */
            this->m_cameraInfo.theta = 1.0f - std::numeric_limits<float>::epsilon();
		}

        this->updateCameraInfo();
	}
}

inline void CameraController::pan(optix::int2 screenPos)
{
	if (updateMouseInfo(screenPos))
	{
        /* Figure out the new looking at destination position. */

        optix::float2 uv = optix::make_float2(this->m_cameraInfo.deltaValue.x, this->m_cameraInfo.deltaValue.y) / this->m_cameraInfo.trackSpeed; // todo:check consistence with orbit's calculation, divided by film resolution.
		this->m_cameraInfo.lookAtDestination = this->m_cameraInfo.lookAtDestination - 
            uv.x * this->m_cameraInfo.left + uv.y * this->m_cameraInfo.up;

        this->updateCameraInfo();
	}
}

inline void CameraController::dolly(optix::int2 screenPos)
{
	if (updateMouseInfo(screenPos))
	{
		/* Figure out the new radial distance to looking at destination. */
		float w = static_cast<float>(this->m_cameraInfo.deltaValue.y) / this->m_cameraInfo.trackSpeed;// todo:check consistence with orbit's calculation, divided by film resolution.
                                   // todo:use euclidean distance to compute w.

        this->m_cameraInfo.radialDistance -= w;
        /* Clamp radialDistance to avoid swapping side of view. */
		if (this->m_cameraInfo.radialDistance < this->m_cameraInfo.minimumRadialDistance) 
		{
            this->m_cameraInfo.radialDistance = 0.0001f;
		}

        this->updateCameraInfo();
	}
}

inline void CameraController::updateCameraInfo()
{
	const float cosPhi   = cosf(this->m_cameraInfo.phi * 2.0f * M_PIf);
	const float sinPhi   = sinf(this->m_cameraInfo.phi * 2.0f * M_PIf);
	const float cosTheta = cosf(this->m_cameraInfo.theta * M_PIf);
	const float sinTheta = sinf(this->m_cameraInfo.theta * M_PIf);

	optix::float3 normal = optix::make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
	//optix::float3 normal = optix::make_float3(cosPhi * sinTheta, cosTheta, sinPhi * sinTheta); // "normal", unit vector from origin to spherical coordinates (phi, theta)
	this->m_cameraInfo.eye = this->m_cameraInfo.lookAtDestination + this->m_cameraInfo.radialDistance * normal;

    this->m_camera->setCameraToWorld(this->m_cameraInfo.eye, this->m_cameraInfo.lookAtDestination);

    /* Fetching feedback vectors after setting cameraToWorld. */
    this->m_cameraInfo.left = this->m_camera->GetLeftVector();
    this->m_cameraInfo.up   = this->m_camera->GetUpVector();
}

//Camera binding functions ended
//////////////////////////////////////////////////////////////////////////