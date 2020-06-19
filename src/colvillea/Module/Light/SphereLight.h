#pragma once
#include "colvillea/Module/Light/Light.h"

#include <cmath>

#include "colvillea/Module/Geometry/Sphere.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"
#include "colvillea/Device/Toolkit/Utility.h"

class LightPool;

/**
 * @brief Spherical area light describes a diffusely
 * emissive light from a sphere which is able
 * to produce soft shadow and be physically plausible.
 * 
 * @note This class contains description of
 * what a single SpherLight should be. Buffers
 * storing SpherLight is delegated to SceneGraph.
 */
class SphereLight : public Light
{
public:
    /**
     * @brief Factory method for creating a SphereLight instance.
     *
     * @param[in] context
     * @param[in] programsMap  map to store Programs
     * @param[in] color        light color
     * @param[in] intensity    light intensity
     * @param[in] sphereShape  underlying Sphere shape
     */
    static std::unique_ptr<SphereLight> createSphereLight(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3& color, float intensity, std::shared_ptr<Sphere> sphereShape, LightPool *lightPool)
    {
        std::unique_ptr<SphereLight> sphereLight = std::make_unique<SphereLight>(context, programsMap, color, intensity, sphereShape, lightPool);
        return sphereLight;
    }

    SphereLight(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3& color, float intensity, std::shared_ptr<Sphere> sphereShape, LightPool *lightPool) :
        Light(context, programsMap, "Sphere", "Sphere Light", IEditableObject::IEditableObjectType::SphereLight), m_sphereShape(sphereShape), m_intensity(intensity), m_color(color), m_lightPool(lightPool)
    {
        optix::float3 csIntensity = this->m_intensity * this->m_color;

        /* Create SphereLight Struct for GPU program. */
        this->m_csSphereLight.lightType = CommonStructs::LightType::SphereLight;
        this->m_sphereShape->getMatrix(this->m_csSphereLight.lightToWorld, this->m_csSphereLight.worldToLight); 
        /* note that |lightToWorld| is directly decided by |m_sphereShape| */

        this->m_csSphereLight.invSurfaceArea = 1.f / this->m_sphereShape->getSurfaceArea();
        this->m_csSphereLight.center = this->m_sphereShape->getPosition();
        this->m_csSphereLight.radius = this->m_sphereShape->getRadius();
        this->m_csSphereLight.intensity = optix::make_float4(csIntensity.x, csIntensity.y, csIntensity.z, 1.0f);
    }

    float getLightIntensity() const
    {
        return this->m_intensity;
    }

    void setLightIntensity(float intensity);

    optix::float3 getLightColor() const
    {
        return this->m_color;
    }

    void setLightColor(const optix::float3 &color);

    optix::float3 getPosition() const
    {
        return this->m_sphereShape->getPosition();
    }

    void setPosition(const optix::float3 &position);

    float getRadius() const
    {
        return this->m_sphereShape->getRadius();
    }

    void setRadius(const float radius);

    const CommonStructs::SphereLight &getCommonStructsLight() const
    {
        return this->m_csSphereLight;
    }

    std::shared_ptr<Sphere> getSphereShape() const
    {
        return this->m_sphereShape;
    }

private:
    /**
     * @brief Update |m_csSphereLight.lightToWorld| and its inverse,
     * ,inverse of surface area, center and radius.
     */
    void updateMatrixParameter()
    {
        this->m_sphereShape->getMatrix(this->m_csSphereLight.lightToWorld, this->m_csSphereLight.worldToLight); /* note that |lightToWorld| is directly decided by |m_sphereShape| */
        this->m_csSphereLight.invSurfaceArea = 1.f / this->m_sphereShape->getSurfaceArea();
    }



private:
    LightPool *m_lightPool;

    CommonStructs::SphereLight m_csSphereLight;
    std::shared_ptr<Sphere>    m_sphereShape;

    /// Color (host only)
    optix::float3 m_color;
    /// Intensity (host only)
    float m_intensity;

    //static constexpr int lmax = 9;
    //static const float BSDFMatrix_Rawdata[(lmax + 1)*(lmax + 1)][(lmax + 1)*(lmax + 1)];
};