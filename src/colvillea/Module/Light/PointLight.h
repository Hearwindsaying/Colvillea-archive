#pragma once
#include "Light.h"

#include "colvillea/Device/Toolkit/CommonStructs.h"
#include "colvillea/Device/Toolkit/Utility.h"

class LightPool;

/**
 * @brief PointLight describes a infinitesimal
 * point uniformly emitting light in all directions.
 * Its distribution is typically depicted as 
 * dirac delta distribution.
 * 
 * @note This class contains description of
 * what a single PointLight should be. Buffers
 * storing pointLight is delegated to SceneGraph.
 */
class PointLight : public Light
{
public:
    /**
     * @brief Factory method for creating a PointLight instance.
     *
     * @param[in] application
     * @param[in] context
     * @param[in] programsMap  map to store Programs
     * @param[in] color        light color
     * @param[in] intensity    light intensity
     * @param[in] lightPosition
     */
    static std::unique_ptr<PointLight> createPointLight(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3& color, float intensity, const optix::float3 &lightPosition, LightPool *lightPool)
    {
        std::unique_ptr<PointLight> pointLight = std::make_unique<PointLight>(context, programsMap, color, intensity, lightPosition, lightPool);
        return pointLight;
    }

    PointLight(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3& color, float intensity, const optix::float3 &lightPosition, LightPool *lightPool) :
        Light(context, programsMap, "Point", "Point Light", IEditableObject::IEditableObjectType::PointLight),
        m_intensity(intensity), m_color(color), m_lightPool(lightPool)
    {
        optix::float3 csIntensity = this->m_intensity * this->m_color;

        /* Setup CommonStructs::PointLight Struct for GPU program. */
        this->m_csPointLight.lightType = CommonStructs::LightType::PointLight;
        this->m_csPointLight.intensity = optix::make_float4(csIntensity.x, csIntensity.y, csIntensity.z, 1.0f);
        this->m_csPointLight.lightPos = lightPosition;
    }

    optix::float3 getLightPosition() const
    {
        return this->m_csPointLight.lightPos;
    }

    void setLightPosition(const optix::float3 &lightPosition);

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


    const CommonStructs::PointLight &getCommonStructsLight() const
    {
        return this->m_csPointLight;
    }

private:
    LightPool *m_lightPool;

    CommonStructs::PointLight m_csPointLight;

    /// Color (host only)
    optix::float3 m_color;
    /// Intensity (host only)
    float m_intensity;
};