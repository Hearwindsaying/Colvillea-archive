#pragma once
#include "Light.h"

#include "../../Device/Toolkit/CommonStructs.h"
#include "../../Device/Toolkit/Utility.h"

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
     * @param[in] intensity    light intensity
     * @param[in] lightToWorld
     */
    static std::unique_ptr<PointLight> createPointLight(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3 intensity, const optix::Matrix4x4 &lightToWorld)
    {
        std::unique_ptr<PointLight> pointLight = std::make_unique<PointLight>(context, programsMap, intensity);
        pointLight->initializeLight(lightToWorld);
        return pointLight;
    }

    PointLight(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3 intensity) : Light(context, programsMap, "Point")
    {
        this->m_csPointLight.intensity = optix::make_float4(intensity.x, intensity.y, intensity.z, 1.f);
    }


    void initializeLight(const optix::Matrix4x4 &lightToWorld) override
    { 
        this->m_lightToWorld = lightToWorld;

        /* Create PointLight Struct for GPU program. */
        this->m_csPointLight.lightType = CommonStructs::LightType::PointLight;
        this->m_csPointLight.lightPos = TwUtil::xfmPoint(optix::make_float3(0.f, 0.f, 0.f), lightToWorld);
    }



    const CommonStructs::PointLight &getCommonStructsLight() const
    {
        return this->m_csPointLight;
    }

private:
    CommonStructs::PointLight m_csPointLight;

    /// lightToWorld doesn't exist in CommonStructs::PointLight
    optix::Matrix4x4 m_lightToWorld;
};