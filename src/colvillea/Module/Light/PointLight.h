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
    PointLight(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3 intensity) : Light(context, programsMap, "Point")
    {
        this->m_csPointLight.intensity = optix::make_float4(intensity.x, intensity.y, intensity.z, 1.f);
    }


    void loadLight(const optix::Matrix4x4 &lightToWorld) override
    {/* todo:redundant ops. */
        /* Create PointLight Struct for GPU program. */
        this->m_csPointLight.lightType = CommonStructs::LightType::PointLight;
        this->m_csPointLight.lightPos = TwUtil::xfmPoint(optix::make_float3(0.f, 0.f, 0.f), lightToWorld);
    }

    /**
     * @brief Initialize light for context, including
     * setup light buffers and variables related to
     * context. This function should be invoked internally
     * in SceneGraph::initScene()
     *
     * @note Note that different from SceneGraph::loadLight()
     * which is responsible for loading one individual light
     * while this is a context-wide job and independent of
     * one light.
     *
     * @see SceneGraph::initScene()
     *
     */
    static void initPointLight(optix::Context context) 
    {
        constexpr int initPointLightNum = 0;
        auto pointLightBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, initPointLightNum);
        pointLightBuffer->setElementSize(sizeof(CommonStructs::PointLight));
        context["pointLightBuffer"]->setBuffer(pointLightBuffer);
    }

    const CommonStructs::PointLight &getCommonStructsLight() const
    {//todo:review
        return this->m_csPointLight;
    }

private:
    CommonStructs::PointLight m_csPointLight;
};