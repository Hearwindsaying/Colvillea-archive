#pragma once

#include <map>
#include <vector>

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "colvillea/Module/Light/PointLight.h"
#include "colvillea/Module/Light/HDRILight.h"
#include "colvillea/Module/Light/QuadLight.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"
#include "colvillea/Application/TWAssert.h"
#include "colvillea/Application/SceneGraph.h"

class Application;


/**
 * @brief A simple pool used for creating, containing and
 * managing lights. This is analogous to MaterialPool.
 * todo: add a pool interface?
 * 
 */
class LightPool
{
public:
    LightPool(Application *application, const std::map<std::string, optix::Program> &programsMap, const optix::Context context, std::shared_ptr<SceneGraph> sceneGraph)
        : m_programsMap(programsMap), m_context(context), m_application(application), m_sceneGraph(sceneGraph)
    {
        /* Initial values for |sysLightBuffers|. */
        this->m_csLightBuffers.hdriLight.hdriEnvmap = RT_TEXTURE_ID_NULL;
        this->m_csLightBuffers.pointLightBuffer     = RT_BUFFER_ID_NULL;
        this->m_csLightBuffers.quadLightBuffer      = RT_BUFFER_ID_NULL;

        this->updateLightBuffers();
    }

    /**
     * @brief Create HDRILight object and add to SceneGraph.
     * In current implmentation, only one instance to HDRILight
     * is hold which means changing HDRI is permitted but it
     * will destroy previous existing HDRILight. Recomputation
     * for initialization of HDRILight needs to be done again.
     *
     * @param HDRIFilename filename including path to HDRI.
     */
    void createHDRILight(const std::string & HDRIFilename, const optix::Matrix4x4 &lightToWorld)
    {
        this->m_HDRILight = HDRILight::createHDRILight(this->m_application, this->m_context, this->m_programsMap, HDRIFilename, lightToWorld, this);

        /* Note that after HDRILight::preprocess() we need to update LightBuffers again. 
         * -- We can't fetch a complete CommonStructs::HDRILight now. */
        this->m_csLightBuffers.hdriLight = this->m_HDRILight->getCommonStructsLight();
        this->updateLightBuffers();
    }

    /**
     * @brief Create a pointLight object and add to SceneGraph.
     *
     * @param lightToWorld matrix representing position, only
     * translation component is needed for an ideal point light.
     * @param intensity representing both color and intensity,
     * whose components could be larger than 1.f.
     *
     * @note Only adding light is supported. //todo:use LightPool
     */
    void createPointLight(const optix::Matrix4x4 &lightToWorld, const optix::float3 &intensity)
    {
        std::shared_ptr<PointLight> pointLight = PointLight::createPointLight(this->m_context, this->m_programsMap, intensity, lightToWorld);

        this->m_pointLights.push_back(pointLight);


        /* Setup pointLightBuffer for GPU Program */
        auto pointLightBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, this->m_pointLights.size());
        pointLightBuffer->setElementSize(sizeof(CommonStructs::PointLight));
        auto pointLightArray = static_cast<CommonStructs::PointLight *>(pointLightBuffer->map());
        for (auto itr = this->m_pointLights.cbegin(); itr != this->m_pointLights.cend(); ++itr)
        {
            pointLightArray[itr - this->m_pointLights.cbegin()] = (*itr)->getCommonStructsLight();
        }

        this->m_csLightBuffers.pointLightBuffer = pointLightBuffer->getId();
        this->updateLightBuffers();

        pointLightBuffer->unmap();
    }

    /**
     * @brief Create a quadLight object and add to SceneGraph.
     *
     * @param[in] lightToWorld  matrix representing position, only
     * translation component is needed for an ideal point light.
     * @param[in] intensity     representing both color and intensity,
     * whose components could be larger than 1.f.
     * @param[in] materialIndex index to materialBuffer, call
     * MaterialPool::createEmissiveMaterial() to create the material
     * for quadlight.
     * @param[in] flipNormal    flip light's geometry if necessary
     *
     * @note Only adding light is supported. //todo:use LightPool
     * @see MaterialPool::createEmissiveMaterial()
     */
    void createQuadLight(const optix::Matrix4x4 &lightToWorld, const optix::float3 &intensity, const int materialIndex, bool flipNormal = false)
    {
        std::shared_ptr<Quad> lightQuadShape = this->m_sceneGraph->createQuad(materialIndex, lightToWorld, this->m_quadLights.size() /* size() is the index we want for the current creating quad */, flipNormal);
        std::shared_ptr<QuadLight> quadLight = QuadLight::createQuadLight(this->m_context, this->m_programsMap, intensity, lightQuadShape);

        this->m_quadLights.push_back(quadLight);

        /* Setup quadLightBuffer for GPU Program */
        auto quadLightBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, this->m_quadLights.size());
        quadLightBuffer->setElementSize(sizeof(CommonStructs::QuadLight));
        auto quadLightArray = static_cast<CommonStructs::QuadLight *>(quadLightBuffer->map());
        for (auto itr = this->m_quadLights.cbegin(); itr != this->m_quadLights.cend(); ++itr)
        {
            quadLightArray[itr - this->m_quadLights.cbegin()] = (*itr)->getCommonStructsLight();
        }

        this->m_csLightBuffers.quadLightBuffer = quadLightBuffer->getId();
        updateLightBuffers();

        quadLightBuffer->unmap();
    }

public:
    /**
     * @brief Set CommonStructs::HDRILight for |m_csLightBuffers|.
     * 
     * @todo HDRILight class should have a friend of LightPool and
     * set this method to private.
     */
    void setCSHDRILight(const CommonStructs::HDRILight &hdriLight)
    {
        this->m_csLightBuffers.hdriLight = hdriLight;
        updateLightBuffers();
    }

private:
    void updateLightBuffers()
    {
        this->m_context["sysLightBuffers"]->setUserData(sizeof(CommonStructs::LightBuffers), &this->m_csLightBuffers);
    }

private:
    Application *m_application;
    const std::map<std::string, optix::Program> &m_programsMap;
    optix::Context     m_context;

    std::shared_ptr<SceneGraph> m_sceneGraph;

    std::shared_ptr<HDRILight>                  m_HDRILight; 
    std::vector<std::shared_ptr<PointLight>>    m_pointLights;
    std::vector<std::shared_ptr<QuadLight>>     m_quadLights;

    CommonStructs::LightBuffers m_csLightBuffers;
};