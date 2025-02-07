#pragma once

#define  CL_CHECK_MEMORY_LEAKS
#ifdef CL_CHECK_MEMORY_LEAKS
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#define CL_CHECK_MEMORY_LEAKS_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new CL_CHECK_MEMORY_LEAKS_NEW
#endif

#include <algorithm>
#include <map>
#include <vector>
#include <memory>

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
 * @todo: add a pool interface.
 * 
 */
class LightPool
{
public:
    static std::shared_ptr<LightPool> createLightPool(Application *application, const std::map<std::string, optix::Program> &programsMap, const optix::Context context, std::shared_ptr<SceneGraph> sceneGraph)
    {
        std::shared_ptr<LightPool> lightPool = std::make_shared<LightPool>(application, programsMap, context, sceneGraph);
        lightPool->m_application->m_lightPool = lightPool;
        return lightPool;
    }

    LightPool(Application *application, const std::map<std::string, optix::Program> &programsMap, const optix::Context context, std::shared_ptr<SceneGraph> sceneGraph)
        : m_programsMap(programsMap), m_context(context), m_application(application), m_sceneGraph(sceneGraph)
    {
        /* Initial values for |sysLightBuffers|. */
        this->initializeLoadingDefaultHDRILight();
        this->m_csLightBuffers.pointLightBuffer     = RT_BUFFER_ID_NULL;
        this->m_csLightBuffers.quadLightBuffer      = RT_BUFFER_ID_NULL;

        this->updateLightBuffers();
    }

private:
    void initializeLoadingDefaultHDRILight()
    {
        /* Create a dummy HDRILight. */
        this->m_HDRILight = HDRILight::createHDRILight(this->m_application, this->m_context, this->m_programsMap, this);
    }

public:

    /**
     * @brief Create HDRILight object and add to SceneGraph.
     * In current implmentation, only one instance to HDRILight
     * is hold which means changing HDRI is permitted but it
     * will destroy previous existing HDRILight. Recomputation
     * for initialization of HDRILight needs to be done again.
     * 
     * 
     * @note Note that this is for hard code creating HDRILight
     * usage. For default behavior (no HDRILight is specified in
     * hard code), a dummy disabled HDRILight will be created.
     *
     * @param[in] HDRIFilename filename including path to HDRI.
     */
    void createHDRILight(const std::string & HDRIFilename, const optix::float3 &rotationAngles)
    {
        TW_ASSERT(this->m_HDRILight);

        /* Replace m_HDRILight directly. */
        this->m_HDRILight = HDRILight::createHDRILight(this->m_application, this->m_context, this->m_programsMap, HDRIFilename, rotationAngles, this);

        /* Note that after HDRILight::preprocess() we need to update LightBuffers again.
         * -- We can't fetch a complete CommonStructs::HDRILight now. */
    }

    /**
     * @brief Create a pointLight object and add to SceneGraph.
     *
     * @param lightToWorld matrix representing position, only
     * translation component is needed for an ideal point light.
     * @param intensity representing both color and intensity,
     * whose components could be larger than 1.f.
     */
    void createPointLight(const optix::float3 &lightPosition, const optix::float3 &color, float intensity)
    {
        std::shared_ptr<PointLight> pointLight = PointLight::createPointLight(this->m_context, this->m_programsMap, color, intensity, lightPosition, this);
        this->m_pointLights.push_back(pointLight);

        this->updateAllPointLights();
    }


    /**
    * @brief Remove a PointLight.
    */
    void removePointLight(const std::shared_ptr<PointLight> &pointLight)
    {
        auto itrToErase = std::find_if(this->m_pointLights.begin(), this->m_pointLights.end(),
            [&pointLight](const auto& curPointLight)
        {
            return curPointLight->getId() == pointLight->getId();
        });

        TW_ASSERT(itrToErase != this->m_pointLights.end());

        this->m_pointLights.erase(itrToErase);

        /* Update PointLights. */
        this->updateAllPointLights();
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
     * @note Only adding light is supported.
     * @see MaterialPool::createEmissiveMaterial()
     */
    void createQuadLight(const optix::float3 &position, const optix::float3 &rotation, const optix::float3 &scale, const optix::float3& color, float intensity, int32_t materialIndex, const std::shared_ptr<BSDF> &bsdf, bool flipNormal = false)
    {
        std::shared_ptr<Quad> lightQuadShape = this->m_sceneGraph->createQuad(materialIndex, 
            position, rotation, scale, 
            static_cast<int32_t>(this->m_quadLights.size()) /* size() is the index we want for the current creating quad */, 
            bsdf, 
            flipNormal);
        std::shared_ptr<QuadLight> quadLight = QuadLight::createQuadLight(this->m_context, this->m_programsMap, color, intensity, lightQuadShape, this);
        this->m_quadLights.push_back(quadLight);

        this->updateAllQuadLights(true);
    }

    /**
     * @brief Remove a QuadLight.
     * @note The underlying Quad shape is removed as well.
     */
    void removeQuadLight(const std::shared_ptr<QuadLight> &quadLight)
    {
        auto itrToErase = std::find_if(this->m_quadLights.begin(), this->m_quadLights.end(),
            [&quadLight](const auto& curQuadLight)
        {
            return curQuadLight->getId() == quadLight->getId();
        });

        TW_ASSERT(itrToErase != this->m_quadLights.end());

        /* 1.Remove Quad shape. */
        std::shared_ptr<Quad> quadShape = (*itrToErase)->getQuadShape();
        quadShape->getGeometry()->destroy();
        this->m_sceneGraph->removeGeometry(quadShape);
        quadShape->getGeometryInstance()->destroy();
        

        /* 2.Remove QuadLight. */
        this->m_quadLights.erase(itrToErase);

        /* 3.Update QuadLights. */
        this->updateAllQuadLights(true);
    }


    /************************************************************************/
    /*                         Getters & Setters                            */
    /************************************************************************/
public:
    /**
     * @brief Getter for |m_HDRILight|.
     */
    std::shared_ptr<HDRILight> getHDRILight() const
    {
        return this->m_HDRILight;
    }

    /**
     * @brief Getter for |m_pointLights|.
     */
    const std::vector<std::shared_ptr<PointLight>> &getPointLights() const
    {
        return this->m_pointLights;
    }

    /**
     * @brief Getter for |m_quadLights|.
     */
    const std::vector<std::shared_ptr<QuadLight>> &getQuadLights() const
    {
        return this->m_quadLights;
    }


private:
    /************************************************************************/
    /*                             Update functions                         */
    /************************************************************************/

    /**
     * @brief Update CommonStructs::HDRILight for |m_csLightBuffers|.
     * 
     * @note There is only one instance available all the time but we
     * still name it as "Current" in consistence with SceneGraph::updateCurrentFilter().
     */
    void updateCurrentHDRILight()
    {
        TW_ASSERT(this->m_HDRILight);
        this->m_csLightBuffers.hdriLight = this->m_HDRILight->getCommonStructsLight();
        this->updateLightBuffers();
    }

    /**
     * @brief Update CommonStructs::HDRILight for |m_csLightBuffers| in HDRILight
     * constructor.
     * 
     * @note When this->m_HDRILight is empty (creating default HDRILight),
     * an extra CommonStructs::HDRILight is needed for updating |m_csLightBuffers.hdriLight|.
     */
    void updateCurrentHDRILight(const CommonStructs::HDRILight &hdriLight)
    {
        /*TW_ASSERT(!this->m_HDRILight); // If we have a hard code creating HDRILight, this->m_HDRILight is not empty. */
        this->m_csLightBuffers.hdriLight = hdriLight;
        this->updateLightBuffers();
    }

    /**
     * @brief Update all PointLights. This is applicable for all modification operations to
     * PointLight, adding, modifying and removing.
     * @todo Rewrite createPointLight() and this function to support update one
     * single PointLight a time.
     *       -- add "bool resizeBuffer" to avoid unnecessary resizing.
     */
    void updateAllPointLights()
    {
        optix::Buffer pointLightBuffer;

        /* PointLight Buffer has yet been set up. */
        if (this->m_csLightBuffers.pointLightBuffer.getId() == RT_BUFFER_ID_NULL)
        {
            pointLightBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, this->m_pointLights.size());
            pointLightBuffer->setElementSize(sizeof(CommonStructs::PointLight));

            this->m_csLightBuffers.pointLightBuffer = pointLightBuffer->getId();

            std::cout << "[Info] Created PointLight Buffer." << std::endl;
        }
        else
        {
            pointLightBuffer = this->m_context->getBufferFromId(this->m_csLightBuffers.pointLightBuffer.getId());
            TW_ASSERT(pointLightBuffer);
            pointLightBuffer->setSize(this->m_pointLights.size());

            std::cout << "[Info] Updated PointLight Buffer." << std::endl;
        }
            
        /* Setup pointLightBuffer for GPU Program */
        auto pointLightArray = static_cast<CommonStructs::PointLight *>(pointLightBuffer->map());
        for (auto itr = this->m_pointLights.cbegin(); itr != this->m_pointLights.cend(); ++itr)
        {
            pointLightArray[itr - this->m_pointLights.cbegin()] = (*itr)->getCommonStructsLight();
        }

        this->updateLightBuffers();

        /* Unmap buffer. */
        pointLightBuffer->unmap();
    }

    /**
     * @brief Update all QuadLights. This is applicable for all modification operations to
     * QuadLight, adding, modifying and removing.
     * 
     * @todo Rewrite createQuadLight() and this function to support update one
     * single QuadLight a time.
     *       -- add "bool resizeBuffer" to avoid unnecessary resizing.
     *       
     * @param[in] rebuildAccel acceleration structure should be rebuilt? This is required
     * when adding, modifying light transform and removing light.
     */
    void updateAllQuadLights(bool rebuildAccel)
    {
        optix::Buffer quadLightBuffer;
        /* QuadLight Buffer has yet to be set up. */
        if (this->m_csLightBuffers.quadLightBuffer.getId() == RT_BUFFER_ID_NULL)
        {
            TW_ASSERT(rebuildAccel);

            quadLightBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, this->m_quadLights.size());
            quadLightBuffer->setElementSize(sizeof(CommonStructs::QuadLight));

            this->m_csLightBuffers.quadLightBuffer = quadLightBuffer->getId();

            std::cout << "[Info] Created QuadLight Buffer." << std::endl;
        }
        else
        {
            quadLightBuffer = this->m_context->getBufferFromId(this->m_csLightBuffers.quadLightBuffer.getId());
            TW_ASSERT(quadLightBuffer);
            quadLightBuffer->setSize(this->m_quadLights.size());

            std::cout << "[Info] Updated QuadLight Buffer." << std::endl;
        }

        /* Setup quadLightBuffer for GPU Program */
        auto quadLightArray = static_cast<CommonStructs::QuadLight *>(quadLightBuffer->map());
        for (int32_t quadLightIndex = 0; quadLightIndex < this->m_quadLights.size(); ++quadLightIndex)
        {
            this->m_quadLights[quadLightIndex]->getQuadShape()->setQuadLightIndex(quadLightIndex);
            quadLightArray[quadLightIndex] = this->m_quadLights[quadLightIndex]->getCommonStructsLight();
        }

        this->updateLightBuffers();

        if (rebuildAccel)
        {
            this->m_sceneGraph->rebuildGeometry();
        }

        /* Unmap buffer. */
        quadLightBuffer->unmap();
    }


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

    friend class HDRILight;
    friend class PointLight;
    friend class QuadLight;
};