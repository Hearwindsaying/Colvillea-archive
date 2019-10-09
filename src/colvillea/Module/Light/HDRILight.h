#pragma once
#include "Light.h"

#include "../Image/ImageLoader.h"
#include "../../Device/Toolkit/CommonStructs.h"
#include "../../Device/Toolkit/Utility.h"
#include "../../Application/GlobalDefs.h"

#include <map>
/**
 * @brief HDRILight describing an infinite area
 * light illuminated by High-Dynamic-Range Image
 * (HDRI). It's also known as Image Based Lighting.
 * 
 * @note The creation for HDRILight should be performed
 * finally. See HDRILight::preprocess() for details.
 * 
 * @see HDRILight::preprocess()
 */
class HDRILight : public Light 
{
public:
    HDRILight(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const std::string & hdriFilename, const optix::Matrix4x4 &lightToWorld) :Light(context, programsMap, "HDRI"/*fix name*/), m_HDRIFilename(hdriFilename)
    {
        auto programItr = this->m_programsMap.find("RayGeneration_PrefilterHDRILight");
        TW_ASSERT(programItr != this->m_programsMap.end());
        this->m_context->setRayGenerationProgram(toUnderlyingValue(RayGenerationEntryType::HDRI), programItr->second);
       
        /* Check whether transform matrix has scale. */
        if (TwUtil::hasScale(lightToWorld))
            std::cerr << "[Warning] HDRILight has scale, which could lead to undefined behavior!" << std::endl;
        std::cout << "[Info] Scale component for HDRILight is: (" << TwUtil::getXScale(lightToWorld) << "," << TwUtil::getYScale(lightToWorld) << "," << TwUtil::getZScale(lightToWorld) << ")." << std::endl;

        this->m_csHDRILight.lightToWorld = lightToWorld;
        this->m_csHDRILight.worldToLight = lightToWorld.inverse();
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
    static void initHDRILight(optix::Context context, const std::map<std::string, optix::Program> &programsMap)
    {
        //todo:move from HDRILight ctor to here:
        if (!context->getRayGenerationProgram(toUnderlyingValue(RayGenerationEntryType::HDRI)))
        {
            auto programItr = programsMap.find("RayGeneration_PrefilterHDRILight");
            TW_ASSERT(programItr != programsMap.end());
            context->setRayGenerationProgram(toUnderlyingValue(RayGenerationEntryType::HDRI), programItr->second);
        }

        

        optix::Buffer prefilteringLaunchBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, 0, 0);
        context["hdriEnvmapLuminance"]->setBuffer(prefilteringLaunchBuffer);

        CommonStructs::HDRILight hdriLight;
        context["hdriLight"]->setUserData(sizeof(CommonStructs::HDRILight), &hdriLight);

        CommonStructs::Distribution2D dist2D;
        context["hdriLightDistribution"]->setUserData(sizeof(CommonStructs::Distribution2D), &dist2D);

        context["hdriEnvmap"]->setInt(RT_TEXTURE_ID_NULL);
    }

    //todo:bad design, light related parameters should be as loadLight?
    void loadLight() override
    {
        /* Load HDRI texture and set |hdriEnvmap|, which is shared by spherical skybox (Miss program) and HDRILight program. */
        auto context = this->m_context;
        this->m_HDRITextureSampler = ImageLoader::LoadImageTexture(context, this->m_HDRIFilename, optix::make_float4(0.f));
        this->updateHDRIEnvmap();

        /* Create HDRILight Struct for GPU program. */
        /*this->m_csHDRILight.lightToWorld = lightToWorld;
        this->m_csHDRILight.worldToLight = lightToWorld.inverse();*/
        this->m_csHDRILight.hdriEnvmap   = this->m_HDRITextureSampler->getId();
        this->m_csHDRILight.lightType    = CommonStructs::LightType::HDRILight;
        this->m_csHDRILight.nSamples     = 1; //todo:add support for nSamples

        /* Setup HDRILight Struct. */
        context["hdriLight"]->setUserData(sizeof(CommonStructs::HDRILight), &this->m_csHDRILight);

        this->preprocess();
    }

private:

    /**
     * @brief Setup hdriEnvmap using |m_HDRITextureSampler|
     * This is a part of HDRILight::loadLight() for updating 
     * spherical environment map.
     */
    void updateHDRIEnvmap()
    {
        auto context = this->m_context;
        context["hdriEnvmap"]->setInt(this->m_HDRITextureSampler->getId());
    }

protected:
    /**
     * @brief Preprocessing for HDRILight. Besides fundamental
     * properties describing HDRILight, some precomputation
     * needed to be done before launching to render. Meanwhile,
     * prefiltering HDRI texture for precomputation accelerated
     * by OptiX using another launch is necessary before that.
     * 
     * @note This function will invoke context->launch() which
     * needs an implicit context->validate(). In other words,
     * it is able to continue launch() after all other status
     * for OptiX are set. Consequently, HDRILight's creation
     * should be in the last.
     */
    void preprocess() override;

private:
    optix::TextureSampler m_HDRITextureSampler;
    std::string           m_HDRIFilename;

    CommonStructs::HDRILight m_csHDRILight;    // storage for struct data used in GPU programs
};