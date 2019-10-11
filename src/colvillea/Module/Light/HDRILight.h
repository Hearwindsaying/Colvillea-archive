#pragma once
#include "Light.h"

#include <map>

#include "../Image/ImageLoader.h"
#include "../../Device/Toolkit/CommonStructs.h"
#include "../../Device/Toolkit/Utility.h"
#include "../../Application/GlobalDefs.h"

class LightPool;

/**
 * @brief HDRILight describing an infinite area
 * light illuminated by High-Dynamic-Range Image
 * (HDRI). It's also known as Image Based Lighting.
 */
class HDRILight : public Light 
{
public:
    /**
     * @brief Factory method for creating a HDRILight instance.
     *
     * @param[in] application
     * @param[in] context
     * @param[in] programsMap  map to store Programs
     * @param[in] hdriFilename HDRI filename
     * @param[in] lightToWorld
     */
    static std::unique_ptr<HDRILight> createHDRILight(Application *application, optix::Context context, const std::map<std::string, optix::Program> &programsMap, const std::string & hdriFilename, const optix::Matrix4x4 &lightToWorld, /*std::shared_ptr<LightPool>*/LightPool * lightPool)
    {
        std::unique_ptr<HDRILight> hdriLight = std::make_unique<HDRILight>(application, context, programsMap, hdriFilename, lightPool);
        hdriLight->initializeLight(lightToWorld);
        return hdriLight;
    }

    HDRILight(Application *application, optix::Context context, const std::map<std::string, optix::Program> &programsMap, const std::string & hdriFilename, /*std::shared_ptr<LightPool>*/LightPool * lightPool);

    void initializeLight(const optix::Matrix4x4 &lightToWorld) override
    {
        /* Load HDRI texture. */
        auto context = this->m_context;
        this->m_HDRITextureSampler = ImageLoader::LoadImageTexture(context, this->m_HDRIFilename, optix::make_float4(0.f));

        /* Create HDRILight Struct for GPU program. */

        /* Check whether transform matrix has scale. */
        if (TwUtil::hasScale(lightToWorld))
            std::cerr << "[Warning] HDRILight has scale, which could lead to undefined behavior!" << std::endl;
        std::cout << "[Info] Scale component for HDRILight is: (" << TwUtil::getXScale(lightToWorld) << "," << TwUtil::getYScale(lightToWorld) << "," << TwUtil::getZScale(lightToWorld) << ")." << std::endl;

        this->m_csHDRILight.lightToWorld = lightToWorld;
        this->m_csHDRILight.worldToLight = lightToWorld.inverse();
        this->m_csHDRILight.hdriEnvmap   = this->m_HDRITextureSampler->getId();
        this->m_csHDRILight.lightType    = CommonStructs::LightType::HDRILight;

        /* HDRILight Struct setup can't be done until finish HDRILight::preprocess(). */
        /*context["hdriLight"]->setUserData(sizeof(CommonStructs::HDRILight), &this->m_csHDRILight);*/
    }

    const CommonStructs::HDRILight &getCommonStructsLight() const
    {
        return this->m_csHDRILight;
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
    /// We use a weak_ptr to hold LightPool instance, note that
    /// -- LightPool owns HDRILight while HDRILight needs to use
    /// LightPool to update LightBuffers.
    /*std::weak_ptr<LightPool>*/LightPool *m_lightPool;

    optix::TextureSampler m_HDRITextureSampler;
    std::string           m_HDRIFilename;

    CommonStructs::HDRILight m_csHDRILight;    // storage for struct data used in GPU programs
};