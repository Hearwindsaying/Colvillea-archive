#include "colvillea/Module/Light/HDRILight.h"

#include <chrono>
#include <functional>
#include <memory>


#include "colvillea/Application/GlobalDefs.h"
#include "colvillea/Application/Application.h"
#include "colvillea/Device/Toolkit/MCSampling.h"
#include "colvillea/Module/Light/LightPool.h"


using namespace optix;

HDRILight::HDRILight(Application *application, optix::Context context, const std::map<std::string, optix::Program> &programsMap, const std::string & hdriFilename, /*std::shared_ptr<LightPool>*/LightPool * lightPool, const optix::float3 &rotation) : 
    Light(context, programsMap, "HDRI", "HDRI Probe", IEditableObject::IEditableObjectType::HDRILight), 
    m_HDRIFilename(hdriFilename), m_lightPool(lightPool), m_enable(true), m_rotationRad(rotation)
{
    TW_ASSERT(application);
    TW_ASSERT(lightPool);
    TW_ASSERT(rotation.x >= 0.f && rotation.y >= 0.f && rotation.z >= 0.f);
    application->setPreprocessFunc(std::bind(&HDRILight::preprocess, this));
}

HDRILight::HDRILight(Application * application, optix::Context context, const std::map<std::string, optix::Program>& programsMap, LightPool * lightPool) : 
    Light(context, programsMap, "HDRI", "HDRI Probe", IEditableObject::IEditableObjectType::HDRILight),
    m_HDRIFilename(""), m_lightPool(lightPool), m_enable(false), m_rotationRad(optix::make_float3(0.f,0.f,0.f))
{
    TW_ASSERT(application);
    TW_ASSERT(lightPool);

    this->m_csHDRILight.lightToWorld = optix::Matrix4x4::identity();
    this->m_csHDRILight.worldToLight = optix::Matrix4x4::identity();
    this->m_csHDRILight.hdriEnvmap = RT_TEXTURE_ID_NULL;
    this->m_csHDRILight.lightType = CommonStructs::LightType::HDRILight;

    lightPool->updateCurrentHDRILight(this->m_csHDRILight);
}

void HDRILight::initializeLight()
{
    /* Load HDRI texture. */
    auto context = this->m_context;
    this->m_HDRITextureSampler = ImageLoader::LoadImageTexture(context, this->m_HDRIFilename, optix::make_float4(0.f));

    /* Create HDRILight Struct for GPU program. */
    this->m_csHDRILight.lightToWorld = Matrix4x4::rotate(this->m_rotationRad.x, make_float3(1.f,0.f,0.f)) *
                                       Matrix4x4::rotate(this->m_rotationRad.y, make_float3(0.f,1.f,0.f)) *
                                       Matrix4x4::rotate(this->m_rotationRad.z, make_float3(0.f,0.f,1.f));
    this->m_csHDRILight.worldToLight = this->m_csHDRILight.lightToWorld.inverse();
    this->m_csHDRILight.hdriEnvmap = this->m_HDRITextureSampler->getId();
    this->m_csHDRILight.lightType = CommonStructs::LightType::HDRILight;

    /* HDRILight Struct setup can't be done until finish HDRILight::preprocess(). */
    /*context["hdriLight"]->setUserData(sizeof(CommonStructs::HDRILight), &this->m_csHDRILight);*/
    this->m_lightPool->updateCurrentHDRILight(this->m_csHDRILight);
}

void HDRILight::setEnableHDRILight(bool enable)
{
    this->m_enable = enable;

    if (!this->m_HDRITextureSampler)
    {
        /* This is a dummy HDRILight.*/
        TW_ASSERT(this->m_csHDRILight.hdriEnvmap == RT_TEXTURE_ID_NULL);
        std::cout << "[Info] " << (enable ? "Enable" : "Disable") << " HDRI Light. This is a dummy HDRILight. Please try to load a HDRI to enable HDRILight." << std::endl;
        return;
    }

    /* Setup device variable to enable/disable HDRILight rendering. */
    this->m_csHDRILight.hdriEnvmap = enable ? this->m_HDRITextureSampler->getId() : RT_TEXTURE_ID_NULL;
    this->m_lightPool->updateCurrentHDRILight();

    std::cout << "[Info] " << (enable ? "Enable" : "Disable") << " HDRI Light." << std::endl;
}

void HDRILight::setLightRotation(const optix::float3 & rotation)
{
    TW_ASSERT(rotation.x >= 0.f && rotation.y >= 0.f && rotation.z >= 0.f);
    this->m_rotationRad = rotation;

    this->m_csHDRILight.lightToWorld = Matrix4x4::rotate(this->m_rotationRad.x, make_float3(1.f,0.f,0.f)) *
                                       Matrix4x4::rotate(this->m_rotationRad.y, make_float3(0.f,1.f,0.f)) *
                                       Matrix4x4::rotate(this->m_rotationRad.z, make_float3(0.f,0.f,1.f));
    this->m_csHDRILight.worldToLight = this->m_csHDRILight.lightToWorld.inverse();
    this->m_lightPool->updateCurrentHDRILight();

    std::cout << "[Info] " << "Updated HDRILight Transform successfully." << std::endl;
}



void HDRILight::preprocess()
{
    auto& context = this->m_context;

    /* Setup buffer for prefiltering launch. */
    RTsize HDRIWidth, HDRIHeight;
    this->m_HDRITextureSampler->getBuffer()->getSize(HDRIWidth, HDRIHeight);
    std::cout << "[Info] Getting HDRIWidth: " << HDRIWidth << " HDRIHeight:" << HDRIHeight << std::endl;

    
    optix::Buffer prefilteringLaunchBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, HDRIWidth, HDRIHeight);
    CommonStructs::HDRIEnvmapLuminanceBufferWrapper hdriEnvmapLuminanceBufferWrapper;
    hdriEnvmapLuminanceBufferWrapper.HDRIEnvmapLuminanceBuffer = prefilteringLaunchBuffer->getId();
    context["sysHDRIEnvmapLuminanceBufferWrapper"]->setUserData(sizeof(CommonStructs::HDRIEnvmapLuminanceBufferWrapper), &hdriEnvmapLuminanceBufferWrapper);

    /* Before launching, we need to update sysLightBuffers.hdriLight.hdriEnvmap
     * -- which will be later used by preprocessing OptiX launch. */
    this->m_lightPool->updateCurrentHDRILight();

    /* Launch prefiltering HDRI. */
    auto currentTime = std::chrono::system_clock::now();
    context->validate();
    context->launch(toUnderlyingValue(RayGenerationEntryType::HDRI), HDRIWidth, HDRIHeight);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - currentTime;
    std::cout << "[Info]Prefiltering launching time elapsed: " << (diff).count() << std::endl;

    #if 0
    {
        /* Test for writing result buffer to image. */
        CommonStructs::HDRIEnvmapLuminanceBufferWrapper outWrapper;
        context["sysHDRIEnvmapLuminanceBufferWrapper"]->getUserData(sizeof(CommonStructs::HDRIEnvmapLuminanceBufferWrapper), &outWrapper);
        TW_ASSERT(outWrapper.HDRIEnvmapLuminanceBuffer.getId() >= 0 && outWrapper.HDRIEnvmapLuminanceBuffer.getId() != RT_BUFFER_ID_NULL);
        
        ImageLoader::SaveLuminanceBufferToImage(
            context->getBufferFromId(outWrapper.HDRIEnvmapLuminanceBuffer.getId()), 
            "hdriEnvmapLuminance.exr");
    }
    #endif 

    /* Compute Distribution data for HDRILight*/ //todo:should be a part of HDRILight
    currentTime = std::chrono::system_clock::now();
    std::unique_ptr<TwUtil::MonteCarlo::Distribution2D> hdri2DDistribution;
    hdri2DDistribution.reset(new TwUtil::MonteCarlo::Distribution2D(static_cast<float *>(prefilteringLaunchBuffer->map()), HDRIWidth, HDRIHeight));

    end = std::chrono::system_clock::now();
    diff = end - currentTime;
    std::cout << "[Info]Precomputing HDRI distribution time elapsed: " << (diff).count() << std::endl;

    //////////////////////////////////////////////////////////////////////////
    /* Part1: Collect CommonStructs::Distribution1D data. */
    CommonStructs::Distribution1D pMarginal;
    pMarginal.funcIntegral = hdri2DDistribution->pMarginal->funcIntegral;

    auto pMarginal_funcSize = hdri2DDistribution->pMarginal->func.size();
    auto pMarginal_cdfSize = hdri2DDistribution->pMarginal->cdf.size();

    Buffer pMarginal_FuncBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, pMarginal_funcSize);
    Buffer pMarginal_cdfBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, pMarginal_cdfSize);

    float *pMarginal_FuncBuffer_Data = static_cast<float *>(pMarginal_FuncBuffer->map());
    float *pMarginal_cdfBuffer_Data = static_cast<float *>(pMarginal_cdfBuffer->map());

    memcpy(pMarginal_FuncBuffer_Data, hdri2DDistribution->pMarginal->func.data(), pMarginal_funcSize * sizeof(float));//todo:optimize away memcpy
    memcpy(pMarginal_cdfBuffer_Data, hdri2DDistribution->pMarginal->cdf.data(), pMarginal_cdfSize * sizeof(float));

    pMarginal_FuncBuffer->unmap();
    pMarginal_cdfBuffer->unmap();

    /* Assign BufferId for pMarginal: */
    pMarginal.func = pMarginal_FuncBuffer->getId();
    pMarginal.cdf = pMarginal_cdfBuffer->getId();

    //////////////////////////////////////////////////////////////////////////
    /* Part2: Collect pCondictionalV part for CommonStructs::Distribution2D data. */
    Buffer pCondV_funcBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, hdri2DDistribution->pCondictionalV[0]->func.size(), hdri2DDistribution->pCondictionalV.size());
    Buffer pCondV_cdfBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, hdri2DDistribution->pCondictionalV[0]->cdf.size(), hdri2DDistribution->pCondictionalV.size());
    Buffer pCondV_funcIntegralBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, hdri2DDistribution->pCondictionalV.size());

    float *pCondV_funcBufferData = static_cast<float *>(pCondV_funcBuffer->map());
    float *pCondV_cdfBufferData = static_cast<float *>(pCondV_cdfBuffer->map());
    void *pCondV_funcIntegralBufferData = pCondV_funcIntegralBuffer->map();

    for (auto row = 0; row < hdri2DDistribution->pCondictionalV.size(); ++row)
    {
        memcpy(pCondV_funcBufferData + row * hdri2DDistribution->pCondictionalV[row]->func.size(), hdri2DDistribution->pCondictionalV[row]->func.data(), hdri2DDistribution->pCondictionalV[row]->func.size() * sizeof(float));
        memcpy(pCondV_cdfBufferData + row * hdri2DDistribution->pCondictionalV[row]->cdf.size(), hdri2DDistribution->pCondictionalV[row]->cdf.data(), hdri2DDistribution->pCondictionalV[row]->cdf.size() * sizeof(float));
    }
    memcpy(pCondV_funcIntegralBufferData, hdri2DDistribution->pMarginal->func.data(), hdri2DDistribution->pMarginal->func.size() * sizeof(float));

    pCondV_funcBuffer->unmap();
    pCondV_cdfBuffer->unmap();
    pCondV_funcIntegralBuffer->unmap();

    /* Finally, setup CommonStructs::Distribution2D. */
    CommonStructs::Distribution2D hdriLightDistribution;
    hdriLightDistribution.pMarginal = pMarginal;
    hdriLightDistribution.pConditionalV_funcIntegral = pCondV_funcIntegralBuffer->getId();
    hdriLightDistribution.pConditionalV_func = pCondV_funcBuffer->getId();
    hdriLightDistribution.pConditionalV_cdf = pCondV_cdfBuffer->getId();

    this->m_csHDRILight.distributionHDRI = hdriLightDistribution;

    /* Setup HDRILight Struct is done by LightPool. */
//     auto lightPool = this->m_lightPool.lock();
//     TW_ASSERT(lightPool);
    this->m_lightPool->updateCurrentHDRILight();

    /* Do not forget unmapping prefiltering buffer we mapped before! */
    prefilteringLaunchBuffer->unmap();
}
