#include "HDRILight.h"
#include "../../Application/GlobalDefs.h"
#include "../../Device/Toolkit/MCSampling.h"

#include <chrono>
#include <memory>

using namespace optix;

void HDRILight::preprocess()
{
    auto& context = this->m_context;

    /* Setup buffer for prefiltering launch. */
    RTsize HDRIWidth, HDRIHeight;
    this->m_HDRITextureSampler->getBuffer()->getSize(HDRIWidth, HDRIHeight);
    std::cout << "[Info] Getting HDRIWidth: " << HDRIWidth << " HDRIHeight:" << HDRIHeight << std::endl;

    optix::Buffer prefilteringLaunchBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, HDRIWidth, HDRIWidth);
    context["hdriEnvmapLuminance"]->setBuffer(prefilteringLaunchBuffer);

    /* Launch prefiltering HDRI. */
    auto currentTime = std::chrono::system_clock::now();
    context->validate();
    context->launch(toUnderlyingValue(RayGenerationEntryType::HDRI), HDRIWidth, HDRIHeight);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - currentTime;
    std::cout << "[Info]Prefiltering launching time elapsed: " << (diff).count() << std::endl;

    /* Test:write result buffer to image
      ImageLoader::SaveLuminanceBufferToImage(this->context["hdriEnvmapLuminance"]->getBuffer(), "hdriEnvmapLuminance.exr");*/

    /* Compute Distribution data for HDRILight*/ //todo:should be a part of HDRILight
    currentTime = std::chrono::system_clock::now();
    std::unique_ptr<TwUtil::MonteCarlo::Distribution2D> hdri2DDistribution;
    hdri2DDistribution.reset(new TwUtil::MonteCarlo::Distribution2D(static_cast<float *>(prefilteringLaunchBuffer->map()), HDRIWidth, HDRIHeight));

    std::chrono::system_clock::now();
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

    /* Assign BufferId for pMarginal:*/
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

    context["hdriLightDistribution"]->setUserData(sizeof(CommonStructs::Distribution2D), &hdriLightDistribution);

    /* Do not forget unmapping prefiltering buffer we mapped before! */
    prefilteringLaunchBuffer->unmap();

    /*todo:destroy the hdriEnvmapLuminanceBuffer
          -- no longer needed after precomputation for hdriLightDistribution.
          However, for changeable HDRILight, it should stay*/
          //this->context->removeVariable(this->context->queryVariable("hdriEnvmapLuminance"));
          //hdriEnvmapLuminanceBuffer->destroy();
}
