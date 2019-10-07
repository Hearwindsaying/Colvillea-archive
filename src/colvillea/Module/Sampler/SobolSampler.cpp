#include "SobolSampler.h"

#include "SobolMatrices.h"
#include "../../Device/Toolkit/CommonStructs.h"
#include "../../Application/TWAssert.h"

#include <algorithm>

void SobolSampler::initSampler()
{
    auto& context = this->m_context;

    auto resolution = this->roundUpPow2(std::max(this->m_filmResolution.x, this->m_filmResolution.y));
    auto log2Resolution = static_cast<int>(log2(resolution));
    if (resolution > 0)
        TW_ASSERT(1 << log2Resolution == resolution);
    std::cout << "[Info] resolution: " << resolution << " log2resolution: " << log2Resolution << std::endl;

    /* Read Sobol Matrices. */
    std::cout << "[Info] Start loading SobolMatrices into device" << std::endl;

    constexpr int SobolMatrices32Size = SobolMatrix::NumSobolDimensions * SobolMatrix::SobolMatrixSize;
    auto SobolMatrices32Buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, SobolMatrices32Size);
    memcpy(SobolMatrices32Buffer->map(), &SobolMatrix::SobolMatrices32[0], SobolMatrices32Size * sizeof(uint32_t));
    SobolMatrices32Buffer->unmap();

    constexpr int VdCSobolMatricesWidthSize = SobolMatrix::SobolMatrixSize;
    constexpr int VdcSobolMatricesHeightSize = 25;
    auto VdCSobolMatricesBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, VdCSobolMatricesWidthSize, VdcSobolMatricesHeightSize);
    VdCSobolMatricesBuffer->setElementSize(sizeof(uint64_t));
    memcpy(VdCSobolMatricesBuffer->map(), &SobolMatrix::VdCSobolMatrices[0], VdCSobolMatricesWidthSize * VdcSobolMatricesHeightSize * sizeof(uint64_t));
    VdCSobolMatricesBuffer->unmap();

    constexpr int VdCSobolMatricesInvWidthSize = SobolMatrix::SobolMatrixSize;
    constexpr int VdcSobolMatricesInvHeightSize = 26;
    auto VdCSobolMatricesInvBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, VdCSobolMatricesInvWidthSize, VdcSobolMatricesInvHeightSize);
    VdCSobolMatricesInvBuffer->setElementSize(sizeof(uint64_t));
    memcpy(VdCSobolMatricesInvBuffer->map(), &SobolMatrix::VdCSobolMatricesInv[0], VdCSobolMatricesInvWidthSize * VdcSobolMatricesInvHeightSize * sizeof(uint64_t));
    VdCSobolMatricesInvBuffer->unmap();

    /* Load into |globalSobolSampler|. */
    settingSobolSamplerParameters(resolution, log2Resolution, SobolMatrices32Buffer->getId(), VdCSobolMatricesBuffer->getId(), VdCSobolMatricesInvBuffer->getId());
    context["globalSobolSampler"]->setUserData(sizeof(CommonStructs::GlobalSobolSampler), &this->m_globalSobolSampler);

    std::cout << "[Info] Finish loading SobolMatrices into device" << std::endl;
}
