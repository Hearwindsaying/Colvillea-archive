// Include Eigen before STL and after OptiX (for macro support like __host__)
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix.h>

#include "colvillea/Device/SH/Test/SH.hpp"

// STL include
#include <random>

#include "colvillea/Module/Light/QuadLight.h"

#include "colvillea/Application/Application.h" // For LightPool::createLightPool()
#include "colvillea/Module/Light/LightPool.h"

void QuadLight::setPosition(const optix::float3 &position)
{
    /* Update underlying Quad shape. */
    this->m_quadShape->setPosition(position);

    /* Update QuadLight Struct for GPU program. */
    this->updateMatrixParameter();

    //update all quadlights
    this->m_lightPool->updateAllQuadLights(true);
}

void QuadLight::setRotation(const optix::float3 &rotation)
{
    /* Update underlying Quad shape. */
    this->m_quadShape->setRotation(rotation);

    /* Update QuadLight Struct for GPU program. */
    this->updateMatrixParameter();

    //update all quadlights
    this->m_lightPool->updateAllQuadLights(true);
}

void QuadLight::setScale(const optix::float3 &scale)
{
    /* Update underlying Quad shape. */
    this->m_quadShape->setScale(scale);

    /* Update QuadLight Struct for GPU program. */
    this->updateMatrixParameter();

    //update all quadlights
    this->m_lightPool->updateAllQuadLights(true);
}

void QuadLight::setLightIntensity(float intensity)
{
    this->m_intensity = intensity;
    optix::float3 csIntensity = this->m_intensity * this->m_color;
    this->m_csQuadLight.intensity = optix::make_float4(csIntensity.x, csIntensity.y, csIntensity.z, 1.0f);

    this->m_lightPool->updateAllQuadLights(false);
}

void QuadLight::setLightColor(const optix::float3 &color)
{
    this->m_color = color;
    optix::float3 csIntensity = this->m_intensity * this->m_color;
    this->m_csQuadLight.intensity = optix::make_float4(csIntensity.x, csIntensity.y, csIntensity.z, 1.0f);

    this->m_lightPool->updateAllQuadLights(false);
}

namespace CLTest
{
    namespace CommonStruct
    {
        /**
         * A internal struct used for QuadLight testing (SH Integration).
         * Similar to optix::float3 but support index accessor.
         */
        struct QuadLight_float3 {
            float x, y, z;

            float& operator[](int idx) {
                if (idx == 0)return x;
                if (idx == 1)return y;
                if (idx == 2)return z;
                TW_ASSERT(false);
            }
            const float& operator[](int idx) const
            {
                if (idx == 0)return x;
                if (idx == 1)return y;
                if (idx == 2)return z;
                TW_ASSERT(false);
            }
        };
        QuadLight_float3 make_QuadLight_float3(float x, float y, float z)
        {
            QuadLight_float3 fl;
            fl.x = x; fl.y = y; fl.z = z;
            return fl;
        }
    }
}

void QuadLight::ClipQuadToHorizon(optix::float3 L[5], int n)
{
    // detect clipping config
    int config = 0;
    if (L[0].z > 0.0) config += 1;
    if (L[1].z > 0.0) config += 2;
    if (L[2].z > 0.0) config += 4;
    if (L[3].z > 0.0) config += 8;

    // clip
    n = 0;

    if (config == 0)
    {
        // clip all
    }
    else if (config == 1) // V1 clip V2 V3 V4
    {
        n = 3;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 2) // V2 clip V1 V3 V4
    {
        n = 3;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 3) // V1 V2 clip V3 V4
    {
        n = 4;
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
        L[3] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 4) // V3 clip V1 V2 V4
    {
        n = 3;
        L[0] = -L[3].z * L[2] + L[2].z * L[3];
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
    }
    else if (config == 5) // V1 V3 clip V2 V4) impossible
    {
        n = 0;
    }
    else if (config == 6) // V2 V3 clip V1 V4
    {
        n = 4;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 7) // V1 V2 V3 clip V4
    {
        n = 5;
        L[4] = -L[3].z * L[0] + L[0].z * L[3];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 8) // V4 clip V1 V2 V3
    {
        n = 3;
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
        L[1] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = L[3];
    }
    else if (config == 9) // V1 V4 clip V2 V3
    {
        n = 4;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[2].z * L[3] + L[3].z * L[2];
    }
    else if (config == 10) // V2 V4 clip V1 V3) impossible
    {
        n = 0;
    }
    else if (config == 11) // V1 V2 V4 clip V3
    {
        n = 5;
        L[4] = L[3];
        L[3] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 12) // V3 V4 clip V1 V2
    {
        n = 4;
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
    }
    else if (config == 13) // V1 V3 V4 clip V2
    {
        n = 5;
        L[4] = L[3];
        L[3] = L[2];
        L[2] = -L[1].z * L[2] + L[2].z * L[1];
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
    }
    else if (config == 14) // V2 V3 V4 clip V1
    {
        n = 5;
        L[4] = -L[0].z * L[3] + L[3].z * L[0];
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
    }
    else if (config == 15) // V1 V2 V3 V4
    {
        n = 4;
    }

    if (n == 3)
        L[3] = L[0];
    if (n == 4)
        L[4] = L[0];
}

void QuadLight::TestClippingAlgorithm()
{
}

bool QuadLight::TestDiffuseFlmVector_Order3(const std::vector<float>& flmVector, int maximumIteration)
{
    /* Sampling functions. */
    auto uniformSamplingHemisphere = [](float x, float y)->CLTest::CommonStruct::QuadLight_float3
    {
        float z = x;
        float r = std::sqrt(std::max(0.f, 1.f - z * z));
        float phi = 2 * M_PI * y;
        return CLTest::CommonStruct::make_QuadLight_float3(r * std::cos(phi), r * std::sin(phi), z);
    };

    auto pdfHemisphere = []()->float {return 1.f / (2.f*M_PI); };

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);

    constexpr int order = 2;
    Eigen::VectorXf ylmRes(9);
    ylmRes << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    for (int i = 1; i <= maximumIteration; ++i)
    {
        Eigen::VectorXf ylm(9);
        ylm << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

        CLTest::CommonStruct::QuadLight_float3 sample = uniformSamplingHemisphere(dis(gen), dis(gen));
        SHEvalFast(sample, order, ylm);
        assert(ylm.size() == (order + 1)*(order + 1));
        ylmRes += ylm * sample.z / pdfHemisphere();
    }
    ylmRes *= (1.0 / maximumIteration);

    int numFailures = 0;
    for (int i = 0; i < ylmRes.size(); ++i)
    {
        if (std::abs(ylmRes[i] - flmVector[i]) >= 1e-5f)
        {
            ++numFailures;
            std::cout << "Failed Test ylmRes[i]:" << ylmRes[i] << "  flmVector[i]:" << flmVector[i] << std::endl;
        }
    }
    std::cout << "Num of failed tests:" << numFailures << std::endl;
    return numFailures == 0;
}

void QuadLight::initializeAreaLight(optix::Context& context)
{
    constexpr size_t l = 2; // N=3 order SH, lmax = 2, l goes from [0,2]
    constexpr size_t AProws = (l + 1)*(l + 1);
    constexpr size_t APcols = (l + 1)*(2*l + 1);
    TW_ASSERT(AProws == 9 && APcols == 15);
    optix::Buffer areaLightAPMatrixBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, APcols, AProws);
    std::vector<float> APdata{ 0.282095, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0.0232673, 0, 0, -0.0465345, 0, 0, -0.519401, 0, 0, 0, 0, 0, 0, 0,
                               0, 0.411914, 0, 0, 0.397679, 0, 0, 0.247148, 0, 0, 0, 0, 0, 0, 0,
                               0, -0.265119, 0, 0, 0.530239, 0, 0, 0.329531, 0, 0, 0, 0, 0, 0, 0,
                               -0.824085, 0, 2.47225, 0.061849, 0, -0.185547, -0.0179691, 0, 0.0539073, 0.350888, 0, -1.05266, 1.03784, 0, -3.11352,
                               1.40929, 0, -4.22786, -0.170478, 0, 0.511434, -0.0252959, 0, 0.0758877, 0.047984, 0, -0.143952, -1.50602, 0, 4.51806,
                               1.06279, 0, -3.18837, 2.05213, 0, -6.1564, 0.452105, 0, -1.35631, 2.05213, 0, -6.1564, 1.0628, 0, -3.18839,
                               0.679056, 0, -2.03717, 0.688339, 0, -2.06502, 0.28824, 0, -0.864721, 0.707511, 0, -2.12253, 0.423207, 0, -1.26962,
                               -0.768895, 0, 2.30669, -1.19225, 0, 3.57675, 0.101583, 0, -0.304749, -1.14112, 0, 3.42337, -0.439553, 0, 1.31866 };
    void *APMatrixBufferData = areaLightAPMatrixBuffer->map();
    memcpy(APMatrixBufferData, APdata.data(), sizeof(float)*APdata.size());
    areaLightAPMatrixBuffer->unmap();

    context["areaLightAPMatrix"]->setBuffer(areaLightAPMatrixBuffer);

    /* Flm diffuse Vector. */
    std::vector<float> Flm_data{ 0.886227,-0,1.02333,-0,0,-0,0.495416,-0,0 };
    //TW_ASSERT(TestDiffuseFlmVector_Order3(Flm_data, 100000));
    TW_ASSERT(AProws == 9);
    optix::Buffer areaLightFlmVectorBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, AProws);
    void *areaLightFlmVectorBufferData = areaLightFlmVectorBuffer->map();
    memcpy(areaLightFlmVectorBufferData, Flm_data.data(), sizeof(float)*Flm_data.size());
    areaLightFlmVectorBuffer->unmap();

    context["areaLightFlmVector"]->setBuffer(areaLightFlmVectorBuffer);
}