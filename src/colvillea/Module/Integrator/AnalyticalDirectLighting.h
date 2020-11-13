#pragma once
//#include "../../Application/Application.h"

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "colvillea/Module/Image/ImageLoader.h"
#include "colvillea/Module/Integrator/Integrator.h"

/**
 * @brief Analytical Direct Lighting using Linear Transform Cosines.
 * 
 * @ref [Heitz et al. 2016] Real-Time Polygonal-Light Shading with Linearly Transformed Cosines
 */
class AnalyticalDirectLighting : public Integrator
{
public:
    /**
     * @brief Factory method for creating a DirectLighting instance.
     *
     * @param[in] context
     * @param[in] programsMap  map to store Programs
     */
    static std::unique_ptr<AnalyticalDirectLighting> createIntegrator(const optix::Context context, const std::map<std::string, optix::Program>& programsMap)
    {
        std::unique_ptr<AnalyticalDirectLighting> adirectLighting = std::make_unique<AnalyticalDirectLighting>(context, programsMap);
        adirectLighting->initializeIntegratorMaterialNode();
        return adirectLighting;
    }

    AnalyticalDirectLighting(optix::Context context, const std::map<std::string, optix::Program>& programsMap) : Integrator(context, programsMap, "AnalyticalDirectLighting", IntegratorType::AnalyticalDirectLighting)
    {
        auto getDataDirectoryPath = []()->std::string
        {
            const std::string filename = __FILE__;
            std::string::size_type extension_index = filename.find_last_of("\\");
            std::string filePath = extension_index != std::string::npos ?
                filename.substr(0, extension_index) :
                std::string();

            return filePath + "\\..\\data\\";
        };

        this->m_ltc1TS = ImageLoader::LoadImageTexture(context, getDataDirectoryPath() + "ltc_1.exr", optix::make_float4(0.f));
        CommonStructs::LTCBuffers ltcBuffersHost;
        ltcBuffersHost.ltc1 = this->m_ltc1TS->getId();
        context["ltcBuffers"]->setUserData(sizeof(CommonStructs::LTCBuffers), &ltcBuffersHost);
    }

protected:
    optix::TextureSampler m_ltc1TS;
};