#pragma once
#include "Light.h"

#include <cmath>

#include "../../Device/Toolkit/CommonStructs.h"
#include "../../Device/Toolkit/Utility.h"

/**
 * @brief QuadLight is the simplest area light 
 * supported in Colvillea. It describes a diffusely
 * emissive light from a rectangle which is able
 * to produce soft shadow and be physically plausible.
 * Always prefer using quadlight to create two planar
 * triangles and attach emissive material to get 
 * a quadlight due to performance consideration.
 *
 * @note This class contains description of
 * what a single Quadlight should be. Buffers
 * storing quadlight is delegated to SceneGraph.
 */
class QuadLight : public Light
{
public:
    QuadLight(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3 intensity, std::shared_ptr<Quad> quadShape) : 
        Light(context, programsMap, "Quad"), m_quadShape(quadShape)
    {
        this->m_csQuadLight.intensity = optix::make_float4(intensity.x, intensity.y, intensity.z, 1.f);
    }

    /**
     * @note todo:Never use |lightToWorld| parameter!
     */
    void loadLight(const optix::Matrix4x4 &lightToWorld) override
    {
        /* Create QuadLight Struct for GPU program. */
        this->m_csQuadLight.lightType = CommonStructs::LightType::QuadLight;
        this->m_quadShape->getMatrix(this->m_csQuadLight.lightToWorld, this->m_csQuadLight.worldToLight);
        this->m_csQuadLight.reverseOrientation = this->m_quadShape->isFlippedGeometryNormal();
        this->m_csQuadLight.invSurfaceArea = 1.f / this->m_quadShape->getSurfaceArea();

        //todo:deleteme assertions:
        optix::float3 nn = TwUtil::xfmNormal(
            optix::make_float3(0.f, 0.f, (this->m_csQuadLight.reverseOrientation ? -1.f : 1.f)), this->m_csQuadLight.lightToWorld);
        optix::float3 nn2 = TwUtil::xfmNormal(
            optix::make_float3(0.f, 0.f, (this->m_csQuadLight.reverseOrientation ? -1.f : 1.f)), this->m_csQuadLight.worldToLight);
        std::cout << "[Info] nn for xfmNormal:(" << nn.x << "," << nn.y << "," << nn.z << ")." << std::endl;
        std::cout << "[Info] nn2 for xfmNormal:(" << nn2.x << "," << nn2.y << "," << nn2.z << ")." << std::endl;
        TW_ASSERT(std::fabs(1.f - (TwUtil::sqr_length(nn))) <= 1e-6f);
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
    static void initQuadLight(optix::Context context)
    {
        
    }

    const CommonStructs::QuadLight &getCommonStructsLight() const
    {//todo:review
        return this->m_csQuadLight;
    }

private:
    CommonStructs::QuadLight m_csQuadLight;
    std::shared_ptr<Quad>    m_quadShape;
};

