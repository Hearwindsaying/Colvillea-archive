#pragma once
#include "colvillea/Module/Light/Light.h"

#include <cmath>

#include "colvillea/Module/Geometry/Quad.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"
#include "colvillea/Device/Toolkit/Utility.h"

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
    /**
     * @brief Factory method for creating a QuadLight instance.
     *
     * @param[in] context
     * @param[in] programsMap  map to store Programs
     * @param[in] intensity    light intensity
     * @param[in] quadShape    underlying Quad shape
     */
    static std::unique_ptr<QuadLight> createQuadLight(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3 intensity, std::shared_ptr<Quad> quadShape)
    {
        std::unique_ptr<QuadLight> quadLight = std::make_unique<QuadLight>(context, programsMap, intensity, quadShape);

        optix::Matrix4x4 lightToWorld, worldToLight;
        quadShape->getMatrix(lightToWorld, worldToLight);

        quadLight->initializeLight();
        return quadLight;
    }

    QuadLight(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3 intensity, std::shared_ptr<Quad> quadShape) :
        Light(context, programsMap, "Quad", "Quad Light", IEditableObject::IEditableObjectType::QuadLight), m_quadShape(quadShape)
    {
        this->m_csQuadLight.intensity = optix::make_float4(intensity.x, intensity.y, intensity.z, 1.f);
    }

    /**
     * @param[in] lightToWorld the transform matrix 
     * will have no effect on QuadLight, whose transform
     * matrix is decided by its underlying Quad shape.
     */
    void initializeLight() 
    {
        /* Create QuadLight Struct for GPU program. */
        this->m_csQuadLight.lightType = CommonStructs::LightType::QuadLight;
        this->m_quadShape->getMatrix(this->m_csQuadLight.lightToWorld, this->m_csQuadLight.worldToLight); /* note that |lightToWorld| is directly decided by |m_quadShape| */
        this->m_csQuadLight.reverseOrientation = this->m_quadShape->isFlippedGeometryNormal();
        this->m_csQuadLight.invSurfaceArea = 1.f / this->m_quadShape->getSurfaceArea();

        optix::float3 nn = TwUtil::xfmNormal(
            optix::make_float3(0.f, 0.f, (this->m_csQuadLight.reverseOrientation ? -1.f : 1.f)), this->m_csQuadLight.lightToWorld);
        optix::float3 nn2 = TwUtil::xfmNormal(
            optix::make_float3(0.f, 0.f, (this->m_csQuadLight.reverseOrientation ? -1.f : 1.f)), this->m_csQuadLight.worldToLight);
        std::cout << "[Info] nn for xfmNormal:(" << nn.x << "," << nn.y << "," << nn.z << ")." << std::endl;
        std::cout << "[Info] nn2 for xfmNormal:(" << nn2.x << "," << nn2.y << "," << nn2.z << ")." << std::endl;
        TW_ASSERT(std::fabs(1.f - (TwUtil::sqr_length(nn))) <= 1e-6f);
    }

 

    const CommonStructs::QuadLight &getCommonStructsLight() const
    {
        return this->m_csQuadLight;
    }

private:
    CommonStructs::QuadLight m_csQuadLight;
    std::shared_ptr<Quad>    m_quadShape;
};

