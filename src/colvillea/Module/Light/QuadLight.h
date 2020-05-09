#pragma once
#include "colvillea/Module/Light/Light.h"

#include <cmath>

#include "colvillea/Module/Geometry/Quad.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"
#include "colvillea/Device/Toolkit/Utility.h"

class LightPool;

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
     * @param[in] color        light color
     * @param[in] intensity    light intensity
     * @param[in] quadShape    underlying Quad shape
     */
    static std::unique_ptr<QuadLight> createQuadLight(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3& color, float intensity, std::shared_ptr<Quad> quadShape, LightPool *lightPool)
    {
        std::unique_ptr<QuadLight> quadLight = std::make_unique<QuadLight>(context, programsMap, color, intensity, quadShape, lightPool);
        return quadLight;
    }

    QuadLight(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3& color, float intensity, std::shared_ptr<Quad> quadShape, LightPool *lightPool) :
        Light(context, programsMap, "Quad", "Quad Light", IEditableObject::IEditableObjectType::QuadLight), m_quadShape(quadShape), m_intensity(intensity), m_color(color), m_lightPool(lightPool)
    {
        optix::float3 csIntensity = this->m_intensity * this->m_color;

        /* Create QuadLight Struct for GPU program. */
        this->m_csQuadLight.lightType = CommonStructs::LightType::QuadLight;
        this->m_quadShape->getMatrix(this->m_csQuadLight.lightToWorld, this->m_csQuadLight.worldToLight); /* note that |lightToWorld| is directly decided by |m_quadShape| */
        this->m_csQuadLight.reverseOrientation = this->m_quadShape->isFlippedGeometryNormal();
        this->m_csQuadLight.invSurfaceArea = 1.f / this->m_quadShape->getSurfaceArea();
        this->m_csQuadLight.intensity = optix::make_float4(csIntensity.x, csIntensity.y, csIntensity.z, 1.0f);

        optix::float3 nn = TwUtil::xfmNormal(
            optix::make_float3(0.f, 0.f, (this->m_csQuadLight.reverseOrientation ? -1.f : 1.f)), this->m_csQuadLight.lightToWorld);
        optix::float3 nn2 = TwUtil::xfmNormal(
            optix::make_float3(0.f, 0.f, (this->m_csQuadLight.reverseOrientation ? -1.f : 1.f)), this->m_csQuadLight.worldToLight);
        std::cout << "[Info] nn for xfmNormal:(" << nn.x << "," << nn.y << "," << nn.z << ")." << std::endl;
        std::cout << "[Info] nn2 for xfmNormal:(" << nn2.x << "," << nn2.y << "," << nn2.z << ")." << std::endl;
        TW_ASSERT(std::fabs(1.f - (TwUtil::sqr_length(nn))) <= 1e-6f);
    }

    float getLightIntensity() const
    {
        return this->m_intensity;
    }

    void setLightIntensity(float intensity);

    optix::float3 getLightColor() const
    {
        return this->m_color;
    }

    void setLightColor(const optix::float3 &color);

    optix::float3 getPosition() const
    {
        return this->m_quadShape->getPosition();
    }

    void setPosition(const optix::float3 &position);

    optix::float3 getRotation() const
    {
        return this->m_quadShape->getRotation();
    }

    void setRotation(const optix::float3 &rotation);

    optix::float3 getScale() const
    {
        return this->m_quadShape->getScale();
    }

    void setScale(const optix::float3 &scale);

    const CommonStructs::QuadLight &getCommonStructsLight() const
    {
        return this->m_csQuadLight;
    }

    std::shared_ptr<Quad> getQuadShape() const
    {
        return this->m_quadShape;
    }

    /************************************************************************/
    /*         Integrating Clipped Spherical Harmonics Expansions           */
    /************************************************************************/
    /**
     * @brief Clip Quad to upper hemisphere.
     * @param L[] Quad in shading coordinates
     * @param n   number of vertices after clipping, the clipped polygon will
     * be written back to L[].
     */
    static void ClipQuadToHorizon(optix::float3 L[5], int &n );

    static void TestSolidAngle();
    static void TestZHRecurrence();

    /**
     * @brief UnitTest function for checking ZHIntegral (i.e. P*Cp Column Vector in Laurent's method).
     * Compare the MC Integration result with given parameters.
     * @param order
     * @param lobeDirections
     * @param maximumIteration
     */
    static bool TestZHIntegral(int order, const std::vector<optix::float3> &lobeDirections, int maximumIteration);

    /**
     * @brief UnitTest function for checking diffuse BRDF SH projection. Compare
     * the Monte Carlo Integration result with given |flmVector|.
     * @param flmVector         coeff vector to be validated
     * @param maximumIteration  maximum iteration times for MC
     * @return test passed?
     */
    static bool TestDiffuseFlmVector_Order3(const std::vector<float>& flmVector, int maximumIteration);

    /**
     * @brief Initialize AreaLight AP Matrix for "Integrating Clipped Spherical Harmonics Expansions".
     * Should be called only once for AreaLight.
     */
    static void initializeAreaLight(optix::Context& context);

    /************************************************************************/
    /*         Integrating Clipped Spherical Harmonics Expansions           */
    /************************************************************************/

private:
    /**
     * @brief Update |m_csQuadLight.lightToWorld| and its inverse,
     * and inverse of surface area.
     */
    void updateMatrixParameter()
    {
        this->m_quadShape->getMatrix(this->m_csQuadLight.lightToWorld, this->m_csQuadLight.worldToLight); /* note that |lightToWorld| is directly decided by |m_quadShape| */
        this->m_csQuadLight.invSurfaceArea = 1.f / this->m_quadShape->getSurfaceArea();
    }

    

private:
    LightPool *m_lightPool;

    CommonStructs::QuadLight m_csQuadLight;
    std::shared_ptr<Quad>    m_quadShape;

    /// Color (host only)
    optix::float3 m_color;
    /// Intensity (host only)
    float m_intensity;
};

