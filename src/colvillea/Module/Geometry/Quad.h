#pragma once
//#include "../../Application/Application.h"

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "Shape.h"
#include "../../Device/Toolkit/Utility.h"
#include "../../../tinyobjloader/tiny_obj_loader.h"

class Application;

/**
 * @brief Quad represents a planar quadrilateral or
 * known as rectangle shape. When no transformation
 * is applied, it occpuies on [-1,-1]*[1,1] on x-y 
 * plane.
 * This simple shape could be useful when a plane
 * or quadlight is required but we don't want to
 * load a trianglemesh. It also helps to improve
 * performance while applying to quadlight.
 */
class Quad : public Shape
{
public:
    /**
     * @brief Constructor for an ordinary quad shape given transform matrix.
     * 
     * @note |objectToWorld| matrix should have no scale component in z-axis.
     */
    Quad(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::Matrix4x4 &objectToWorld)
        : Shape(context, programsMap, "Quad"), m_objectToWorld(objectToWorld), m_worldToObject(objectToWorld.inverse())
    {
        /* Check whether transform matrix has z-component scale. */
        if (TwUtil::hasZScale(objectToWorld))
            std::cerr << "[Warning] Quad shape has z-component scale, which could lead to undefined behavior!" << std::endl;
        std::cout << "[Info] Scale component for quad shape is: (" << TwUtil::getXScale(objectToWorld) << "," << TwUtil::getYScale(objectToWorld) << ")." << std::endl;
    }

    /**
     * @brief Constructor for a quad shape to serve as an underlying shape for
     * quadLight.
     *
     * @note |objectToWorld| matrix should have no scale component in z-axis.
     */
    Quad(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::Matrix4x4 &objectToWorld, int quadLightIndex)
        : Shape(context, programsMap, "Quad"),
          m_objectToWorld(objectToWorld), m_worldToObject(objectToWorld.inverse()), m_quadLightIndex(quadLightIndex), m_isAreaLight(true)
    {
        TW_ASSERT(quadLightIndex >= 0);

        /* Check whether transform matrix has z-component scale. */
        if (TwUtil::hasZScale(objectToWorld))
            std::cerr << "[Warning] Quad shape has z-component scale, which could lead to undefined behavior!" << std::endl;
        std::cout << "[Info] Scale component for quad shape is: (" << TwUtil::getXScale(objectToWorld) << "," << TwUtil::getYScale(objectToWorld) << ")." << std::endl;
    }

    void loadShape(optix::Material integrator, const int materialIndex) override
    {
        TW_ASSERT(integrator && materialIndex >= 0);

        /* Create nodes for SceneGraph and initialize. */
        this->setupGeometry();
        this->setupGeometryInstance(integrator);

        this->setMaterialIndex(materialIndex);
        this->updateMatrixParameter();
        this->m_geometryInstance["reverseOrientation"]->setInt(0);
        this->m_geometryInstance["quadLightIndex"]->setInt(this->m_isAreaLight ? this->m_quadLightIndex : -1);
    }

    /**
     * @brief Setter method for quad matrices.
     */
    void setMatrix(const optix::Matrix4x4 &objectToWorld)
    {
        this->m_objectToWorld = objectToWorld;
        this->m_worldToObject = objectToWorld.inverse();
        updateMatrixParameter();
    }

    /**
     * @brief Getter method for quad matrices
     * 
     * @param[out] objectToWorld
     * @param[out] worldToObject
     */
    void getMatrix(optix::Matrix4x4 &objectToWorld, optix::Matrix4x4 &worldToObject)
    {
        objectToWorld = this->m_objectToWorld;
        worldToObject = this->m_worldToObject;
    }

    /**
     * @brief Calculate quad surface area.
     */
    float getSurfaceArea() const override
    {
        //todo:delete assertion:
        float area = optix::length(
            TwUtil::xfmVector(optix::make_float3(2, 0, 0), this->m_objectToWorld)) *
                     optix::length(
            TwUtil::xfmVector(optix::make_float3(0, 2, 0), this->m_objectToWorld));
        TW_ASSERT(area == TwUtil::getXScale(this->m_objectToWorld) * TwUtil::getYScale(this->m_objectToWorld) * 4);

        return TwUtil::getXScale(this->m_objectToWorld) * TwUtil::getYScale(this->m_objectToWorld) * 4;
    }

private:
    void setupGeometry()
    {
        /* Set primitive count and call Shape::setupGeometry(). */
        this->m_primitiveCount = 1;
        Shape::setupGeometry();
    }

    void updateMatrixParameter()
    {
        TW_ASSERT(this->m_geometryInstance);

        this->m_geometryInstance["objectToWorld"]->setMatrix4x4fv(false, this->m_objectToWorld.getData());
        this->m_geometryInstance["worldToObject"]->setMatrix4x4fv(false, this->m_worldToObject.getData());
    }

private:
    /// Transform matrix for quad shape
    optix::Matrix4x4 m_objectToWorld, m_worldToObject;

    /// Store index to |quadLightBuffer|, be careful for the circular reference if we want to have another std::shared_ptr to quadLight
    bool m_isAreaLight;
    int  m_quadLightIndex;
};