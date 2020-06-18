#pragma once
#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "colvillea/Module/Geometry/GeometryShape.h"
#include "colvillea/Device/Toolkit/Utility.h"

class Application;
class SceneGraph;

/**
 * @brief Analytical sphere geometry shape.
 */
class Sphere : public GeometryShape
{
public:
    /**
     * @brief Factory method for creating a Sphere instance for ordinary
     * usage but not for sphere light.
     *
     * @param[in] sceneGraph
     * @param[in] context
     * @param[in] programsMap      map to store Programs
     * @param[in] centerPosition   center position
     * @param[in] radius           sphere radius
     * @param[in] integrator       integrator of optix::Material type
     * @param[in] materialIndex    material index in |materialBuffer|
     */
    static std::unique_ptr<Sphere> createSphere(SceneGraph *sceneGraph, optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3 &centerPosition, const float radius, optix::Material integrator, int32_t materialIndex)
    {
        return Sphere::createSphereInner(sceneGraph, context, programsMap, centerPosition, radius, integrator, materialIndex);
    }

    /**
     * @brief Factory method for creating a Sphere instance for sphere light.
     *
     * @param[in] context
     * @param[in] programsMap      map to store Programs
     * @param[in] centerPosition   center position
     * @param[in] radius           sphere radius
     * @param[in] sphereLightIndex   index to |sphereLightBuffer|
     * @param[in] integrator       integrator of optix::Material type
     * @param[in] materialIndex    material index in |materialBuffer|
     */
    static std::unique_ptr<Sphere> createSphere(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3 &centerPosition, const float radius, int32_t sphereLightIndex, optix::Material integrator, int32_t materialIndex)
    {
        return Sphere::createSphereInner(context, programsMap, centerPosition, radius, sphereLightIndex, integrator, materialIndex);
    }

private:
    /**
     * @brief Inner factory method.
     */
    template<typename... Ts>
    static std::unique_ptr<Sphere> createSphereInner(Ts&&... params)
    {
        std::unique_ptr<Sphere> sphere = std::make_unique<Sphere>(std::forward<Ts>(params)...);
        sphere->initializeShape();
        return sphere;
    }

public:

    /**
     * @brief Constructor for an ordinary sphere shape.
     *
     */
    Sphere(SceneGraph *sceneGraph, optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3 &centerPosition, const float radius, optix::Material integrator, int32_t materialIndex)
        : GeometryShape(context, programsMap, "Sphere", integrator, materialIndex, "Sphere", IEditableObject::IEditableObjectType::SphereGeometry, false),
        m_sceneGraph(sceneGraph),
        m_position(centerPosition), m_radius(radius),
        m_sphereLightIndex(-1)
    {
        TW_ASSERT(sceneGraph);
        TW_ASSERT(radius > 0.f);
    }

    /**
     * @brief Constructor for a sphere shape to serve as an underlying shape for
     * sphereLight.
     */
    Sphere(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3 &centerPosition, const float radius, int32_t sphereLightIndex, optix::Material integrator, int32_t materialIndex)
        : GeometryShape(context, programsMap, "Sphere", integrator, materialIndex, "Sphere", IEditableObject::IEditableObjectType::QuadGeometry, true),
        m_sceneGraph(nullptr),
        m_position(centerPosition), m_radius(radius),
        m_sphereLightIndex(sphereLightIndex)
    {
        TW_ASSERT(sphereLightIndex >= 0);
        TW_ASSERT(radius > 0.f);
    }

    void initializeShape() override
    {
        TW_ASSERT(this->m_integrator && this->m_materialIndex >= 0);

        /* Create nodes for SceneGraph and initialize. */
        this->setupGeometry();
        GeometryShape::setupGeometryInstance(this->m_integrator);

        this->setMaterialIndex(this->m_materialIndex);
        this->updateMatrixParameter();
        this->m_geometryInstance["sphereLightIndex"]->setInt(this->m_isAreaLight ? this->m_sphereLightIndex : -1);
    }


public:
    optix::float3 getPosition() const
    {
        return this->m_position;
    }

    void setPosition(const optix::float3 &position)
    {
        this->m_position = position;
        this->updateMatrixParameter();
    }

    float getRadius() const
    {
        return this->m_radius;
    }

    void setRadius(const float radius)
    {
        TW_ASSERT(radius > 0.f);

        this->m_radius = radius;
        this->updateMatrixParameter();
    }

    /**
     * @brief Getter method for sphere matrices
     *
     * @param[out] objectToWorld
     * @param[out] worldToObject
     */
    void getMatrix(optix::Matrix4x4 &objectToWorld, optix::Matrix4x4 &worldToObject)
    {
        this->m_geometryInstance["objectToWorld"]->getMatrix4x4(false, objectToWorld.getData());
        worldToObject = objectToWorld.inverse();

        /*TW_ASSERT(optix::Matrix4x4(worldToObject) * optix::Matrix4x4(objectToWorld) == optix::Matrix4x4::identity());*/
    }

    /**
     * @brief Calculate sphere surface area.
     */
    float getSurfaceArea() const override
    {
        return 4.f * M_PIf * this->m_radius * this->m_radius;
    }

    /**
     * @brief Update spherelight index. This is used
     * by LightPool::updateAllQuadLights() to ensure
     * the sphere shape's light index is consistent with
     * LightPool::m_quadLights.
     *
     * @param[in] sphereLightIndex
     * @see LightPool::updateAllQuadLights()
     */
    void setSphereLightIndex(int32_t sphereLightIndex)
    {
        //static_assert(sizeof(int32_t) == sizeof(int), "int32_t != int");
        TW_ASSERT(sphereLightIndex >= 0 && this->m_sphereLightIndex != -1 && this->m_isAreaLight);
        this->m_sphereLightIndex = sphereLightIndex;
        this->m_geometryInstance["sphereLightIndex"]->setInt(this->m_sphereLightIndex);
    }

protected:
    void setupGeometry() override
    {
        /* Set primitive count and call Shape::setupGeometry(). */
        this->m_primitiveCount = 1;
        GeometryShape::setupGeometry();
    }
private:
    void updateMatrixParameter();


private:
    /// Store index to |sphereLightBuffer|, be careful for the circular reference if we want to have another std::shared_ptr to quadLight
    int32_t  m_sphereLightIndex;

    /// Record user-friendly transform elements.
    optix::float3 m_position;
    float         m_radius;

    /// Pointer to SceneGraph
    SceneGraph *m_sceneGraph;
};