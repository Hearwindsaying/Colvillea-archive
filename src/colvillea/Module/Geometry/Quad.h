#pragma once
#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "colvillea/Module/Geometry/GeometryShape.h"
#include "colvillea/Device/Toolkit/Utility.h"
#include "tinyobjloader/tiny_obj_loader.h"

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
class Quad : public GeometryShape
{
public:
    /**
     * @brief Factory method for creating a Quad instance for ordinary
     * usage but not for quad light.
     *
     * @param[in] context
     * @param[in] programsMap      map to store Programs
     * @param[in] position
     * @param[in] rotation         XYZ rotation angle in radian
     * @param[in] scale            Z-component is zero 
     * @param[in] integrator       integrator of optix::Material type
     * @param[in] materialIndex    material index in |materialBuffer|
     */
    static std::unique_ptr<Quad> createQuad(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3 &position, const optix::float3 &rotation, const optix::float3 &scale, optix::Material integrator, const int materialIndex)
    {
        return Quad::createQuadInner(context, programsMap, position, rotation, scale, integrator, materialIndex);
    }

    /**
     * @brief Factory method for creating a Quad instance for quad light.
     *
     * @param[in] context
     * @param[in] programsMap      map to store Programs
     * @param[in] position
     * @param[in] rotation         XYZ rotation angle in radian
     * @param[in] scale            Z-component is zero
     * @param[in] quadLightIndex   index to |quadLightBuffer|
     * @param[in] integrator       integrator of optix::Material type
     * @param[in] materialIndex    material index in |materialBuffer|
     */
    static std::unique_ptr<Quad> createQuad(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3 &position, const optix::float3 &rotation, const optix::float3 &scale, int quadLightIndex, optix::Material integrator, const int materialIndex)
    {
        return Quad::createQuadInner(context, programsMap, position, rotation, scale, quadLightIndex, integrator, materialIndex);
    }

private:
    /**
     * @brief Inner factory method.
     */
    template<typename... Ts>
    static std::unique_ptr<Quad> createQuadInner(Ts&&... params)
    {
        std::unique_ptr<Quad> quad = std::make_unique<Quad>(std::forward<Ts>(params)...);
        quad->initializeShape();
        return quad;
    }


public:

    /**
     * @brief Constructor for an ordinary quad shape given transform matrix.
     * 
     * @param[in] position
     * @param[in] rotation         XYZ rotation angle in radian
     * @param[in] scale            Z-component is zero
     */
    Quad(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3 &position, const optix::float3 &rotation, const optix::float3 &scale, optix::Material integrator, const int materialIndex)
        : GeometryShape(context, programsMap, "Quad", integrator, materialIndex), 
        m_position(position), m_rotationRad(rotation), m_scale(scale)
    {
        /* Check whether transform matrix has z-component scale. */
        TW_ASSERT(m_scale.z == 1.f);
        if (m_scale.z != 1.f)
            std::cerr << "[Warning] Quad shape has z-component scale, which could lead to undefined behavior!" << std::endl;
        std::cout << "[Info] Scale component for quad shape is: (" << scale.x << "," << scale.y << ")." << std::endl;
    }

    /**
     * @brief Constructor for a quad shape to serve as an underlying shape for
     * quadLight.
     *
     * @param[in] position
     * @param[in] rotation         XYZ rotation angle in radian
     * @param[in] scale            Z-component is zero
     */
    Quad(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const optix::float3 &position, const optix::float3 &rotation, const optix::float3 &scale, int quadLightIndex, optix::Material integrator, const int materialIndex)
        : GeometryShape(context, programsMap, "Quad", integrator, materialIndex),
         m_position(position), m_rotationRad(rotation), m_scale(scale),  
         m_quadLightIndex(quadLightIndex), m_isAreaLight(true)
    {
        TW_ASSERT(quadLightIndex >= 0);

        TW_ASSERT(m_scale.z == 1.f);
        /* Check whether transform matrix has z-component scale. */
        if (m_scale.z != 1.f)
            std::cerr << "[Warning] Quad shape has z-component scale, which could lead to undefined behavior!" << std::endl;
        std::cout << "[Info] Scale component for quad shape is: (" << scale.x << "," << scale.y << ")." << std::endl;
    }

    void initializeShape() override
    {
        TW_ASSERT(this->m_integrator && this->m_materialIndex >= 0);

        /* Create nodes for SceneGraph and initialize. */
        this->setupGeometry();
        GeometryShape::setupGeometryInstance(this->m_integrator);

        this->setMaterialIndex(this->m_materialIndex);
        this->updateMatrixParameter();
        this->m_geometryInstance["reverseOrientation"]->setInt(0);
        this->m_geometryInstance["quadLightIndex"]->setInt(this->m_isAreaLight ? this->m_quadLightIndex : -1);
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

    optix::float3 getRotation() const
    {
        return this->m_rotationRad;
    }

    void setRotation(const optix::float3 &rotation)
    {
        this->m_rotationRad = rotation;
        this->updateMatrixParameter();
    }

    optix::float3 getScale() const
    {
        return this->m_scale;
    }

    void setScale(const optix::float3 &scale)
    {
        TW_ASSERT(m_scale.z == 1.f);
        if (scale.z != 1.f)
        {
            std::cout << "[Info] Quad shape scale's z-component is not zero!" << std::endl;
        }

        this->m_scale = scale;
        this->updateMatrixParameter();
    }

    /**
     * @brief Getter method for quad matrices
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
     * @brief Calculate quad surface area.
     */
    float getSurfaceArea() const override
    {
        //todo:delete assertion:
        /*float area = optix::length(
            TwUtil::xfmVector(optix::make_float3(2, 0, 0), this->m_objectToWorld)) *
                     optix::length(
            TwUtil::xfmVector(optix::make_float3(0, 2, 0), this->m_objectToWorld));
        TW_ASSERT(area == TwUtil::getXScale(this->m_objectToWorld) * TwUtil::getYScale(this->m_objectToWorld) * 4);

        return TwUtil::getXScale(this->m_objectToWorld) * TwUtil::getYScale(this->m_objectToWorld) * 4;*/
        return this->m_scale.x * this->m_scale.y;
    }

    /**
     * @brief Update quadlight index. This is used
     * by LightPool::updateAllQuadLights() to ensure
     * the quad shape's light index is consistent with
     * LightPool::m_quadLights.
     * 
     * @param[in] quadLightIndex
     * @see LightPool::updateAllQuadLights()
     */
    void setQuadLightIndex(int quadLightIndex)
    {
        TW_ASSERT(quadLightIndex >= 0 && this->m_quadLightIndex != -1 && this->m_isAreaLight);
        this->m_quadLightIndex = quadLightIndex;
        this->m_geometryInstance["quadLightIndex"]->setInt(this->m_quadLightIndex);
    }

protected:
    void setupGeometry() override
    {
        /* Set primitive count and call Shape::setupGeometry(). */
        this->m_primitiveCount = 1;
        GeometryShape::setupGeometry();
    }

private:
    void updateMatrixParameter()
    {
        TW_ASSERT(this->m_geometryInstance);

        optix::Matrix4x4 objectToWorld = 
            optix::Matrix4x4::translate(this->m_position) * 
            optix::Matrix4x4::rotate(this->m_rotationRad.x, optix::make_float3(1.f, 0.f, 0.f)) *
            optix::Matrix4x4::rotate(this->m_rotationRad.y, optix::make_float3(0.f, 1.f, 0.f)) *
            optix::Matrix4x4::rotate(this->m_rotationRad.z, optix::make_float3(0.f, 0.f, 1.f)) *
            optix::Matrix4x4::scale(this->m_scale);

        std::cout << TwUtil::getXScale(objectToWorld) << " " << TwUtil::getYScale(objectToWorld) << std::endl;

        this->m_geometryInstance["objectToWorld"]->setMatrix4x4fv(false, objectToWorld.getData());
        this->m_geometryInstance["worldToObject"]->setMatrix4x4fv(false, objectToWorld.inverse().getData());
    }

private:
    /// Store index to |quadLightBuffer|, be careful for the circular reference if we want to have another std::shared_ptr to quadLight
    bool m_isAreaLight;
    int  m_quadLightIndex;

    /// Record user-friendly transform elements.
    optix::float3 m_rotationRad;
    optix::float3 m_position;
    optix::float3 m_scale;
};