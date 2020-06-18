#pragma once

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include <map>

#include "colvillea/Module/Geometry/Shape.h"
#include "colvillea/Application/TWAssert.h"

/**
 * @brief GeometryShape class indicates its shape
 * type is ordinary Geometry in OptiX. It means
 * that it uses its own intersection and bounding
 * box program which does not get hardware acceleration
 * supported by Turing cores.
 */
class GeometryShape : public Shape
{
public:
    GeometryShape(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const std::string &shapeClassName, optix::Material integrator, int32_t materialIndex, const std::string &shapeObjectName, IEditableObject::IEditableObjectType objectType, bool isAreaLight) :
        Shape(context, programsMap, integrator, materialIndex, shapeObjectName, objectType)
    {
        std::cout << "[Info] Derived class name from Shape is: " << shapeClassName << std::endl;

        /* Load boundingbox and intersection program */
        auto programItr = this->m_programsMap.find("BoundingBox_" + shapeClassName);
        TW_ASSERT(programItr != this->m_programsMap.end());
        this->m_boundingBoxProgram = programItr->second;

        programItr = this->m_programsMap.find("Intersect_" + shapeClassName);
        TW_ASSERT(programItr != this->m_programsMap.end());
        this->m_intersectionProgram = programItr->second;

        this->m_isAreaLight = isAreaLight;
    }

    optix::Geometry getGeometry() const
    {
        return this->m_geometry;
    }


    bool isAreaLight() const
    {
        return this->m_isAreaLight;
    }

protected:
    void setupGeometry() override
    {
        TW_ASSERT(this->m_boundingBoxProgram && this->m_intersectionProgram);
        TW_ASSERT(this->m_primitiveCount > 0);

        if (!this->m_geometry)
        {
            /* Note that some shapes may need to createGeometry in advance
             * -- to setup buffers and those common parameters like primitive
             * -- count and programs could be set later.
             * However, for simpler shape that does not need to do those
             * -- extra stuff could simply invoke Shape::setupGeometry()
             * -- to finish all fundamental work. */
            this->m_geometry = this->m_context->createGeometry();
        }

        this->m_geometry->setPrimitiveCount(this->m_primitiveCount);
        this->m_geometry->setBoundingBoxProgram(this->m_boundingBoxProgram);
        this->m_geometry->setIntersectionProgram(this->m_intersectionProgram);
        this->m_geometry->setFlags(RTgeometryflags::RT_GEOMETRY_FLAG_DISABLE_ANYHIT);
    }

    void setupGeometryInstance(optix::Material integrator) override
    {
        TW_ASSERT(this->m_geometry && integrator);

        this->m_geometryInstance = this->m_context->createGeometryInstance();
        this->m_geometryInstance->setGeometry(this->m_geometry);
        this->m_geometryInstance->addMaterial(integrator);
    }

protected:
    optix::Geometry m_geometry;
    optix::Program  m_boundingBoxProgram;
    optix::Program  m_intersectionProgram;

    /// Area Light support for non-triangle mesh GeometryShape.
    bool m_isAreaLight;
};