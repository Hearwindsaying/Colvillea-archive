#pragma once

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include <map>

#include "colvillea/Module/Geometry/Shape.h"
#include "colvillea/Application/TWAssert.h"

/**
 * @brief GeometryShape class indicates its shape
 * type is GeometryTriangles in OptiX. It means
 * that it could use RTCore acceleration for 
 * geometryTriangle when Turing GPU available.
 */
class GeometryTrianglesShape : public Shape
{
public:
    GeometryTrianglesShape(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const std::string &shapeClassName, optix::Material integrator, const int materialIndex) :
        Shape(context, programsMap, integrator, materialIndex)
    {
        std::cout << "[Info] Derived class name from Shape is: " << shapeClassName << std::endl;

        /* Load attributes program */
        auto programItr = this->m_programsMap.find("Attributes_" + shapeClassName);
        TW_ASSERT(programItr != this->m_programsMap.end());
        this->m_attributesProgram = programItr->second;
    }

protected:
    void setupGeometry() override
    {
        TW_ASSERT(this->m_attributesProgram);
        TW_ASSERT(this->m_primitiveCount > 0);

        if (!this->m_geometryTriangles)
        {
            /* Note that some shapes may need to createGeometry in advance
             * -- to setup buffers and those common parameters like primitive
             * -- count and programs could be set later.
             * However, for simpler shape that does not need to do those
             * -- extra stuff could simply invoke Shape::setupGeometry()
             * -- to finish all fundamental work. */
            this->m_geometryTriangles = this->m_context->createGeometryTriangles();
        }

        this->m_geometryTriangles->setPrimitiveCount(this->m_primitiveCount);
        this->m_geometryTriangles->setAttributeProgram(this->m_attributesProgram);
        this->m_geometryTriangles->setBuildFlags(RTgeometrybuildflags::RT_GEOMETRY_BUILD_FLAG_NONE);
        this->m_geometryTriangles->setFlagsPerMaterial(0, RTgeometryflags::RT_GEOMETRY_FLAG_DISABLE_ANYHIT);
        /* Note that we use geometryInstance's flage to override geometry's flag. */
    }

    void setupGeometryInstance(optix::Material integrator) override
    {
        TW_ASSERT(this->m_geometryTriangles && integrator);

        this->m_geometryInstance = this->m_context->createGeometryInstance(this->m_geometryTriangles, integrator);
    }

protected:
    optix::GeometryTriangles m_geometryTriangles;
    optix::Program           m_attributesProgram;
};