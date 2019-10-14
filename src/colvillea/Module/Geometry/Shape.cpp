#include "Shape.h"
#include "colvillea/Application/TWAssert.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"

void Shape::setupGeometry()
{
	TW_ASSERT(this->m_boundingBoxProgram && this->m_intersectionProgram);
	TW_ASSERT(this->m_primitiveCount > 0);

    if (!this->m_geometry)
    {
        /* Note that some shape may need to createGeometry in advance to setup buffers and those common parameters like primitive count and programs could be set later. 
         * However, for simpler shape that does not need to do those extra stuff could simply invoke Shape::setupGeometry() to finish all fundamental work. */
        this->m_geometry = this->m_context->createGeometry();
    }
        
	this->m_geometry->setPrimitiveCount(this->m_primitiveCount);
	this->m_geometry->setBoundingBoxProgram(this->m_boundingBoxProgram);
	this->m_geometry->setIntersectionProgram(this->m_intersectionProgram);
}

void Shape::setupGeometryInstance(optix::Material integrator)
{
	TW_ASSERT(this->m_geometry && integrator);

	this->m_geometryInstance = this->m_context->createGeometryInstance();
	this->m_geometryInstance->setGeometry(this->m_geometry);
	this->m_geometryInstance->addMaterial(integrator);
}
