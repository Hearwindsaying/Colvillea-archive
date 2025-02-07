#pragma once

#define  CL_CHECK_MEMORY_LEAKS
#ifdef CL_CHECK_MEMORY_LEAKS
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#define CL_CHECK_MEMORY_LEAKS_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new CL_CHECK_MEMORY_LEAKS_NEW
#endif

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include <map>

#include "colvillea/Application/TWAssert.h"
#include "colvillea/Application/GlobalDefs.h"



/** 
 * @brief Shape class is the base class for all supported
 * primitive types in Colvillea. It should be able to
 * provide fundamental function to prepare geometry for
 * rendering on GPU.
 */
class Shape : public IEditableObject
{
public:
	/**
	 * @brief constructor for Shape class, collect all necessary
	 * parameters to initialize and set up GeometryInstance.
	 */
    Shape(optix::Context context, const std::map<std::string, optix::Program> &programsMap, optix::Material integrator, int32_t materialIndex, const std::string &shapeObjectName, IEditableObject::IEditableObjectType objectType) :
        IEditableObject(shapeObjectName, objectType),
        m_context(context), m_programsMap(programsMap), 
        m_materialIndex(materialIndex), m_integrator(integrator), m_primitiveCount(-1)
	{
		
	}

	/**
	 * @brief Setup shape properties and prepare for SceneGraph.
	 */
    virtual void initializeShape() = 0;

    /************************************************************************/
    /*                         Getters & Setters                            */
    /************************************************************************/

	/**
	 * @brief materialIndex setter
	 */
	void setMaterialIndex(int32_t materialIndex)
	{
		TW_ASSERT(materialIndex >= 0);

		this->m_materialIndex = materialIndex;
		this->updateMaterialParameter();
	}

	/**
	 * @brief materialIndex getter
	 */
    int32_t getMaterialIndex() const
	{
		return this->m_materialIndex;
	}

	/**
	 * @brief Getter for geometryInstance. The const
	 * qualifier is a hint that it should not be 
	 * modified other place.
	 */
	optix::GeometryInstance getGeometryInstance() const
	{
		return this->m_geometryInstance;
	}

    virtual float getSurfaceArea() const
    {//todo:turn into pure virtual.
        throw std::runtime_error("Derived shape class method getSurfaceArea() not implemented!");
    }

    /**
     * @brief Change Integrator type after setting up GeometryInstance.
     * @note It's named as "change" prefix because it should not be used
     * during initialization.
     * @param[in] integratorMaterial  Expected integrator to use
     * @todo change "change" prefix to "set" which supports both initialization
     * and later GUI interaction.
     */
    void changeIntegrator(optix::Material integratorMaterial)
    {
        this->m_integrator = integratorMaterial;
        this->m_geometryInstance->setMaterial(0, this->m_integrator);
    }

protected:
    /************************************************************************/
    /*              Helper functions for initialization                     */
    /************************************************************************/

	/**
	 * @brief Setup geometry for the shape, including creating
	 * geometry node, setting bounding box program and intersect
	 * program, setting appropriate parameters, etc.
	 *
	 * This is a part of Shape::initializeShape().
	 * 
	 * @note This is designed to be virtual (a interface) which
	 * needs further information of the shape's underlying geometry
	 * type (whether it is GeometryTriangles or Geometry) for
	 * implementation.
	 */
    virtual void setupGeometry() = 0;

	/**
	 * @brief Setup geometryInstance for the shape, including
	 * creating geometryInstance, loading buffers, setting up
	 * integrators by specified |m_integrator| program.
	 *
	 * @note Integrators are binded with material program which
	 * doesn't represent any material parameters for the shape.
	 * Keep in mind that |m_integrator| parameter is actually a
	 * material node.
	 *
	 * The |m_integrator| parameter specify which integrator we want
	 * to use and this should be always the same as other shapes
	 * in the scene for consistence.
	 *
	 * Fetch |m_integrator| program from SceneGraph
	 *
	 * This is a part of Shape::initializeShape().
	 * 
     * @note This is designed to be virtual (a interface) which
     * needs further information of the shape's underlying geometry
     * type (whether it is GeometryTriangles or Geometry) for
     * implementation.
	 */
    virtual void setupGeometryInstance(optix::Material integrator) = 0;

private:
    /************************************************************************/
    /*                             Inner Setters                            */
    /************************************************************************/

	/**
	 * @brief Setup material parameter by |m_materialIndex|.
	 * Note that this has nothing to do with Material node,
	 * a variable setting to GeometryInstance node will be
	 * used instead.
	 *
	 * This is a part of Shape::initializeShape() and setter
	 * for updating material parameters as well.
	 */
	void updateMaterialParameter()
	{
		TW_ASSERT(this->m_geometryInstance);

		this->m_geometryInstance["materialIndex"]->setInt(this->m_materialIndex);
	}

protected:
    optix::Context m_context;
    const std::map<std::string, optix::Program> &m_programsMap;

    /// The holding underlying geometry could be GeometryTriangles or Geometry.
	optix::GeometryInstance m_geometryInstance;
	
    /// Count of the shape.
    unsigned int    m_primitiveCount;

    /// Integrator for the shape.
    optix::Material m_integrator;

    /// Material index to |materialBuffer|.
    int32_t        m_materialIndex;
};