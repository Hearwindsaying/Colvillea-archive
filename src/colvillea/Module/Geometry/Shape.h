#pragma once

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include <map>

#include "../../Application/TWAssert.h"



/** 
 * @brief Shape class is the base class for all supported
 * primitive types in Colvillea. It should be able to
 * provide fundamental function to prepare geometry for
 * rendering on GPU.
 */
class Shape
{
public:
	/**
	 * @brief constructor for Shape class, collect all necessary
	 * parameters to initialize and set up GeometryInstance.
	 */
	Shape(optix::Context context, const std::map<std::string, optix::Program> &programsMap, const std::string &shapeClassName, optix::Material integrator, const int materialIndex) : 
        m_context(context), m_programsMap(programsMap), m_primitiveCount(-1), 
        m_materialIndex(materialIndex), m_integrator(integrator)
	{
		std::cout << "[Info] Derived class name from Shape is: " << shapeClassName << std::endl;

		/* Load boundingbox and intersection program */
        // todo:delegate to loadShape(), if not, it's not possible to construct Shape object at the moment!
		auto programItr = this->m_programsMap.find("BoundingBox_" + shapeClassName);
		TW_ASSERT(programItr != this->m_programsMap.end());
		this->m_boundingBoxProgram = programItr->second;

		programItr = this->m_programsMap.find("Intersect_" + shapeClassName);
		TW_ASSERT(programItr != this->m_programsMap.end());
		this->m_intersectionProgram = programItr->second;
	}

	/**
	 * @brief Setup shape properties and prepare for SceneGraph.
	 * @param integrator material node for specifying an integrator
	 * @param materialIndex index of material parameter stack
	 */
    virtual void initializeShape() = 0;

	/**
	 * @brief materialIndex setter
	 */
	void setMaterialIndex(const int materialIndex)
	{
		TW_ASSERT(materialIndex >= 0);

		this->m_materialIndex = materialIndex;
		this->updateMaterialParameter();
	}

	/**
	 * @brief materialIndex getter
	 */
	int getMaterialIndex() const
	{
		return this->m_materialIndex;
	}

	/**
	 * @brief Getter for geometryInstance. The const
	 * qualifier is a hint that it should not be 
	 * modified other place.
	 */
	const optix::GeometryInstance getGeometryInstance() const
	{
		return this->m_geometryInstance;
	}

    virtual float getSurfaceArea() const
    {//todo:turn into pure virtual.
        throw std::runtime_error("Derived shape class method getSurfaceArea() not implemented!");
    }

	/**
	 * @brief flip computed geometry normal(nGeometry) in Intersect()
	 * @note need to review, |reverseOrientation| is binded to geometry
	 * instance such that we could have multiple instance of the same
	 * geometry with various |reverseOrientation|.
	 */
	void flipGeometryNormal()
	{
		TW_ASSERT(this->m_geometryInstance);

		this->m_geometryInstance["reverseOrientation"]->setInt(1);
	}

    /**
     * @brief is current geometry instance's normal flipped?
     * @return 0 for not flipped, 1 for flipped normal.
     */
    //[[nodiscard]]
    int isFlippedGeometryNormal()/* const*/
    {
        TW_ASSERT(this->m_geometryInstance);

        return this->m_geometryInstance["reverseOrientation"]->getInt();
    }


protected:
	/**
	 * @brief Setup geometry for the shape, including creating
	 * geometry node, setting bounding box program and intersect
	 * program, setting appropriate parameters, etc.
	 *
	 * This is a part of Shape::loadShape().
	 */
	virtual void setupGeometry();

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
	 * This is a part of Shape::loadShape().
	 */
	virtual void setupGeometryInstance(optix::Material integrator);

private:
	

	/**
	 * @brief Setup material parameter by |m_materialIndex|.
	 * Note that this has nothing to do with Material node,
	 * a variable setting to GeometryInstance node will be
	 * used instead.
	 *
	 * This is a part of Shape::loadShape() and setter
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

	optix::GeometryInstance m_geometryInstance;
	optix::Geometry         m_geometry;
	optix::Program          m_boundingBoxProgram;
	optix::Program          m_intersectionProgram;
	unsigned int            m_primitiveCount;

    optix::Material m_integrator;
	int             m_materialIndex;
};