#pragma once

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

/* Global definiations and helper functions used in Colvillea.*/

/** Scoped enum specifying RayGeneration entry.
 *  Use CountOfType to get number of entries and 
 *  cast to unsigned int using toUnderlyingValue.
 */
enum class RayGenerationEntryType : unsigned int
{
	Render,     // Main entry for rendering

	HDRI,       // Image Based Lighting precomputation entry

	InitFilter, // Progressive filtering initialization entry
	Filter,     // Progressive filtering filter entry

	CountOfType
};

enum class IntegratorType : unsigned int
{
    DirectLighting, // Direct light only.
    PathTracing,    // Direct light + indirect light.

    CountOfType
};





/**
 * @brief Helper template function to convert scoped
 * enum type to its underlying value for access.
 */
template<typename E>
static __device__ __host__ __inline__ constexpr auto toUnderlyingValue(E enumerator) noexcept
{
	return static_cast<std::underlying_type_t<E>>(enumerator);
}


/**
 * @brief Interface for editable object, including lights and
 * geometries.
 * @todo Re-design a better API for showing GUI widgets. It
 * shouldn't hold information about what type of object it
 * stores actually.
 */
class IEditableObject
{
public:
    /**
     * @brief The actual type of object stores in IEditableObject.
     * @todo Discard this enum class design, it's quite awkward.
     */
    enum class IEditableObjectType
    {
        HDRILight,
        PointLight,
        QuadLight,

        QuadGeometry,
        TriangleMeshGeometry,

        BSDF
    };

protected:
    IEditableObject(const std::string & name, IEditableObjectType objectType) : m_name(name), m_objectType(objectType)
    {
        this->m_id = IEditableObject::IEditableObjectId++;
    }
public:
    std::string getName() const
    {
        return this->m_name;
    }
    
    void setName(const std::string &name)
    {
        this->m_name = name;
    }

    int64_t getId()
    {
        return this->m_id;
    }

    IEditableObjectType getObjectType() const
    {
        return this->m_objectType;
    }
private:
    /// Displayed name.
    std::string m_name;
    /// Unique Identifier for widget.
    int64_t m_id;
    /// Actual type of object stored.
    IEditableObjectType m_objectType;

    /// Holding Id for next widget.
    static int64_t IEditableObjectId;
};

