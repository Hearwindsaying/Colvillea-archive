#pragma once

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


