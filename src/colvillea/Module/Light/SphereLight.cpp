// Include Eigen before STL and after OptiX (for macro support like __host__)
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix.h>

// #include "colvillea/Device/SH/Test/SH.hpp"

// STL include
#include <random>
#include <utility>
#include <cstdio>
#include <fstream>
#include <limits>
#include <algorithm>

#include "colvillea/Module/Light/SphereLight.h"

#include "colvillea/Application/Application.h" // For LightPool::createLightPool()
#include "colvillea/Module/Light/LightPool.h"

using namespace optix;

void SphereLight::setLightIntensity(float intensity)
{
    this->m_intensity = intensity;
    optix::float3 csIntensity = this->m_intensity * this->m_color;
    this->m_csSphereLight.intensity = optix::make_float4(csIntensity.x, csIntensity.y, csIntensity.z, 1.0f);

    this->m_lightPool->updateAllSphereLights(false);
}

void SphereLight::setLightColor(const optix::float3 & color)
{
    this->m_color = color;
    optix::float3 csIntensity = this->m_intensity * this->m_color;
    this->m_csSphereLight.intensity = optix::make_float4(csIntensity.x, csIntensity.y, csIntensity.z, 1.0f);

    this->m_lightPool->updateAllSphereLights(false);
}

void SphereLight::setPosition(const optix::float3 & position)
{
    /* Update underlying Sphere shape. */
    this->m_sphereShape->setPosition(position);

    /* Update SphereLight Struct for GPU program. */
    this->updateMatrixParameter();
    this->m_csSphereLight.center = position;

    //update all SphereLights
    this->m_lightPool->updateAllSphereLights(true);
}

void SphereLight::setRadius(const float radius)
{
    /* Update underlying Sphere shape. */
    this->m_sphereShape->setRadius(radius);

    /* Update SphereLight Struct for GPU program. */
    this->updateMatrixParameter();
    this->m_csSphereLight.radius = radius;

    //update all SphereLights
    this->m_lightPool->updateAllSphereLights(true);
}