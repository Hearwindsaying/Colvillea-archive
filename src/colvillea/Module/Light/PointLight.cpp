#include "colvillea/Module/Light/PointLight.h"

#include "colvillea/Application/Application.h" // For LightPool::createLightPool()
#include "colvillea/Module/Light/LightPool.h"

void PointLight::setLightPosition(const optix::float3 & lightPosition)
{
    this->m_csPointLight.lightPos = lightPosition;

    this->m_lightPool->updateAllPointLights();
}

void PointLight::setLightIntensity(float intensity)
{
    this->m_intensity = intensity;
    optix::float3 csIntensity = this->m_intensity * this->m_color;
    this->m_csPointLight.intensity = optix::make_float4(csIntensity.x, csIntensity.y, csIntensity.z, 1.0f);

    this->m_lightPool->updateAllPointLights();
}

void PointLight::setLightColor(const optix::float3 &color)
{
    this->m_color = color;
    optix::float3 csIntensity = this->m_intensity * this->m_color;
    this->m_csPointLight.intensity = optix::make_float4(csIntensity.x, csIntensity.y, csIntensity.z, 1.0f);

    this->m_lightPool->updateAllPointLights();
}