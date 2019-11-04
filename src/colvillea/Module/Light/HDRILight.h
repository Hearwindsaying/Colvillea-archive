#pragma once
#include "Light.h"

#include <map>

#include "colvillea/Module/Image/ImageLoader.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"
#include "colvillea/Device/Toolkit/Utility.h"
#include "colvillea/Application/GlobalDefs.h"

class LightPool;

/**
 * @brief HDRILight describing an infinite area
 * light illuminated by High-Dynamic-Range Image
 * (HDRI). It's also known as Image Based Lighting.
 */
class HDRILight : public Light 
{
public:
    /**
     * @brief Factory method for creating a HDRILight instance.
     *
     * @param[in] application
     * @param[in] context
     * @param[in] programsMap  map to store Programs
     * @param[in] hdriFilename HDRI filename
     * @param[in] rotation     rotation radian angles in X,Y,Z
     */
    static std::unique_ptr<HDRILight> createHDRILight(Application *application, optix::Context context, const std::map<std::string, optix::Program> &programsMap, const std::string & hdriFilename, const optix::float3 &rotation, /*std::shared_ptr<LightPool>*/LightPool * lightPool)
    {
        std::unique_ptr<HDRILight> hdriLight = std::make_unique<HDRILight>(application, context, programsMap, hdriFilename, lightPool, rotation);
        hdriLight->initializeLight();
        return hdriLight;
    }

    /**
     * @brief Factory method for creating a dummy HDRILight instance.
     */
    static std::unique_ptr<HDRILight> createHDRILight(Application *application, optix::Context context, const std::map<std::string, optix::Program> &programsMap, LightPool * lightPool)
    {
       return std::make_unique<HDRILight>(application, context, programsMap, lightPool);
    }

    HDRILight(Application *application, optix::Context context, const std::map<std::string, optix::Program> &programsMap, const std::string & hdriFilename, /*std::shared_ptr<LightPool>*/LightPool * lightPool, const optix::float3 &rotation);

    HDRILight(Application *application, optix::Context context, const std::map<std::string, optix::Program> &programsMap, LightPool * lightPool);

    void initializeLight(); //override this
    void initializeLight(const optix::Matrix4x4 &lightToWorld) { __debugbreak(); }

    const CommonStructs::HDRILight &getCommonStructsLight() const
    {
        return this->m_csHDRILight;
    }

    /**
     * @brief Set status of Enable/Disable HDRILight.
     * @param[in]  enable Enable/Disable HDRILight
     */
    void setEnableHDRILight(bool enable);
    
    
    /**
     * @brief Get status of Enable/Disable HDRILight.
     * @return enable/disable status
     */
    bool getEnableHDRILight() const
    {
        return this->m_enable;
    }

    /**
     * @brief Setter for |m_rotation|.
     */
    void setLightRotation(const optix::float3 &rotation);

    /**
     * @brief Getter for |m_rotation|.
     */
    optix::float3 getLightRotation() const
    {
        return this->m_rotationRad;
    }

    /**
     * @brief Getter for |m_HDRIFilename|.
     */
    std::string getHDRIFilename() const
    {
        return this->m_HDRIFilename;
    }

    /**
     * @brief Setter for change a HDRIFile.
     */
    void setHDRIFilename(const std::string &HDRIFilename)
    {
        /* Load HDRI texture. */
        this->m_HDRIFilename = HDRIFilename;
        /* todo: Note that this could lead to memory leaking because it's creating rtBuffer 
         * -- each time we want to change a HDRI file but never call *rtRemove... like function
         * -- to destroy device buffers (Optixpp might not a RAII-style wrapper...). */
        this->m_HDRITextureSampler = ImageLoader::LoadImageTexture(this->m_context, HDRIFilename, optix::make_float4(0.f), false);

        this->m_csHDRILight.hdriEnvmap = this->m_HDRITextureSampler->getId();

        /* HDRILight Struct setup can't be done until finish HDRILight::preprocess(). */
        this->preprocess();

        std::cout << "[Info] " << "Update HDRI file to " << HDRIFilename << std::endl;
    }
    
 
protected:
    /**
     * @brief Preprocessing for HDRILight. Besides fundamental
     * properties describing HDRILight, some precomputation
     * needed to be done before launching to render. Meanwhile,
     * prefiltering HDRI texture for precomputation accelerated
     * by OptiX using another launch is necessary before that.
     * 
     * @note This function will invoke context->launch() which
     * needs an implicit context->validate(). In other words,
     * it is able to continue launch() after all other status
     * for OptiX are set. Consequently, HDRILight's creation
     * should be in the last.
     */
    void preprocess() override;

private:
    /// We use a weak_ptr to hold LightPool instance, note that
    /// -- LightPool owns HDRILight while HDRILight needs to use
    /// LightPool to update LightBuffers.
    /*std::weak_ptr<LightPool>*/LightPool *m_lightPool;

    optix::TextureSampler m_HDRITextureSampler;
    /// HDRI Filename (host only)
    std::string           m_HDRIFilename;

    /// Storage for struct data used in GPU programs
    CommonStructs::HDRILight m_csHDRILight;    

    /// Record status of Enable/Disable HDRILight (host only)
    bool m_enable;

    /// Record user-friendly transform elements: Rotation angle in radian (host only)
    optix::float3 m_rotationRad;
};