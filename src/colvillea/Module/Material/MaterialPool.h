#pragma once
#include <algorithm>
#include <memory>
#include <vector>
#include <map>

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "colvillea/Device/Toolkit/CommonStructs.h"
#include "colvillea/Application/TWAssert.h"
#include "colvillea/Application/GlobalDefs.h"
#include "colvillea/Module/Image/ImageLoader.h"

#include "colvillea/Module/Geometry/Shape.h"

class Application;
class MaterialPool;

/**
 * @brief BSDF is a simple wrapper of CommonStructs::ShaderParams
 * to provide a convenient interface to GUI widgets for supporting
 * interactive editing BSDF parameters.
 */
class BSDF : public IEditableObject
{
public:
    /**
     * @brief Factory method for creating BSDF instance with reflectance 
     * texture support.
     * 
     * @param[in] context
     * @param[in] materialPool
     * @param[in] shaderParams
     * @param[in] reflectanceTextureFilename Reflectance Texture filename
     * @param[in] reflectanceTS
     *
     * @return BSDF instance
     */
    static std::unique_ptr<BSDF> createBSDF(optix::Context &context, MaterialPool *materialPool, const CommonStructs::ShaderParams &shaderParams, const std::string &reflectanceTextureFilename, const optix::TextureSampler &reflectanceTS)
    {
        std::unique_ptr<BSDF> bsdf = std::make_unique<BSDF>(context, materialPool, shaderParams, reflectanceTextureFilename, reflectanceTS);
        return bsdf;
    }

    /**
     * @brief Factory method for creating BSDF instance without reflectance
     * texture support.
     *
     * @param[in] context
     * @param[in] materialPool
     * @param[in] shaderParams
     * 
     * @return BSDF instance
     */
    static std::unique_ptr<BSDF> createBSDF(optix::Context &context, MaterialPool *materialPool, const CommonStructs::ShaderParams &shaderParams)
    {
        std::unique_ptr<BSDF> bsdf = std::make_unique<BSDF>(context, materialPool, shaderParams);
        return bsdf;
    }


public:
    
    BSDF(optix::Context &context, MaterialPool *materialPool, const CommonStructs::ShaderParams &shaderParams, const std::string &reflectanceTextureFilename, const optix::TextureSampler &reflectanceTS) :
        BSDF(context, materialPool, shaderParams)
    {
        this->m_reflectanceTS = reflectanceTS;
        this->m_reflectanceTextureFilename = reflectanceTextureFilename;
        this->m_enableReflectanceTexture = true;
    }

    /**
     * @note This is also used as delegated constructor.
     */
    BSDF(optix::Context &context, MaterialPool *materialPool, const CommonStructs::ShaderParams &shaderParams) :
        m_context(context), m_materialPool(materialPool),
        IEditableObject("BSDF", IEditableObject::IEditableObjectType::BSDF),
        m_csShaderParams(shaderParams), m_enableReflectanceTexture(false) /* this could be overridden by BSDF ctor */
    {
        TW_ASSERT(materialPool);
        switch (shaderParams.bsdfType)
        {
        case CommonStructs::BSDFType::Lambert:
            this->setName("Lambert");
            break;
        case CommonStructs::BSDFType::RoughMetal:
            this->setName("RoughMetal");
            break;
        case CommonStructs::BSDFType::RoughDielectric:
            this->setName("RoughDielectric");
            break;
        case CommonStructs::BSDFType::SmoothGlass:
            this->setName("SmoothGlass");
            break;
        case CommonStructs::BSDFType::Plastic:
            this->setName("SmoothGlass");
            break;
        case CommonStructs::BSDFType::Emissive:
            this->setName("Emissive");
            break;
        case CommonStructs::BSDFType::SmoothMirror:
            this->setName("SmoothGlass");
            break;
        case CommonStructs::BSDFType::FrostedMetal:
            this->setName("FrostedMetal");
            break;
        }
    }

    /************************************************************************/
    /*                         Getters & Setters                            */
    /************************************************************************/

public:
    const CommonStructs::ShaderParams &getCommonStructsShaderParams() const
    {
        return this->m_csShaderParams;
    }

    CommonStructs::BSDFType getBSDFType() const
    {
        return this->m_csShaderParams.bsdfType;
    }

    inline void setBSDFType(CommonStructs::BSDFType bsdfType);

    optix::float4 getReflectance() const
    {
        return this->m_csShaderParams.Reflectance;
    }

    inline void setReflectance(const optix::float4 &reflectance);

    bool getEnableReflectanceTexture() const
    {
        return this->m_enableReflectanceTexture;
    }

    inline void setEnableReflectanceTexture(bool enable);

    std::string getReflectanceTextureFilename() const
    {
        return this->m_reflectanceTextureFilename;
    }

    inline void setReflectanceTextureFilename(const std::string &reflectanceTextureFilename);

    float getRoughness() const
    {
        return this->m_csShaderParams.alphax;
    }

    inline void setRoughness(float roughness);

    optix::float4 getSpecular() const
    {
        return this->m_csShaderParams.Specular;
    }

    inline void setSpecular(const optix::float4 &specular);

    optix::float4 getEta() const
    {
        return this->m_csShaderParams.FrCond_e;
    }

    inline void setEta(const optix::float4 &eta);

    optix::float4 getKappa() const
    {
        return this->m_csShaderParams.FrCond_k;
    }

    inline void setKappa(const optix::float4 &kappa);

    float getIOR() const
    {
        return this->m_csShaderParams.ior;
    }

    inline void setIOR(float IOR);

    std::shared_ptr<Shape> getShape() const
    {
        return this->m_shape;
    }

    /**
     * @brief Set BSDF's shape once creating a shape.
     * Never use on Quad shape that serves as an underlying
     * type of QuadLight. They share the same BSDF.
     *
     * @note It should be called exactly once.
     * @see SceneGraph::createTriangleMesh()
     * SceneGraph::createQuad()
     * MaterialPool::createQuadLight()
     */
    void setShape(const std::shared_ptr<Shape> &shape)
    {
        TW_ASSERT(!this->m_shape);
        this->m_shape = shape;
    }

private:
    optix::Context m_context;

    /// ShaderParams
    CommonStructs::ShaderParams m_csShaderParams;

    /// TextureSampler for reflectance texture
    optix::TextureSampler m_reflectanceTS;
    /// Filename of reflectance texture
    std::string m_reflectanceTextureFilename;
    /// Enable status of reflectance texture
    bool m_enableReflectanceTexture;

    MaterialPool *m_materialPool;

    /// Shape that owns this BSDF
    std::shared_ptr<Shape> m_shape;
};

/**
 * @brief A pool used for creating, containing and
 * managing materials.
 * @note First material is always an emissive material served as 
         underlying BSDF for Area Light.
 */
class MaterialPool
{
public:
    /**
     * @brief Factory method for creating MaterialPool instance.
     * 
     * @param[in] application
     * @param[in] programsMap
     * @param[in] context
     */
    static std::shared_ptr<MaterialPool> createMaterialPool(Application *application, const std::map<std::string, optix::Program> &programsMap, const optix::Context context);

	MaterialPool(const std::map<std::string, optix::Program> &programsMap, const optix::Context context):m_programsMap(programsMap),m_context(context)
	{
		/* Add a default Emissive BSDF and setup shaderBuffer. */
		this->m_shaderBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 1);
		this->m_shaderBuffer->setElementSize(sizeof(CommonStructs::ShaderParams));

        /* First material is always an emissive material served as
         * underlying BSDF for Area Light. */
        TW_ASSERT(createEmissiveMaterial(this->m_emissiveBSDF) == 0);

        this->updateAllMaterials(false);
	}

    /**
     * @brief Try to get an emissive material. If there is no
     * emissive material available, create one and return corresponding
     * index.
     * @return Return the corresponding materialIndex pointed to materialBuffer.
     */
    int getEmissiveMaterial(std::shared_ptr<BSDF> &outBSDF) const 
    {
        TW_ASSERT(this->m_bsdfs.size() > 0);
        /* First material is always an emissive material served as
         * underlying BSDF for Area Light. */
        outBSDF = this->m_emissiveBSDF;
        return 0;
    }

#pragma region CodeFromSceneGraph
private:
    /**
     * @note Use MaterialPool::getEmissiveMaterial() to
     * fetch a Emissive BSDF. This is for internal usage.
     */
    inline int createEmissiveMaterial(std::shared_ptr<BSDF> &outBSDF)
    {//todo,review
        CommonStructs::ShaderParams materialParams;
        MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::Emissive, optix::make_float4(1.f, 1.f, 1.f, 1.f));
        outBSDF = BSDF::createBSDF(this->m_context, this, materialParams);

        return this->addMaterial(outBSDF);
    }
    
public:
    /**
	 * @brief create a lambert material without texture support for trianglemesh
	 * @param reflectance diffuse reflectance
	 * @return return the material index to materialBuffer
	 */
	inline int createLambertMaterial(const optix::float4 & reflectance, std::shared_ptr<BSDF> &outBSDF)
	{
		CommonStructs::ShaderParams materialParams;
		MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::Lambert,
			reflectance);
        outBSDF = BSDF::createBSDF(this->m_context, this, materialParams);
		return this->addMaterial(outBSDF);
	}
	/**
	 * @brief create a lambert material with texture support for trianglemesh
	 * @param[in] reflectanceTextureFile  diffuse reflectance
	 * @return return the created shaderparams object
	 */
	inline int createLambertMaterial(const std::string &reflectanceTextureFile, std::shared_ptr<BSDF> &outBSDF)
	{
        optix::TextureSampler reflectanceTS = ImageLoader::LoadImageTexture(this->m_context, reflectanceTextureFile, optix::make_float4(0.f));
        int reflectanceMapId = reflectanceTS->getId();

		CommonStructs::ShaderParams materialParams;
		MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::Lambert,
            optix::make_float4(0.f), reflectanceMapId);
        outBSDF = BSDF::createBSDF(this->m_context, this, materialParams, reflectanceTextureFile, reflectanceTS);
		return this->addMaterial(outBSDF);
	}

	/**
	 * @brief create a perfect smooth glass material for trianglemesh
	 * @param ior index of refraction of the glass
	 * @return return the created shaderparams object
	 */
	inline int createSmoothGlassMaterial(const float ior, std::shared_ptr<BSDF> &outBSDF)
	{
		CommonStructs::ShaderParams materialParams;
		MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::SmoothGlass);
		materialParams.ior = ior;
        outBSDF = BSDF::createBSDF(this->m_context, this, materialParams);
		return this->addMaterial(outBSDF);
	}
	/**
	 * @brief create a perfect smooth mirror material for trianglemesh(Fresnel=1)
	 * @return return the created shaderparams object
	 */
	inline int createSmoothMirrorMaterial(std::shared_ptr<BSDF> &outBSDF)
	{
		CommonStructs::ShaderParams materialParams;
		MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::SmoothMirror);
        outBSDF = BSDF::createBSDF(this->m_context, this, materialParams);
		return this->addMaterial(outBSDF);
	}

	/**
	 * @brief create a rough metal material for trianglemesh
	 * @param roughness roughness of rough metal
	 * @param eta eta(index of refraction) of rough metal, with wavelength dependent component
	 * @param k absorption coefficient of rough metal, with wavelength dependent component
	 * @return return the created shaderparams object
	 */
	inline int createRoughMetalMaterial(std::shared_ptr<BSDF> &outBSDF, const float roughness, const optix::float4 &eta, const optix::float4 &k, const optix::float4 & specular = optix::make_float4(1.f))
	{
		CommonStructs::ShaderParams materialParams;
		MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::RoughMetal);

		materialParams.alphax = roughness;
		materialParams.FrCond_e = eta;
		materialParams.FrCond_k = k;
		materialParams.Specular = specular;
        outBSDF = BSDF::createBSDF(this->m_context, this, materialParams);
		return this->addMaterial(outBSDF);
	}

	/**
	 * @brief create a rough plastic material for trianglemesh
	 * @param roughness roughness of rough plastic
	 * @param ior the ior of the rough dielectric coated on the buttom lambertian surface
	 * @param reflectance the reflectance of lambertian surface lying on the buttom of the plastic
	 * @return return the created shaderparams object
	 */
	inline int createPlasticMaterial(const float roughness, const float ior, const optix::float4& reflectance, const optix::float4 &specular, std::shared_ptr<BSDF> &outBSDF)
	{
		CommonStructs::ShaderParams materialParams;
		MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::Plastic);

		materialParams.Reflectance = reflectance;
		materialParams.ior = ior;
		materialParams.alphax = roughness;
		materialParams.Specular = specular;
        outBSDF = BSDF::createBSDF(this->m_context, this, materialParams);
		return this->addMaterial(outBSDF);
	}
	/**
	 * @brief create a rough plastic material for trianglemesh with diffuse texture support
	 * @param[in] roughness              roughness of rough plastic
	 * @param[in] ior                    the ior of the rough dielectric coated on the buttom lambertian surface
	 * @param[in] reflectanceTextureFile the reflectance of lambertian surface lying on the buttom of the plastic
	 * @param[in] specular
	 * @return return the created shaderparams object
	 */
	inline int createPlasticMaterial(const float roughness, const float ior, const std::string &reflectanceTextureFile, const optix::float4 &specular, std::shared_ptr<BSDF> &outBSDF)
	{
        optix::TextureSampler reflectanceTS = ImageLoader::LoadImageTexture(this->m_context, reflectanceTextureFile, optix::make_float4(0.f));
        int reflectanceMapId = reflectanceTS->getId();

		CommonStructs::ShaderParams materialParams;
		MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::Plastic);

		materialParams.ReflectanceID = reflectanceMapId;
		materialParams.ior = ior;
		materialParams.alphax = roughness;
		materialParams.Specular = specular;
        outBSDF = BSDF::createBSDF(this->m_context, this, materialParams, reflectanceTextureFile, reflectanceTS);
		return this->addMaterial(outBSDF);
	}

	/**
	 * @brief create a rough dielectric material for trianglemesh(lacking of transmittance support)
	 * @param roughness roughness of rough dielectric
	 * @param ior the ior of the rough dielectric
	 * @param reflectance the reflectance of rough dielectric
	 * @return return the created shaderparams object
	 */
	inline int createRoughDielectricMaterial(std::shared_ptr<BSDF> &outBSDF, const float roughness, const float ior, const optix::float4& reflectance, const optix::float4& specular = optix::make_float4(1.f))
	{
		CommonStructs::ShaderParams materialParams;
		MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::RoughDielectric);

		materialParams.Reflectance = reflectance;
		materialParams.ior = ior;
		materialParams.alphax = roughness;
		materialParams.Specular = specular;
        outBSDF = BSDF::createBSDF(this->m_context, this, materialParams);
		return this->addMaterial(outBSDF);
	}

    //todo:improv
    inline int createFrostedMetalMaterial(std::shared_ptr<BSDF> &outBSDF)
    {
        CommonStructs::ShaderParams materialParams;
        MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::FrostedMetal);
        this->m_context["eta1"]->setFloat(optix::make_float3(1.f));
        this->m_context["kappa1"]->setFloat(optix::make_float3(1.f, 0.1f, 0.1f));
        outBSDF = BSDF::createBSDF(this->m_context, this, materialParams);
        return this->addMaterial(outBSDF);
    }
#pragma endregion CodeFromSceneGraph_createXXMaterial()

    /**
     * @brief Remove a BSDF.
     * @note It just removes a BSDF from |m_bsdfs| and don't ensure 
     * the removal is a "legal" operation. In other words, if the BSDF
     * is removed when there is still at least one Shape referencing it,
     * it could possibly crash.
     *       Also notice that BSDF could be shared by mulitple shapes 
     * -- in hard code (but not in GUI).
     * @todo This operation should be part of remove shape when possible.
     */
    void removeBSDF(const std::shared_ptr<BSDF> &bsdf)
    {
        auto itrToErase = std::find_if(this->m_bsdfs.cbegin(), this->m_bsdfs.cend(),
            [&bsdf](const auto& curBSDF)
        {
            return curBSDF->getId() == bsdf->getId();
        });

        TW_ASSERT(itrToErase != this->m_bsdfs.end());

        /* Remove BSDF. */
        this->m_bsdfs.erase(itrToErase);

        /* Update Materials. */
        this->updateAllMaterials(true);
    }


    /************************************************************************/
    /*                         Getters & Setters                            */
    /************************************************************************/
public:
    /**
     * @brief Get |m_bsdfs|.
     */
    const std::vector<std::shared_ptr<BSDF>> &getBSDFs() const
    {
        return this->m_bsdfs;
    }

    /**
     * @brief Get BSDF given |materialIndex|.
     * @param[in] materialIndex materialIndex from Shape
     * @return BSDF instance
     */
    const std::shared_ptr<BSDF> &getBSDF(int materialIndex) const
    {
        return this->m_bsdfs[materialIndex];
    }

    /**
     * @brief Convert comboBSDFType to CommonStructs::BSDFType.
     */
    static CommonStructs::BSDFType comboBSDFTypeToCommonStructsBSDFType(int comboBSDFType)
    {
        /* Note that comboBSDFType is not exactly the same
         * -- as CommonStructs::BSDFType because Emissive BSDF
         * -- is not shown in Combo. */
        if (comboBSDFType <= comboBSDFType_Plastic)
            return static_cast<CommonStructs::BSDFType>(comboBSDFType);
        else
            return static_cast<CommonStructs::BSDFType>(comboBSDFType + 1);
    }

    /**
     * @brief Convert CommonStructs::BSDFType to comboBSDFType.
     */
    static int CommonStructsBSDFTypeToComboBSDFType(CommonStructs::BSDFType bsdfType)
    {
        int bsdfTypeIndex = toUnderlyingValue(bsdfType);
        if (bsdfTypeIndex <= comboBSDFType_Plastic)
            return bsdfTypeIndex;
        else
            return bsdfTypeIndex - 1;
    }

private:
	/**
	 * @brief Helper function for specifying shaderParams.
	 * Todo:should be put into ctor of CommonStructs::ShaderParams?
	 * 
	 * @note Note that here we give some default values to avoid NaN complaints
	 * when changing BSDF to a new type.
	 */
	 void settingMaterialParameters(CommonStructs::ShaderParams &shaderParams, CommonStructs::BSDFType bsdfType,
         optix::float4 reflectance = optix::make_float4(1.f), int reflectanceId = RT_TEXTURE_ID_NULL,
         optix::float4 specular = optix::make_float4(1.f), int specularId = RT_TEXTURE_ID_NULL,
		float alphax = 0.005f, int roughnessId = RT_TEXTURE_ID_NULL,
         optix::float4 eta = optix::make_float4((1.66f, 0.95151f, 0.7115f, 0.f)), optix::float4 kappa = optix::make_float4(8.0406f, 6.3585f, 5.1380f, 0.f),
		float ior = 1.5f) const
	{
		shaderParams.bsdfType = bsdfType;
		shaderParams.Reflectance = reflectance;
		shaderParams.ReflectanceID = reflectanceId;
		shaderParams.Specular = specular;
		shaderParams.SpecularID = specularId;
		shaderParams.alphax = alphax;
		shaderParams.RoughnessID = roughnessId;
		shaderParams.FrCond_e = eta;
		shaderParams.FrCond_k = kappa;
		shaderParams.ior = ior;
	}

	/**
	 * @brief Add a material to materialBuffer and send to OptiX.
	 * @return Return the corresponding materialIndex pointed to materialBuffer.
	 */
	int addMaterial(const std::shared_ptr<BSDF> &bsdf)
	{
		this->m_bsdfs.push_back(bsdf);
        int index = this->m_bsdfs.size() - 1;

        /* Update all materials. */
        this->updateAllMaterials(false);

		return index;
	}

private:
    /************************************************************************/
    /*                             Update functions                         */
    /************************************************************************/

    /**
     * @brief Update all materials. This is applicable for all modification operations to
     * MaterialBuffer, adding, modifying and removing.
     *
     * @todo Rewrite addMaterial() and this function to support update one
     * single material a time.
     *       -- add "bool resizeBuffer" to avoid unnecessary resizing.
     */
    void updateAllMaterials(bool updateMaterialIndex)
    {
        TW_ASSERT(this->m_shaderBuffer);
        /* At least an Emissive BSDF is stored. */
        TW_ASSERT(this->m_bsdfs.size() > 0);

        this->m_shaderBuffer->setSize(this->m_bsdfs.size());
        std::cout << "[Info] Updated ShaderBuffer." << std::endl;

        /* Setup shaderBuffer for GPU Program */
        auto shaderParamsArray = static_cast<CommonStructs::ShaderParams *>(this->m_shaderBuffer->map());
        TW_ASSERT(this->m_bsdfs.size() >= 1);
        for (auto itr = this->m_bsdfs.cbegin(); itr != this->m_bsdfs.cend(); ++itr)
        {
            if (updateMaterialIndex)
            {
                /* For the Emissive BSDF's referencing shape, no need to setMaterialIndex. */
                if (itr != this->m_bsdfs.cbegin())
                {
                    (*itr)->getShape()->setMaterialIndex(itr - this->m_bsdfs.cbegin());
                }
            }
                
            shaderParamsArray[itr - this->m_bsdfs.cbegin()] = (*itr)->getCommonStructsShaderParams();
        }

        this->m_context["shaderBuffer"]->setBuffer(this->m_shaderBuffer);

        /* Unmap buffer. */
        this->m_shaderBuffer->unmap();
    }


public:
    /// Constants for combo label, basically corresponding to CommonStructs::BSDFType except for Emissive
    static constexpr int comboBSDFType_Lambert = 0;
    static constexpr int comboBSDFType_RoughMetal = 1;
    static constexpr int comboBSDFType_RoughDielectric = 2;
    static constexpr int comboBSDFType_SmoothGlass = 3;
    static constexpr int comboBSDFType_Plastic = 4;
    /*static constexpr int comboBSDFType_Emissive = 5;*/
    static constexpr int comboBSDFType_SmoothMirror = 5;
    static constexpr int comboBSDFType_FrostedMetal = 6;

private:
    const std::map<std::string, optix::Program> &m_programsMap;
    optix::Context     m_context;

    /// ShaderBuffer
	optix::Buffer m_shaderBuffer;
    /// MaterialBuffer contains CommonStructs::ShaderParams
    std::vector<std::shared_ptr<BSDF>> m_bsdfs;

    /// Emissive BSDF
    std::shared_ptr<BSDF> m_emissiveBSDF;

    friend class BSDF;
};

void BSDF::setEnableReflectanceTexture(bool enable)
{
    this->m_enableReflectanceTexture = enable;

    if (!this->m_reflectanceTS)
    {
        /* Reflectance TS is dummy. */
        TW_ASSERT(this->m_csShaderParams.ReflectanceID == RT_TEXTURE_ID_NULL);
        std::cout << "[Info] Attempt to" << (enable ? "Enable" : "Disable") << " Reflectance Texture without specifying Reflectance Texture for current BSDF. Please try loading a Reflectance Texture first." << std::endl;
        return;
    }

    /* Setup device variable to enable/disable Reflectance Texture. */
    this->m_csShaderParams.ReflectanceID = enable ? this->m_reflectanceTS->getId() : RT_TEXTURE_ID_NULL;

    /* Update all materials. */
    this->m_materialPool->updateAllMaterials(false);

    std::cout << "[Info] " << (enable ? "Enable" : "Disable") << " Reflectance Texture." << std::endl;
}

void BSDF::setReflectanceTextureFilename(const std::string &reflectanceTextureFilename)
{
    if (!this->m_reflectanceTS)
    {
        /* Reflectance TS is dummy. After loading texture, it
         * -- is enabled automatically. */
        this->m_enableReflectanceTexture = true;
    }

    this->m_reflectanceTextureFilename = reflectanceTextureFilename;
    /* todo: Note that this could lead to memory leaking because it's creating rtBuffer
     * -- each time we want to change a HDRI file but never call *rtRemove... like function
     * -- to destroy device buffers (Optixpp might not be a RAII-style wrapper...). */
    this->m_reflectanceTS = ImageLoader::LoadImageTexture(this->m_context, reflectanceTextureFilename, optix::make_float4(0.f), false);

    this->m_csShaderParams.ReflectanceID = this->m_reflectanceTS->getId();

    this->m_materialPool->updateAllMaterials(false);

    std::cout << "[Info] " << "BSDF:" << this->getName() << " updates Reflectance Texture to " << reflectanceTextureFilename << std::endl;
}

void BSDF::setRoughness(float roughness)
{
    this->m_csShaderParams.alphax = roughness;
    this->m_materialPool->updateAllMaterials(false);
}

void BSDF::setSpecular(const optix::float4 &specular)
{
    this->m_csShaderParams.Specular = specular;
    this->m_materialPool->updateAllMaterials(false);
}

void BSDF::setEta(const optix::float4 &eta)
{
    this->m_csShaderParams.FrCond_e = eta;
    this->m_materialPool->updateAllMaterials(false);
}

void BSDF::setKappa(const optix::float4 &kappa)
{
    this->m_csShaderParams.FrCond_k = kappa;
    this->m_materialPool->updateAllMaterials(false);
}

void BSDF::setIOR(float IOR)
{
    this->m_csShaderParams.ior = IOR;
    this->m_materialPool->updateAllMaterials(false);
}

void BSDF::setBSDFType(CommonStructs::BSDFType bsdfType)
{
    this->m_csShaderParams.bsdfType = bsdfType;
    this->m_materialPool->updateAllMaterials(false);
}

void BSDF::setReflectance(const optix::float4 &reflectance)
{
    this->m_csShaderParams.Reflectance = reflectance;
    this->m_materialPool->updateAllMaterials(false);
}