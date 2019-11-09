#pragma once
#include <vector>

#include <optix_world.h>
#include <optix_host.h>
#include <optixu_math_namespace.h>

#include "colvillea/Device/Toolkit/CommonStructs.h"
#include "colvillea/Application/TWAssert.h"

class Application;


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
    static std::shared_ptr<MaterialPool> createMaterialPool(Application *application, const std::map<std::string, optix::Program> &programsMap, const optix::Context context)
    {
        std::shared_ptr<MaterialPool> materialPool = std::make_shared<MaterialPool>(programsMap, context);
        application->m_materialPool = materialPool;
        return materialPool;
    }

	MaterialPool(const std::map<std::string, optix::Program> &programsMap, const optix::Context context):m_programsMap(programsMap),m_context(context)
	{
		/* Add a default Emissive BSDF and setup shaderBuffer. */
		this->m_shaderBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 1);
		this->m_shaderBuffer->setElementSize(sizeof(CommonStructs::ShaderParams));

        /* First material is always an emissive material served as
         * underlying BSDF for Area Light. */
        TW_ASSERT(createEmissiveMaterial() == 0);

        this->updateAllMaterials();
	}

    /**
     * @brief Try to get an emissive material. If there is no
     * emissive material available, create one and return corresponding
     * index.
     * @return Return the corresponding materialIndex pointed to materialBuffer.
     */
    int getEmissiveMaterial() const 
    {
        TW_ASSERT(this->m_materialBuffer.size() > 0);
        /* First material is always an emissive material served as
         * underlying BSDF for Area Light. */
        return 0;
    }

#pragma region CodeFromSceneGraph
    inline int createEmissiveMaterial()
    {//todo,review
        CommonStructs::ShaderParams materialParams;
        MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::Emissive, optix::make_float4(1.f, 1.f, 1.f, 1.f));

        return this->addMaterial(materialParams);
    }
    
    /**
	 * @brief create a lambert material without texture support for trianglemesh
	 * @param reflectance diffuse reflectance
	 * @return return the material index to materialBuffer
	 */
	inline int createLambertMaterial(const optix::float4 & reflectance)
	{
		CommonStructs::ShaderParams materialParams;
		MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::Lambert,
			reflectance);

		return this->addMaterial(materialParams);
	}
	/**
	 * @brief create a lambert material with texture support for trianglemesh
	 * @param reflectance diffuse reflectance
	 * @return return the created shaderparams object
	 */
	inline int createLambertMaterial(const int &reflectanceMapId)
	{
		CommonStructs::ShaderParams materialParams;
		MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::Lambert,
            optix::make_float4(0.f),reflectanceMapId);

		return this->addMaterial(materialParams);
	}

	/**
	 * @brief create a perfect smooth glass material for trianglemesh
	 * @param ior index of refraction of the glass
	 * @return return the created shaderparams object
	 */
	inline int createSmoothGlassMaterial(const float ior)
	{
		CommonStructs::ShaderParams materialParams;
		MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::SmoothGlass);
		materialParams.ior = ior;

		return this->addMaterial(materialParams);
	}
	/**
	 * @brief create a perfect smooth mirror material for trianglemesh(Fresnel=1)
	 * @return return the created shaderparams object
	 */
	inline int createSmoothMirrorMaterial()
	{
		CommonStructs::ShaderParams materialParams;
		MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::SmoothMirror);

		return this->addMaterial(materialParams);
	}

	/**
	 * @brief create a rough metal material for trianglemesh
	 * @param roughness roughness of rough metal
	 * @param eta eta(index of refraction) of rough metal, with wavelength dependent component
	 * @param k absorption coefficient of rough metal, with wavelength dependent component
	 * @return return the created shaderparams object
	 */
	inline int createRoughMetalMaterial(const float roughness, const optix::float4 &eta, const optix::float4 &k, const optix::float4 & specular = optix::make_float4(1.f))
	{
		CommonStructs::ShaderParams materialParams;
		MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::RoughMetal);

		materialParams.alphax = roughness;
		materialParams.FrCond_e = eta;
		materialParams.FrCond_k = k;
		materialParams.Specular = specular;

		return this->addMaterial(materialParams);
	}

	/**
	 * @brief create a rough plastic material for trianglemesh
	 * @param roughness roughness of rough plastic
	 * @param ior the ior of the rough dielectric coated on the buttom lambertian surface
	 * @param reflectance the reflectance of lambertian surface lying on the buttom of the plastic
	 * @return return the created shaderparams object
	 */
	inline int createPlasticMaterial(const float roughness, const float ior, const optix::float4& reflectance, const optix::float4 &specular)
	{
		CommonStructs::ShaderParams materialParams;
		MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::Plastic);

		materialParams.Reflectance = reflectance;
		materialParams.ior = ior;
		materialParams.alphax = roughness;
		materialParams.Specular = specular;

		return this->addMaterial(materialParams);
	}
	/**
	 * @brief create a rough plastic material for trianglemesh with diffuse texture support
	 * @param roughness roughness of rough plastic
	 * @param ior the ior of the rough dielectric coated on the buttom lambertian surface
	 * @param reflectance the reflectance of lambertian surface lying on the buttom of the plastic
	 * @return return the created shaderparams object
	 */
	inline int createPlasticMaterial(const float roughness, const float ior, const int& reflectanceMapId, const optix::float4 &specular)
	{
		CommonStructs::ShaderParams materialParams;
		MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::Plastic);

		materialParams.ReflectanceID = reflectanceMapId;
		materialParams.ior = ior;
		materialParams.alphax = roughness;
		materialParams.Specular = specular;

		return this->addMaterial(materialParams);
	}

	/**
	 * @brief create a rough dielectric material for trianglemesh(lacking of transmittance support)
	 * @param roughness roughness of rough dielectric
	 * @param ior the ior of the rough dielectric
	 * @param reflectance the reflectance of rough dielectric
	 * @return return the created shaderparams object
	 */
	inline int createRoughDielectricMaterial(const float roughness, const float ior, const optix::float4& reflectance, const optix::float4& specular = optix::make_float4(1.f))
	{
		CommonStructs::ShaderParams materialParams;
		MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::RoughDielectric);

		materialParams.Reflectance = reflectance;
		materialParams.ior = ior;
		materialParams.alphax = roughness;
		materialParams.Specular = specular;

		return this->addMaterial(materialParams);
	}

    //todo:improv
    inline int createFrostedMetalMaterial()
    {
        CommonStructs::ShaderParams materialParams;
        MaterialPool::settingMaterialParameters(materialParams, CommonStructs::BSDFType::FrostedMetal);
        this->m_context["eta1"]->setFloat(optix::make_float3(1.f));
        this->m_context["kappa1"]->setFloat(optix::make_float3(1.f, 0.1f, 0.1f));

        return this->addMaterial(materialParams);
    }
#pragma endregion CodeFromSceneGraph_createXXMaterial()

private:
	/**
	 * @brief Helper function for specifying shaderParams.
	 * Todo:should be put into ctor of CommonStructs::ShaderParams?
	 */
	 void settingMaterialParameters(CommonStructs::ShaderParams &shaderParams, CommonStructs::BSDFType bsdfType,
         optix::float4 reflectance = optix::make_float4(0.f), int reflectanceId = RT_TEXTURE_ID_NULL,
         optix::float4 specular = optix::make_float4(0.f), int specularId = RT_TEXTURE_ID_NULL,
		float alphax = 0.f, int roughnessId = RT_TEXTURE_ID_NULL,
         optix::float4 eta = optix::make_float4(0.f), optix::float4 kappa = optix::make_float4(0.f),
		float ior = 0.f) const
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
	int addMaterial(CommonStructs::ShaderParams &shaderParams)
	{
		this->m_materialBuffer.push_back(shaderParams);
        int index = this->m_materialBuffer.size() - 1;

        /* Update all materials. */
        this->updateAllMaterials();

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
    void updateAllMaterials()
    {
        TW_ASSERT(this->m_shaderBuffer);
        /* At least an Emissive BSDF is stored. */
        TW_ASSERT(this->m_materialBuffer.size() > 0);

        this->m_shaderBuffer->setSize(this->m_materialBuffer.size());
        std::cout << "[Info] Updated ShaderBuffer." << std::endl;

        /* Setup shaderBuffer for GPU Program */
        auto shaderParamsArray = static_cast<CommonStructs::ShaderParams *>(this->m_shaderBuffer->map());
        for (auto itr = this->m_materialBuffer.begin(); itr != this->m_materialBuffer.end(); ++itr)
        {
            shaderParamsArray[itr - this->m_materialBuffer.begin()] = *itr;
        }

        this->m_context["shaderBuffer"]->setBuffer(this->m_shaderBuffer);

        /* Unmap buffer. */
        this->m_shaderBuffer->unmap();
    }


private:
    const std::map<std::string, optix::Program> &m_programsMap;
    optix::Context     m_context;

    /// ShaderBuffer
	optix::Buffer m_shaderBuffer;
    /// MaterialBuffer contains CommonStructs::ShaderParams
    std::vector<CommonStructs::ShaderParams> m_materialBuffer;

	constexpr static int MAX_ShaderBuffer_Size = 20;
};