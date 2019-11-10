#pragma once

#include "colvillea/Device/Toolkit/CommonStructs.h"
#include "colvillea/Module/Image/ImageLoader.h"
#include "colvillea/Module/Material/MaterialPool.h"

namespace GUIHelper
{
    /**
      * @brief Helper function in Application::drawInspector() for
      * showing BSDF editable reflectance parameter.
      *
      * @param[in] bsdf
      * @param[in] application
      *
      * @see Application::drawInspector()
      */
    inline void drawInspector_MaterialCollapsingHeader_Reflectance(const std::shared_ptr<BSDF> &bsdf, Application *application)
    {
        optix::float4 reflectance = bsdf->getReflectance();
        std::string reflectanceTexture = bsdf->getReflectanceTextureFilename();
        bool enableReflectanceTexture = bsdf->getEnableReflectanceTexture();

        /* Reflectance. */
        ImGui::AlignTextToFramePadding();
        ImGui::Text("                        Reflectance"); ImGui::SameLine(200.f);
        ImGui::SetNextItemWidth(165);
        if (ImGui::ColorEdit3("##Reflectance", static_cast<float *>(&reflectance.x), ImGuiColorEditFlags__OptionsDefault))
        {
            bsdf->setReflectance(reflectance);
            application->resetRenderParams();
        }
        /* Texture. */
        ImGui::AlignTextToFramePadding();
        ImGui::Text("                              Texture"); ImGui::SameLine(200.f);
        ImGui::SetNextItemWidth(165);
        ImGui::Button(reflectanceTexture.c_str()/*, ImVec2(165.f, 0.0f)*/);
        ImGui::SameLine();
        if (ImGui::Button("M"))
        {
            static char const * lFilterPatterns[2] = { "*.tga" };
            const char * reflectanceTextureFilename_c_str = tinyfd_openFileDialog("Select an Image File", "", 1, lFilterPatterns, "TGA Files(*.tga)", 0);
            if (reflectanceTextureFilename_c_str != NULL)
            {
                bsdf->setReflectanceTextureFilename(reflectanceTextureFilename_c_str);
                application->resetRenderParams();
            }
        }
        if (ImGui::Checkbox("##Enable Reflectance Texture", &enableReflectanceTexture))
        {
            bsdf->setEnableReflectanceTexture(enableReflectanceTexture);
            application->resetRenderParams();
        }
    }

    /**
      * @brief Helper function in Application::drawInspector() for
      * showing BSDF editable roughness parameter.
      *
      * @param[in] bsdf
      * @param[in] application
      *
      * @see Application::drawInspector()
      */
    inline void drawInspector_MaterialCollapsingHeader_Roughness(const std::shared_ptr<BSDF> &bsdf, Application *application)
    {
        float roughness = bsdf->getRoughness();
        ImGui::AlignTextToFramePadding();
        ImGui::Text("                         Roughness"); ImGui::SameLine(200.f);
        ImGui::SetNextItemWidth(165);
        if (ImGui::InputFloat("##Roughness", &roughness, 0.005f, 1.f))
        {
            bsdf->setRoughness(clamp(roughness, 0.005f, 1.0f));
            application->resetRenderParams();
        }
    }

    /**
      * @brief Helper function in Application::drawInspector() for
      * showing BSDF editable specular parameter.
      *
      * @param[in] bsdf
      * @param[in] application
      *
      * @see Application::drawInspector()
      */
    inline void drawInspector_MaterialCollapsingHeader_Specular(const std::shared_ptr<BSDF> &bsdf, Application *application)
    {
        optix::float4 specular = bsdf->getSpecular();

        /* Specular. */
        ImGui::AlignTextToFramePadding();
        ImGui::Text(""); ImGui::SameLine(132.f);
        ImGui::Text("Specular"); ImGui::SameLine(200.f);
        ImGui::SetNextItemWidth(165);
        if (ImGui::ColorEdit3("##Specular", static_cast<float *>(&specular.x), ImGuiColorEditFlags__OptionsDefault))
        {
            bsdf->setSpecular(specular);
            application->resetRenderParams();
        }
    }

    /**
      * @brief Helper function in Application::drawInspector() for
      * showing BSDF editable eta parameter.
      *
      * @param[in] bsdf
      * @param[in] application
      *
      * @see Application::drawInspector()
      */
    inline void drawInspector_MaterialCollapsingHeader_Eta(const std::shared_ptr<BSDF> &bsdf, Application *application)
    {
        optix::float4 eta = bsdf->getEta();

        /* Eta. */
        ImGui::AlignTextToFramePadding();
        ImGui::Text("                                     Eta"); ImGui::SameLine(200.f);
        ImGui::SetNextItemWidth(165);
        if (ImGui::ColorEdit3("##Eta", static_cast<float *>(&eta.x), ImGuiColorEditFlags__OptionsDefault))
        {
            bsdf->setEta(eta);
            application->resetRenderParams();
        }
    }

    /**
      * @brief Helper function in Application::drawInspector() for
      * showing BSDF editable kappa parameter.
      *
      * @param[in] bsdf
      * @param[in] application
      *
      * @see Application::drawInspector()
      */
    inline void drawInspector_MaterialCollapsingHeader_Kappa(const std::shared_ptr<BSDF> &bsdf, Application *application)
    {
        optix::float4 kappa = bsdf->getKappa();

        /* Kappa. */
        ImGui::AlignTextToFramePadding();
        ImGui::Text("                                Kappa"); ImGui::SameLine(200.f);
        ImGui::SetNextItemWidth(165);
        if (ImGui::ColorEdit3("##Kappa", static_cast<float *>(&kappa.x), ImGuiColorEditFlags__OptionsDefault))
        {
            bsdf->setKappa(kappa);
            application->resetRenderParams();
        }
    }

    /**
      * @brief Helper function in Application::drawInspector() for
      * showing BSDF editable IOR parameter.
      *
      * @param[in] bsdf
      * @param[in] application
      *
      * @see Application::drawInspector()
      */
    inline void drawInspector_MaterialCollapsingHeader_IOR(const std::shared_ptr<BSDF> &bsdf, Application *application)
    {
        float ior = bsdf->getIOR();
        /* IOR. */
        ImGui::AlignTextToFramePadding();
        ImGui::Text("                                     IOR"); ImGui::SameLine(200.f);
        ImGui::SetNextItemWidth(165);
        if (ImGui::InputFloat("##IOR", &ior, 1.0f, 4.0f))
        {
            bsdf->setIOR(clamp(ior, 1.0f, 4.0f));
            application->resetRenderParams();
        }
    }
}
