#include "Application.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <random>
#include <string>

#include <imgui/imgui.h>

//#define IMGUI_DEFINE_MATH_OPERATORS 1
//#include <imgui/imgui_internal.h>

#include <imgui/imgui_impl_glfw_gl2.h> //legacy

#include <GL/glew.h>
//#if defined( _WIN32 )
//#include <GL/wglew.h> //"HGPUNV" redefinition
//#endif
#include <GLFW/glfw3.h>

#include "TWAssert.h"
#include "GlobalDefs.h"
#include "../Device/Toolkit/CommonStructs.h"
#include "../Module/Camera/CameraController.h"
#include "../Module/Image/ImageLoader.h"
#include "SceneGraph.h"

#include <src/sampleConfig.h>

using namespace optix;

Application::Application(GLFWwindow* glfwWindow, const uint32_t filmWidth, const uint32_t filmHeight, const int optixReportLevel, const uint32_t optixStackSize) : 
    m_filmWidth(filmWidth), m_filmHeight(filmHeight), 
    m_optixReportLevel(optixReportLevel), m_stackSize(optixStackSize),
    m_sysIterationIndex(0),m_resetRenderParamsNotification(true),
    m_sceneGraph(nullptr), m_cameraController(nullptr)
{
    /* Output OptiX Device information. */
    this->outputDeviceInfo();

    /* Initialize ImGui, OpenGL context for RenderView, 
     * -- setup OptiX context, filtering buffers and 
     * -- callable program gropus. */
    this->initializeImGui(glfwWindow);
    this->initializeRenderView();

    try
    {
        this->initializeContext();
        this->initializeOutputBuffers();
        this->initializeCallableProgramGroup();
    }
    catch (Exception& e)
    {
        std::cerr << e.getErrorString() << std::endl;
        DebugBreak();
    }
}

Application::~Application()
{
    this->m_context->destroy();
    std::cout << "[Info] Context has been destroyed." << std::endl;

    ImGui_ImplGlfwGL2_Shutdown();
    ImGui::DestroyContext();
}


void Application::buildSceneGraph(std::unique_ptr<SceneGraph> &sceneGraph)
{
    this->m_sceneGraph = std::move(sceneGraph);
    this->m_cameraController = std::make_unique<CameraController>(this->m_sceneGraph->getCamera(), this->m_filmWidth, this->m_filmHeight);

    try
    {
        this->m_sceneGraph->buildGraph();

        if (this->m_sceneGraph->getHDRILight())
        {
            this->m_sceneGraph->getHDRILight()->loadLight();
        }
        else
        {
            this->m_context->validate();
        }

    }
    catch (optix::Exception& e)
    {
        std::cerr << e.getErrorString() << std::endl;
        DebugBreak();
    }
}

void Application::drawWidget()
{
    static uint32_t frame_count = 0; // todo:use iteration index


    //{
    //    static const ImGuiWindowFlags window_flags = 0;

    //    //ImGui::SetNextWindowPos(ImVec2(2.0f, 40.0f));
    //    ImGui::SetNextWindowSize(ImVec2(550, 680), ImGuiCond_FirstUseEver);
    //    if (!ImGui::Begin("Hierarchy", NULL, window_flags))
    //    {
    //        // Early out if the window is collapsed, as an optimization.
    //        ImGui::End();
    //        return;
    //    }
    //    ImGui::PushItemWidth(-140);

    //    ImGui::Spacing();

    //    ImGui::End();
    //}

    {
        static const ImGuiWindowFlags window_flags = 0;

        ImGui::SetNextWindowPos(ImVec2(-1.0f, 826.0f));
        ImGui::SetNextWindowSize(ImVec2(550, 200), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("System", NULL, window_flags))
        {
            // Early out if the window is collapsed, as an optimization.
            ImGui::End();
            return;
        }
        ImGui::PushItemWidth(-140);

        ImGui::Spacing();


        //////////////////////////////////////////////////////////////////////////
        //System Module:
        ImGui::Text("Current Iterations:%d", this->m_sysIterationIndex);

        auto GetFPS = [](const int frame)->float
        {
            static double fps = -1.0;
            static unsigned last_frame_count = 0;
            static auto last_update_time = std::chrono::system_clock::now();
            static decltype(last_update_time) current_time = last_update_time;
            current_time = std::chrono::system_clock::now();
            std::chrono::duration<double> fp_ms = current_time - last_update_time;
            if (fp_ms.count() > 0.5)
            {
                fps = (frame_count - last_frame_count) / fp_ms.count();
                last_frame_count = frame_count;
                last_update_time = current_time;
            }

            return static_cast<float>(fps);
        };

        auto GetCurrentDateTime = []() ->std::string
        {
            time_t     now = time(0);
            struct tm  tstruct;
            char       buf[80];
            tstruct = *localtime(&now);

            strftime(buf, sizeof(buf), "%Y-%m-%d %H-%M-%S", &tstruct);

            return buf;
        };

        float currfps = GetFPS(frame_count++);
        ImGui::Text("FPS(frame per second):%.2f\nAverage Rendering Time(per launch):%.5fms", currfps, 1000.f / currfps);

        if (ImGui::Button("Save Current Result"))
        {
            ImageLoader::saveHDRBufferToImage(this->m_sysHDRBuffer, (GetCurrentDateTime() + ".exr").c_str());
            //             double currentRenderingTime = sutil::currentTime();
            //             double renderingTimeElapse = currentRenderingTime - this->startRenderingTime;
            //             std::cout << "[Info]currentFrame:" << sysIterationIndex << " time elapsed:" << renderingTimeElapse << std::endl;
        }

        ImGui::End();
    }
}

void Application::render()
{
    try
    {
        if (this->m_resetRenderParamsNotification)
        {
            this->m_sysIterationIndex = 0;
            this->m_context->launch(toUnderlyingValue(RayGenerationEntryType::InitFilter), this->m_filmWidth, this->m_filmHeight);
            this->m_resetRenderParamsNotification = false;
        }

        this->m_context->launch(toUnderlyingValue(RayGenerationEntryType::Render), this->m_filmWidth, this->m_filmHeight);
        this->m_context->launch(toUnderlyingValue(RayGenerationEntryType::Filter), this->m_filmWidth, this->m_filmHeight);

        this->drawRenderView();

        this->m_context["sysIterationIndex"]->setUint(this->m_sysIterationIndex++);
    }
    catch (optix::Exception& e)
    {
        std::cerr << e.getErrorString() << std::endl;
        DebugBreak();
    }
}

void Application::handleInputEvent(bool dispatchMouseInput)
{
    ImGuiIO const& io = ImGui::GetIO();
    const ImVec2 mousePosition = ImGui::GetMousePos();

    CameraController::InputMouseActionType mouseAction;

    if (dispatchMouseInput) /* Only allow camera interactions to begin when interacting with the GUI.  However, release operation is not affected. */
    {
        if (ImGui::IsMouseDown(0))
        {
            mouseAction = static_cast<CameraController::InputMouseActionType>(toUnderlyingValue(CameraController::InputMouseActionType::LeftMouse) | toUnderlyingValue(CameraController::InputMouseActionType::Down));
            this->m_cameraController->handleInputGUIEvent(mouseAction, make_int2(mousePosition.x, mousePosition.y));
        }

        else if (ImGui::IsMouseDown(1))
        {
            mouseAction = static_cast<CameraController::InputMouseActionType>(toUnderlyingValue(CameraController::InputMouseActionType::RightMouse) | toUnderlyingValue(CameraController::InputMouseActionType::Down));
            this->m_cameraController->handleInputGUIEvent(mouseAction, make_int2(mousePosition.x, mousePosition.y));
        }

        else if (ImGui::IsMouseDown(2))
        {
            mouseAction = static_cast<CameraController::InputMouseActionType>(toUnderlyingValue(CameraController::InputMouseActionType::MiddleMouse) | toUnderlyingValue(CameraController::InputMouseActionType::Down));
            this->m_cameraController->handleInputGUIEvent(mouseAction, make_int2(mousePosition.x, mousePosition.y));
        }

        else if (ImGui::IsMouseReleased(0))
        {
            mouseAction = static_cast<CameraController::InputMouseActionType>(toUnderlyingValue(CameraController::InputMouseActionType::LeftMouse) | toUnderlyingValue(CameraController::InputMouseActionType::Release));
            this->m_cameraController->handleInputGUIEvent(mouseAction, make_int2(mousePosition.x, mousePosition.y));
        }

        else if (ImGui::IsMouseReleased(1))
        {
            mouseAction = static_cast<CameraController::InputMouseActionType>(toUnderlyingValue(CameraController::InputMouseActionType::RightMouse) | toUnderlyingValue(CameraController::InputMouseActionType::Release));
            this->m_cameraController->handleInputGUIEvent(mouseAction, make_int2(mousePosition.x, mousePosition.y));
        }

        else if (ImGui::IsMouseReleased(2))
        {
            mouseAction = static_cast<CameraController::InputMouseActionType>(toUnderlyingValue(CameraController::InputMouseActionType::MiddleMouse) | toUnderlyingValue(CameraController::InputMouseActionType::Release));
            this->m_cameraController->handleInputGUIEvent(mouseAction, make_int2(mousePosition.x, mousePosition.y));
        }
    }

    else
    {
        if (ImGui::IsMouseReleased(0))
        {
            mouseAction = static_cast<CameraController::InputMouseActionType>(toUnderlyingValue(CameraController::InputMouseActionType::LeftMouse) | toUnderlyingValue(CameraController::InputMouseActionType::Release));
            this->m_cameraController->handleInputGUIEvent(mouseAction, make_int2(mousePosition.x, mousePosition.y));
        }

        else if (ImGui::IsMouseReleased(1))
        {
            mouseAction = static_cast<CameraController::InputMouseActionType>(toUnderlyingValue(CameraController::InputMouseActionType::RightMouse) | toUnderlyingValue(CameraController::InputMouseActionType::Release));
            this->m_cameraController->handleInputGUIEvent(mouseAction, make_int2(mousePosition.x, mousePosition.y));
        }

        else if (ImGui::IsMouseReleased(2))
        {
            mouseAction = static_cast<CameraController::InputMouseActionType>(toUnderlyingValue(CameraController::InputMouseActionType::MiddleMouse) | toUnderlyingValue(CameraController::InputMouseActionType::Release));
            this->m_cameraController->handleInputGUIEvent(mouseAction, make_int2(mousePosition.x, mousePosition.y));
        }
    }






}

void Application::drawRenderView()
{
    RTformat buffer_format = this->m_sysOutputBuffer->getFormat();
    TW_ASSERT(buffer_format == RT_FORMAT_FLOAT4);

    const unsigned pboId = this->m_sysOutputBuffer->getGLBOId();
    TW_ASSERT(pboId);

    /* Draw RenderView widget. */
    static const ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove;
    ImGui::SetNextWindowPos(ImVec2(733.0f, 58.0f));
    ImGui::SetNextWindowSize(ImVec2(1489.0f, 810.0f));
    if (!ImGui::Begin("RenderView", NULL, window_flags))
    {
        // Early out if the window is collapsed, as an optimization.
        ImGui::End();
        return;
    }

    /* Handle mouse input event for RenderView. */
    this->handleInputEvent(ImGui::IsWindowHovered());

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->m_renderViewTexture);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);
    //todo:using glTexSubImage2D
    //glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)this->filmWidth, (GLsizei)this->filmHeight, GL_RGBA, GL_FLOAT, (void*)0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, static_cast<GLsizei>(this->m_filmWidth), static_cast<GLsizei>(this->m_filmHeight), 0, GL_RGBA, GL_FLOAT, nullptr); // RGBA32F from byte offset 0 in the pixel unpack buffer.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    ImGui::Image((void*)(intptr_t)this->m_renderViewTexture, ImVec2(this->m_filmWidth, this->m_filmHeight), ImVec2(0, 1), ImVec2(1, 0)); /* flip UV Coordinates due to the inconsistence(vertically invert) */

    ImGui::End();
}



/************************************************************************/
/*              Helper functions for initialization                     */
/************************************************************************/ 

void Application::outputDeviceInfo()
{
    unsigned int optixVersion;
    RT_CHECK_ERROR_NO_CONTEXT(rtGetVersion(&optixVersion));

    unsigned int major = optixVersion / 1000; // Check major with old formula.
    unsigned int minor;
    unsigned int micro;
    if (3 < major) // New encoding since OptiX 4.0.0 to get two digits micro numbers?
    {
        major = optixVersion / 10000;
        minor = (optixVersion % 10000) / 100;
        micro = optixVersion % 100;
    }
    else // Old encoding with only one digit for the micro number.
    {
        minor = (optixVersion % 1000) / 10;
        micro = optixVersion % 10;
    }
    std::cout << "OptiX " << major << "." << minor << "." << micro << std::endl;

    unsigned int numberOfDevices = 0;
    RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetDeviceCount(&numberOfDevices));
    std::cout << "Number of Devices = " << numberOfDevices << std::endl << std::endl;

    for (unsigned int i = 0; i < numberOfDevices; ++i)
    {
        char name[256];
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_NAME, sizeof(name), name));
        std::cout << "Device " << i << ": " << name << std::endl;

        int computeCapability[2] = { 0, 0 };
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(computeCapability), &computeCapability));
        std::cout << "  Compute Support: " << computeCapability[0] << "." << computeCapability[1] << std::endl;

        RTsize totalMemory = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(totalMemory), &totalMemory));
        std::cout << "  Total Memory: " << (unsigned long long) totalMemory << std::endl;

        int clockRate = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_CLOCK_RATE, sizeof(clockRate), &clockRate));
        std::cout << "  Clock Rate: " << clockRate << " kHz" << std::endl;

        int maxThreadsPerBlock = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, sizeof(maxThreadsPerBlock), &maxThreadsPerBlock));
        std::cout << "  Max. Threads per Block: " << maxThreadsPerBlock << std::endl;

        int smCount = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, sizeof(smCount), &smCount));
        std::cout << "  Streaming Multiprocessor Count: " << smCount << std::endl;

        int executionTimeoutEnabled = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED, sizeof(executionTimeoutEnabled), &executionTimeoutEnabled));
        std::cout << "  Execution Timeout Enabled: " << executionTimeoutEnabled << std::endl;

        int maxHardwareTextureCount = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_MAX_HARDWARE_TEXTURE_COUNT, sizeof(maxHardwareTextureCount), &maxHardwareTextureCount));
        std::cout << "  Max. Hardware Texture Count: " << maxHardwareTextureCount << std::endl;

        int tccDriver = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_TCC_DRIVER, sizeof(tccDriver), &tccDriver));
        std::cout << "  TCC Driver enabled: " << tccDriver << std::endl;

        int cudaDeviceOrdinal = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(cudaDeviceOrdinal), &cudaDeviceOrdinal));
        std::cout << "  CUDA Device Ordinal: " << cudaDeviceOrdinal << std::endl << std::endl;
    }
}

std::string Application::getPTXFilepath(const std::string & program)
{
    return std::string(SAMPLES_PTX_DIR) + std::string("\\colvillea_generated_") + program + std::string(".cu.ptx");
}

void Application::createProgramsFromPTX()
{
    auto loadProgram = [this](const std::string &file, const std::initializer_list<std::string> programs) -> void
    {
        for (const auto& program : programs)
        {
            this->m_programsMap[program] = this->m_context->createProgramFromPTXFile(this->getPTXFilepath(file), program);
        }
        
    };

    /* Load RayTracingPipeline programs. */
    loadProgram("PinholeCamera", { "Exception_Default",
                                   "RayGeneration_PinholeCamera", 
                                   "RayGeneration_InitializeFilter",
                                   "RayGeneration_Filter" });

    loadProgram("SphericalSkybox", { "Miss_Default" });

    loadProgram("TriangleMesh", { "BoundingBox_TriangleMesh" , "Intersect_TriangleMesh" });
    loadProgram("Quad",         { "BoundingBox_Quad", "Intersect_Quad" });

    loadProgram("DirectLighting", { "ClosestHit_DirectLighting" });
    loadProgram("PathTracing",    { "ClosestHit_PathTracing", "ClosestHit_PTRay_PathTracing" });

    loadProgram("HitProgram", { "AnyHit_ShadowRay_Shape" });


    /* Load Bindless Callable programs. */
    loadProgram("PointLight", { "Sample_Ld_Point", "LightPdf_Point" });
    loadProgram("HDRILight",  { "Sample_Ld_HDRI", "LightPdf_HDRI", 
                                "RayGeneration_PrefilterHDRILight" });
    loadProgram("QuadLight",  { "Sample_Ld_Quad", "LightPdf_Quad" });

    auto loadProgramMaterial = [loadProgram](const std::initializer_list<std::string> materials)->void
    {
        for (const auto& material : materials)
        {
            loadProgram(material, { "Pdf_" + material, "Eval_f_" + material, "Sample_f_" + material });
        }
    };
    loadProgramMaterial({ "Lambert","RoughMetal","RoughDielectric","SmoothGlass","Plastic","SmoothMirror","FrostedMetal" });


}



/*******************************************************************/
/*           Initialization functions called by constructor        */
/*******************************************************************/

void Application::initializeImGui(GLFWwindow *glfwWindow)
{
    /* Create ImGui context. */
    ImGui::CreateContext();
    ImGui_ImplGlfwGL2_Init(glfwWindow, true);
    ImGui_ImplGlfwGL2_NewFrame();
    ImGui::EndFrame();

#pragma region ImGui_Style_Region
    ImGuiStyle& style = ImGui::GetStyle();

    const float r = 1.0f;
    const float g = 1.0f;
    const float b = 1.0f;

    style.Colors[ImGuiCol_Text] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
    style.Colors[ImGuiCol_WindowBg] = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 0.6f);
    style.Colors[ImGuiCol_ChildWindowBg] = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 1.0f);
    style.Colors[ImGuiCol_PopupBg] = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 1.0f);
    style.Colors[ImGuiCol_Border] = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_BorderShadow] = ImVec4(r * 0.0f, g * 0.0f, b * 0.0f, 0.4f);
    style.Colors[ImGuiCol_FrameBg] = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_FrameBgActive] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_TitleBg] = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 0.2f);
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_MenuBarBg] = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 1.0f);
    style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 0.2f);
    style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_CheckMark] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_SliderGrab] = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_Button] = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_ButtonActive] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_Header] = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_HeaderActive] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_Column] = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_ColumnHovered] = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_ColumnActive] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_ResizeGrip] = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
    style.Colors[ImGuiCol_CloseButton] = ImVec4(r * 0.4f, g * 0.4f, b * 0.4f, 0.4f);
    style.Colors[ImGuiCol_CloseButtonHovered] = ImVec4(r * 0.6f, g * 0.6f, b * 0.6f, 0.6f);
    style.Colors[ImGuiCol_CloseButtonActive] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 0.8f);
    style.Colors[ImGuiCol_PlotLines] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 1.0f);
    style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
    style.Colors[ImGuiCol_PlotHistogram] = ImVec4(r * 0.8f, g * 0.8f, b * 0.8f, 1.0f);
    style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
    style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(r * 0.5f, g * 0.5f, b * 0.5f, 1.0f);
    style.Colors[ImGuiCol_ModalWindowDarkening] = ImVec4(r * 0.2f, g * 0.2f, b * 0.2f, 0.2f);
    style.Colors[ImGuiCol_DragDropTarget] = ImVec4(r * 1.0f, g * 1.0f, b * 0.0f, 1.0f); // Yellow
    style.Colors[ImGuiCol_NavHighlight] = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
    style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(r * 1.0f, g * 1.0f, b * 1.0f, 1.0f);
#pragma endregion ImGui_Style_Region
}

void Application::initializeRenderView()
{
    /* Create PBO for the fast OptiX sysOutputBuffer to texture transfer. */

    /* 1.Generate a new buffer object with glGenBuffers(). */
    glGenBuffers(1, &this->m_glPBO);
    TW_ASSERT(this->m_glPBO != 0); // Buffer size must be > 0 or OptiX can't create a buffer from it.

    /* 2.Bind the buffer object with glBindBuffer(). */
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, this->m_glPBO);

    /* 3.Copy pixel data to the buffer object with glBufferData(). */
    glBufferData(GL_PIXEL_UNPACK_BUFFER, this->m_filmWidth * this->m_filmHeight * sizeof(float) * 4, nullptr, GL_STREAM_READ); // RGBA32F from byte offset 0 in the pixel unpack buffer.

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    /* Generate texture storing PBO data, used in RenderView. */
    glGenTextures(1, &this->m_renderViewTexture);
    TW_ASSERT(this->m_renderViewTexture != 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->m_renderViewTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, 0);

    /* DAR ImGui has been changed to push the GL_TEXTURE_BIT so that this works. */
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
}

void Application::initializeContext()
{
    /* Create context. */
    this->m_context = Context::create();

    /* Setup context parameters. */
    this->m_context->setStackSize(this->m_stackSize);
    this->m_context->setPrintEnabled(true);

    if (this->m_optixReportLevel > 0)
    {
        this->m_context->setUsageReportCallback(
            [](int lvl, const char * tag, const char * msg, void * cbdata)
        {
            Application::OptixUsageReportLogger* logger = reinterpret_cast<Application::OptixUsageReportLogger*>(cbdata);
            logger->log(lvl, tag, msg);
        }, this->m_optixReportLevel, &this->m_optixUsageReportLogger);
    }
    
    /* Disable exceptions. */
    /* this->m_context->setExceptionEnabled(RT_EXCEPTION_ALL, true); */

    /* Setup launch entries. */
    this->m_context->setEntryPointCount(toUnderlyingValue(RayGenerationEntryType::CountOfType));

    /* Setup number of ray type. */
    this->m_context->setRayTypeCount(toUnderlyingValue(CommonStructs::RayType::CountOfType));

    // todo:pack scene related paramters(materialBuffer) into a struct
    this->m_context["sysSceneEpsilon"]->setFloat(1e-4f);

    /* Create gpu programs from PTX files and add to programs map. */
    this->createProgramsFromPTX();

    /* Set programs for exception, miss and filters (ray generation). */
    auto programItr = this->m_programsMap.find("Exception_Default");
    TW_ASSERT(programItr != this->m_programsMap.end());
    this->m_context->setExceptionProgram(toUnderlyingValue(RayGenerationEntryType::Render), programItr->second);

    programItr = this->m_programsMap.find("Miss_Default");
    TW_ASSERT(programItr != this->m_programsMap.end());
    this->m_context->setMissProgram(toUnderlyingValue(RayGenerationEntryType::Render), programItr->second);

    programItr = this->m_programsMap.find("RayGeneration_InitializeFilter");
    TW_ASSERT(programItr != this->m_programsMap.end());
    this->m_context->setRayGenerationProgram(toUnderlyingValue(RayGenerationEntryType::InitFilter), programItr->second);

    programItr = this->m_programsMap.find("RayGeneration_Filter");
    TW_ASSERT(programItr != this->m_programsMap.end());
    this->m_context->setRayGenerationProgram(toUnderlyingValue(RayGenerationEntryType::Filter), programItr->second);
}

void Application::initializeOutputBuffers()
{
    /* Setup progressive rendering iteration index, which is used in RandomSampler, progressive filtering. */
    this->m_context["sysIterationIndex"]->setUint(0);

    /* Setup output buffer for RenderView. */
    this->m_sysOutputBuffer = this->m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, this->m_glPBO);
    this->m_sysOutputBuffer->setFormat(RT_FORMAT_FLOAT4); // RGBA32F Buffer for output, todo:review format
    this->m_sysOutputBuffer->setSize(this->m_filmWidth, this->m_filmHeight);
    this->m_context["sysOutputBuffer"]->set(this->m_sysOutputBuffer);

    /* Setup HDR output buffer for exporting result to OpenEXR. */
    this->m_sysHDRBuffer = this->m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, this->m_filmWidth, this->m_filmHeight);
    this->m_context["sysHDRBuffer"]->set(this->m_sysHDRBuffer);

    /* Setup reconstruction and filtering buffers. */
    Buffer weightedSumBuffer = this->m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, this->m_filmWidth, this->m_filmHeight);
    this->m_context["sysSampleWeightedSum"]->setBuffer(weightedSumBuffer);

    /* Setup current radiance and weightedSum buffers for filtering.
     * Note that in current progressive filtering algorithm, it stores elements
     * of the sum of weighted radiance(f(dx,dy)*Li) with respect to the current iteration. */
    Buffer sysCurrResultBuffer = this->m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, this->m_filmWidth, this->m_filmHeight);
    this->m_context["sysCurrResultBuffer"]->setBuffer(sysCurrResultBuffer);

    /*weightedSum buffer with respect to the current iteration*/
    Buffer sysCurrWeightedSumBuffer = this->m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, this->m_filmWidth, this->m_filmHeight);
    this->m_context["sysCurrWeightedSumBuffer"]->setBuffer(sysCurrWeightedSumBuffer);
}

void Application::initializeCallableProgramGroup()
{
    /* Initialize BSDF program group. */
    Buffer bsdfPdfProgramBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, toUnderlyingValue(CommonStructs::BSDFType::CountOfType));
    Buffer bsdfEvalfProgramBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, toUnderlyingValue(CommonStructs::BSDFType::CountOfType));
    Buffer bsdfSamplefProgramBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, toUnderlyingValue(CommonStructs::BSDFType::CountOfType));


    int* bsdfPdfProgramBufferData = static_cast<int *>(bsdfPdfProgramBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
    int* bsdfEvalfProgramBufferData = static_cast<int *>(bsdfEvalfProgramBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
    int* bsdfSamplefProgramBufferData = static_cast<int *>(bsdfSamplefProgramBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));

    // todo:convert bsdfType to string for searching.
    auto loadBSDFPrograms = [this, &bsdfPdfProgramBufferData, &bsdfEvalfProgramBufferData, &bsdfSamplefProgramBufferData](CommonStructs::BSDFType bsdfType, const std::string &programName /* [in] bsdfType in string, no postfix such as _Pdf */)
    {
        //for (const auto& programName : programNames)
        //{
            auto programItr = this->m_programsMap.find("Pdf_" + programName);
            TW_ASSERT(programItr != this->m_programsMap.end());
            bsdfPdfProgramBufferData[toUnderlyingValue(bsdfType)] = programItr->second->getId();

            programItr = this->m_programsMap.find("Eval_f_" + programName);
            TW_ASSERT(programItr != this->m_programsMap.end());
            bsdfEvalfProgramBufferData[toUnderlyingValue(bsdfType)] = programItr->second->getId();

            programItr = this->m_programsMap.find("Sample_f_" + programName);
            TW_ASSERT(programItr != this->m_programsMap.end());
            bsdfSamplefProgramBufferData[toUnderlyingValue(bsdfType)] = programItr->second->getId();
       // }
    };

    // todo:using enumerating to loop over all types instead of handling them.
    loadBSDFPrograms(CommonStructs::BSDFType::Lambert,         "Lambert");
    loadBSDFPrograms(CommonStructs::BSDFType::RoughMetal,      "RoughMetal");
    loadBSDFPrograms(CommonStructs::BSDFType::RoughDielectric, "RoughDielectric");
    loadBSDFPrograms(CommonStructs::BSDFType::SmoothGlass,     "SmoothGlass");
    loadBSDFPrograms(CommonStructs::BSDFType::Plastic,         "Plastic");

    /* Note that this is on purpose that uses Lambert BSDF for emissive material. */
    loadBSDFPrograms(CommonStructs::BSDFType::Emissive, "Lambert");

    loadBSDFPrograms(CommonStructs::BSDFType::SmoothMirror, "SmoothMirror");
    loadBSDFPrograms(CommonStructs::BSDFType::FrostedMetal, "FrostedMetal"); //todo:add to scene graph

    bsdfPdfProgramBuffer->unmap();
    bsdfEvalfProgramBuffer->unmap();
    bsdfSamplefProgramBuffer->unmap();

    this->m_context["Pdf"]->setBuffer(bsdfPdfProgramBuffer);
    this->m_context["Eval_f"]->setBuffer(bsdfEvalfProgramBuffer);
    this->m_context["Sample_f"]->setBuffer(bsdfSamplefProgramBuffer);


    /* Initialize Light program group. */
    Buffer sampleLdProgramBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, toUnderlyingValue(CommonStructs::LightType::CountOfType));
    int* sampleLdProgramBufferData = static_cast<int *>(sampleLdProgramBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));

    Buffer lightPdfProgramBuffer = this->m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, toUnderlyingValue(CommonStructs::LightType::CountOfType));
    int* lightPdfProgramBufferData = static_cast<int *>(lightPdfProgramBuffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD));

    // todo:convert bsdfType to string for searching.
    auto loadLightPrograms = [this, &sampleLdProgramBufferData, &lightPdfProgramBufferData](CommonStructs::LightType lightType, const std::string &programName /* [in] lightType in string, no prefix such as Sample_Ld_ */)
    {
        auto programItr = this->m_programsMap.find("Sample_Ld_" + programName);
        TW_ASSERT(programItr != this->m_programsMap.end());
        sampleLdProgramBufferData[toUnderlyingValue(lightType)] = programItr->second->getId();

        programItr = this->m_programsMap.find("LightPdf_" + programName);
        TW_ASSERT(programItr != this->m_programsMap.end());
        lightPdfProgramBufferData[toUnderlyingValue(lightType)] = programItr->second->getId();
    };

    loadLightPrograms(CommonStructs::LightType::PointLight, "Point");
    loadLightPrograms(CommonStructs::LightType::QuadLight,  "Quad");//todo:fix name 
    loadLightPrograms(CommonStructs::LightType::HDRILight,  "HDRI");

    sampleLdProgramBuffer->unmap();
    lightPdfProgramBuffer->unmap();
    this->m_context["Sample_Ld"]->setBuffer(sampleLdProgramBuffer);
    this->m_context["LightPdf"]->setBuffer(lightPdfProgramBuffer);
}

