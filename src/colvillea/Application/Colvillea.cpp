#include <iostream>
#include <locale>
#include <string>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include "colvillea/Application/Application.h"
#include "colvillea/Device/Toolkit/CommonStructs.h"
#include "colvillea/Module/Material/MaterialPool.h"
#include "colvillea/Module/Light/LightPool.h"
#include "colvillea/Application/SceneGraph.h"

// About Desktop OpenGL function loaders:
//  Modern desktop OpenGL doesn't have a standard portable header file to load OpenGL function pointers.
//  Helper libraries are often used for this purpose! We use glew here.

#include <gl/glew.h>

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>



using namespace optix;

/* Create Cornellbox. */
void create_CornellBoxScene(std::shared_ptr<SceneGraph> &sceneGraph, std::shared_ptr<LightPool> &lightPool, std::unique_ptr<Application> &application, std::shared_ptr<MaterialPool> &materialPool, const std::string & basePath)
{
    /* Create integator. */

    //lightPool->createHDRILight(basePath + "HDRI\\uffizi-large.hdr", optix::make_float3(0.f,0.f,0.f));
    //lightPool->createPointLight(optix::make_float3(0.0f, 0.0f, 11.0f), optix::make_float3(1.0f, 1.0f, 1.0f), 44.0f);


    //sceneGraph->createDirectLightingIntegrator();
    //sceneGraph->createPathTracingIntegrator(true, 5);

    /* Create sampler. */
    //sceneGraph->createSampler(CommonStructs::SamplerType::IndependentSampler);// define USE_HALTON_SAMPLER to enable Halton

    /* Create triangle mesh. */
    sceneGraph->createTriangleMesh(
        basePath + "Cornell\\green.obj",
        materialPool->createLambertMaterial(optix::make_float4(0.63f, 0.065f, 0.05f, 1.f)));
    sceneGraph->createTriangleMesh(
        basePath + "Cornell\\red.obj",
        materialPool->createLambertMaterial(optix::make_float4(0.14f, 0.45f, 0.091f, 1.f)));
    sceneGraph->createTriangleMesh(
        basePath + "Cornell\\grey.obj",
        materialPool->createLambertMaterial(optix::make_float4(0.725f, 0.71f, 0.68f, 1.f)));
    /*sceneGraph->createTriangleMesh(
        basePath + "Cornell\\lucy.obj",
        materialPool->createLambertMaterial(optix::make_float4(0.725f, 0.71f, 0.68f, 1.f)));*/

    /* Create light. */
    lightPool->createQuadLight(
         make_float3(0.f, 0.f, 11.7f), make_float3(0.f),
         make_float3(3.25f, 2.625f, 1.f),
        make_float3(17.f, 12.f, 4.f)/17.f, 17.f, materialPool->createEmissiveMaterial(), true);
}

/* Create test scene. */
void create_TestScene(std::shared_ptr<SceneGraph> &sceneGraph, std::shared_ptr<LightPool> &lightPool, std::unique_ptr<Application> &application, std::shared_ptr<MaterialPool> &materialPool, const std::string & basePath)
{
    //sceneGraph->createDirectLightingIntegrator();
    //sceneGraph->createPathTracingIntegrator(true, 5);
    lightPool->createHDRILight(basePath + "HDRI\\uffizi-large.hdr", optix::make_float3(0.f,0.f,0.f));
    sceneGraph->createSampler(CommonStructs::SamplerType::IndependentSampler);// todo:ifdef USE_HALTON_SAMPLER to enable Halton

    /* TriangleMesh is created with the help of MaterialPool. */
    //std::unique_ptr<MaterialPool> materialPool = std::make_unique<MaterialPool>(application->getProgramsMap(), application->getContext());

    //sceneGraph->createTriangleMesh(
    //    "D:\\Project\\Twilight\\GraphicsRes\\Colvillea\\sofa.obj", 
    //    materialPool->createLambertMaterial(optix::make_float4(.8f)));

    //auto quadShape = sceneGraph->createQuad(
    //    materialPool->createRoughMetalMaterial(0.03f, make_float4(1.66f,0.95151f,0.7115f,0.f), make_float4(8.0406f,6.3585f,5.1380f,0.f)), 
    //    /*materialPool->createLambertMaterial(optix::make_float4(0.8f)),*/
    //       /*Matrix4x4::translate(make_float3(0.f,0.f,4.f)) * */Matrix4x4::rotate(TwUtil::deg2rad(45.f), make_float3(1.f,0.f,0.f)));//todo:fix this
    //quadShape->flipGeometryNormal();

    auto quadShapeB = sceneGraph->createQuad(
        materialPool->createRoughMetalMaterial(0.01f, make_float4(1.66f, 0.95151f, 0.7115f, 0.f), make_float4(8.0406f, 6.3585f, 5.1380f, 0.f)),
        /*materialPool->createLambertMaterial(optix::make_float4(0.8f)),*/
        make_float3(0.f),make_float3(0.f),make_float3(1.f,1.f,1.f));
    //quadShapeB->flipGeometryNormal();

    //sceneGraph->createTriangleMesh("D:\\Project\\Twilight\\GraphicsRes\\Colvillea\\roughmetal.obj",
    //    materialPool->createRoughMetalMaterial(0.03f, make_float4(1.66f, 0.95151f, 0.7115f, 0.f), 
    //    make_float4(8.0406f, 6.3585f, 5.1380f, 0.f)));
    //sceneGraph->createTriangleMesh("D:\\Project\\Twilight\\GraphicsRes\\Colvillea\\sphere.obj", 
    //    /*materialPool->createRoughMetalMaterial(0.03f, make_float4(1.66f, 0.95151f, 0.7115f, 0.f), make_float4(8.0406f, 6.3585f, 5.1380f, 0.f))*/
    //    materialPool->createLambertMaterial(optix::make_float4(0.8f))
    //    /*materialPool->createSmoothGlassMaterial(1.46f)*/);

    //lightPool->createPointLight(optix::make_float3(0.0f, 0.0f, 3.0f), optix::make_float3(1.0f, 1.0f, 1.0f), 4.0f);

    lightPool->createQuadLight(
        (make_float3(-0.5f, 0.f, 1.5f)),make_float3(0.f),make_float3(0.5f,0.5f,1.f),
        make_float3(1.f, 0.f, 0.f), 4.0f, materialPool->getEmissiveMaterial(), true);
    lightPool->createQuadLight(
        (make_float3(0.5f, 0.f, 1.5f)), make_float3(0.f), make_float3(0.5f, 0.5f, 1.f),
        make_float3(0.f, 0.f, 1.f), 4.0f, materialPool->getEmissiveMaterial(), true);
    
    /*lightPool->createQuadLight(
        make_float3(0.f), make_float3(TwUtil::deg2rad(45.f), 0.f, 0.f), make_float3(1.f, 1.f, 1.f), make_float3(1.f), 1.f, emissiveMaterial, true);*/
}



int main(int argc, char *argv[])
{
    //_CrtSetBreakAlloc(3231);

    /* Setup GLFW/GLEW context. */
    auto glfwErrorFunc = [](int error, const char* description)
    {
        std::cerr << "Error: " << error << ": " << description << std::endl; 
    };
    glfwSetErrorCallback(glfwErrorFunc);

    if (!glfwInit())
    {
        glfwErrorFunc(1, "GLFW failed to initialize.");
        return 1;
    }

    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
    GLFWwindow* glfwWindow = glfwCreateWindow(1920, 1080, "Colvillea 0.1.0", NULL, NULL);

    if (!glfwWindow)
    {
        glfwErrorFunc(2, "glfwCreateWindow() failed.");
        glfwTerminate();
        return 2;
    }

    glfwMakeContextCurrent(glfwWindow);
    /* Disable vsync. */
    glfwSwapInterval(0);

    if (glewInit() != GL_NO_ERROR)
    {
        glfwErrorFunc(3, "GLEW failed to initialize.");
        glfwTerminate();
        return 3;
    }

    auto getExampleDirectoryPath = []()->std::string
    {
        const std::string filename = __FILE__;
        std::string::size_type extension_index = filename.find_last_of("\\");
        std::string filePath = extension_index != std::string::npos ?
            filename.substr(0, extension_index) :
            std::string();

        return filePath + "\\..\\..\\..\\examples\\";
    };

    const std::string examplesBasePath = getExampleDirectoryPath();

    
    /* Create scene. */ 

    /* Parameters shared by camera and application. */
    //const uint32_t filmWidth = 1280, filmHeight = 720;
    const uint32_t filmWidth = 1024, filmHeight = 1024;
    const float fov = 60.f;

    /* Create application instance first.*/
    std::unique_ptr<Application> application = std::make_unique<Application>(glfwWindow, filmWidth, filmHeight, 0);

    /* Create sceneGraph instance. */
    std::shared_ptr<SceneGraph> sceneGraph = std::make_unique<SceneGraph>(application.get(), application->getProgramsMap(), application->getContext(), filmWidth, filmHeight);

    /* Create materialPool instance. */
    std::shared_ptr<MaterialPool> materialPool = MaterialPool::createMaterialPool(application.get(), application->getProgramsMap(), application->getContext());

    /* Create lightPool instance. */
    std::shared_ptr<LightPool> lightPool = LightPool::createLightPool(application.get(), application->getProgramsMap(), application->getContext(), sceneGraph);


    /* Create scene using sceneGraph::createXXX methods. */
    sceneGraph->createCamera(Matrix4x4::identity(), fov, filmWidth, filmHeight);
    //create_CornellBoxScene(sceneGraph, lightPool, application, materialPool, examplesBasePath); /* left scene configurations are created... */
    create_TestScene(sceneGraph, lightPool, application, materialPool, examplesBasePath);

    /* Finally initialize scene and prepare for launch. */
    application->buildSceneGraph(sceneGraph);

    /* Main loop for Colvillea. */
    ImGuiIO& io = ImGui::GetIO();
    while (!glfwWindowShouldClose(glfwWindow))
    {
        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        /* Draw ImGui widgets. */
        application->drawWidget();

        /* Launch renderer. */
        application->render();

        /* Launch ImGui rendering. */
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(glfwWindow, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        //glClearColor(0.45f, 0.55f, 0.60f, 1.0f);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        /* Update and Render additional Platform Windows. */
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(glfwWindow);
    }

    /* Destroy OpenGL objects. */
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(glfwWindow);
    glfwTerminate();

    _CrtDumpMemoryLeaks();
    return 0;
}