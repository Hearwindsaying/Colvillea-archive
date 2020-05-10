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

#include "colvillea/Module/Sampler/SobolMatrices.h"

// About Desktop OpenGL function loaders:
//  Modern desktop OpenGL doesn't have a standard portable header file to load OpenGL function pointers.
//  Helper libraries are often used for this purpose! We use glew here.

#include <gl/glew.h>

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>



using namespace optix;

/* SH Clipping Test scene. */
void create_SHClippingTest(std::shared_ptr<SceneGraph> &sceneGraph, std::shared_ptr<LightPool> &lightPool, std::shared_ptr<MaterialPool> &materialPool, const std::string & basePath)
{
    lightPool->createHDRILight(basePath + "HDRI\\uffizi-large.hdr", optix::make_float3(0.f, 0.f, 0.f));
    sceneGraph->createSampler(CommonStructs::SamplerType::IndependentSampler);

    /* Canonical Soup testing*/
    std::shared_ptr<BSDF> soupBSDF;
    int soupIdx = materialPool->createLambertMaterial(optix::make_float4(0.63f, 0.065f, 0.05f, 1.f), soupBSDF);
    float3 A = make_float3(-1.f, -1, 0.f), C = make_float3(1.f, 1.f, 0.f),
        B = make_float3(-1.f, 1.f, 0.f), D = make_float3(1.f, -1.f, 0.f);
    //     float3 A = make_float3(0.f, -2.f*sqrt(3) / 6.f, 0.f), C = make_float3(0.5f, sqrt(3) / 6.f, 0.f),
    //         B = make_float3(-0.5f, sqrt(3) / 6.f, 0.f), D = make_float3(1.f, -2.f*sqrt(3) / 6.f, 0.f);
    sceneGraph->createTriangleSoup(soupIdx, soupBSDF, { A,C,B,A,D,C });

    /* Z=0 Plane */
    std::shared_ptr<BSDF> planeBSDF;
    int planeIdx = materialPool->createLambertMaterial(optix::make_float4(0.63f, 0.065f, 0.05f, 1.f), planeBSDF);
    float3 A1 = make_float3(-10.f, -10, 0.f), C1 = make_float3(10.f, 10.f, 0.f),
        B1 = make_float3(-10.f, 10.f, 0.f), D1 = make_float3(10.f, -10.f, 0.f);
    //     float3 A = make_float3(0.f, -2.f*sqrt(3) / 6.f, 0.f), C = make_float3(0.5f, sqrt(3) / 6.f, 0.f),
    //         B = make_float3(-0.5f, sqrt(3) / 6.f, 0.f), D = make_float3(1.f, -2.f*sqrt(3) / 6.f, 0.f);
    sceneGraph->createTriangleSoup(planeIdx, planeBSDF, { A1,C1,B1,A1,D1,C1 });
}

/* Create Cornellbox. */
void create_CornellBoxScene(std::shared_ptr<SceneGraph> &sceneGraph, std::shared_ptr<LightPool> &lightPool, std::shared_ptr<MaterialPool> &materialPool, const std::string & basePath)
{
    sceneGraph->createSampler(CommonStructs::SamplerType::IndependentSampler);
    /* Create triangle mesh. */
     std::shared_ptr<BSDF> lamBSDF;
     int lamIdx = materialPool->createLambertMaterial(optix::make_float4(0.63f, 0.065f, 0.05f, 1.f), lamBSDF);
     sceneGraph->createTriangleMesh(
         basePath + "Cornell\\red.obj",
         lamIdx, lamBSDF);

     std::shared_ptr<BSDF> lamBSDF_green;
     int lamIdx_green = materialPool->createLambertMaterial(optix::make_float4(0.14f, 0.45f, 0.091f, 1.f), lamBSDF_green);
     sceneGraph->createTriangleMesh(
         basePath + "Cornell\\green.obj",
         lamIdx_green, lamBSDF_green);

     std::shared_ptr<BSDF> lamBSDF_grey;
     int lamIdx_grey = materialPool->createLambertMaterial(optix::make_float4(0.725f, 0.71f, 0.68f, 1.f), lamBSDF_grey);
     sceneGraph->createTriangleMesh(
         basePath + "Cornell\\grey.obj",
         lamIdx_grey, lamBSDF_grey);

    /* Create light. */
     std::shared_ptr<BSDF> emissiveBSDF;
     int emissiveIdx = materialPool->getEmissiveMaterial(emissiveBSDF);
     lightPool->createQuadLight(
          make_float3(0.f, 0.f, 11.7f), make_float3(0.f),
          make_float3(3.25f, 2.625f, 1.f),
         make_float3(17.f, 12.f, 4.f)/17.f, 17.f, emissiveIdx, emissiveBSDF, true);
}

/* Create test scene. */
void create_TestScene(std::shared_ptr<SceneGraph> &sceneGraph, std::shared_ptr<LightPool> &lightPool, std::shared_ptr<MaterialPool> &materialPool, const std::string & basePath)
{
    //lightPool->createHDRILight(basePath + "HDRI\\uffizi-large.hdr", optix::make_float3(0.f,0.f,0.f));
    sceneGraph->createSampler(CommonStructs::SamplerType::IndependentSampler);// todo:ifdef USE_HALTON_SAMPLER to enable Halton

    std::shared_ptr<BSDF> roughmetal_bsdf;
    int roughmetalIdx = materialPool->createRoughMetalMaterial(roughmetal_bsdf, 0.01f, make_float4(1.66f, 0.95151f, 0.7115f, 0.f), make_float4(8.0406f, 6.3585f, 5.1380f, 0.f));
    auto quadShapeB = sceneGraph->createQuad(sceneGraph.get(),
        roughmetalIdx,
        make_float3(0.f),make_float3(0.f),make_float3(1.f,1.f,1.f), roughmetal_bsdf);

    /*std::shared_ptr<BSDF> whiteplastic;
    int whiteplasticId = materialPool->createPlasticMaterial(0.1f, 1.5f, make_float4(1.0f), make_float4(0.04f), whiteplastic);
    sceneGraph->createTriangleMesh(basePath + "Dining-room\\models\\whiteplastic.obj", whiteplasticId, whiteplastic);*/

//     std::shared_ptr<BSDF> chrome;
//     int chromeId = materialPool->createRoughMetalMaterial(chrome, 0.05f, make_float4(4.369683f, 2.916703f, 1.654701f, 0.0f), make_float4(5.206434f, 4.231365f, 3.754947f, 0.0f));
//     sceneGraph->createTriangleMesh(basePath + "Dining-room\\models\\dragon.obj", chromeId, chrome);
    /*std::shared_ptr<BSDF> chrome;
    int chromeId = materialPool->createRoughMetalMaterial(chrome, 0.05f, make_float4(4.369683f, 2.916703f, 1.654701f, 0.0f), make_float4(5.206434f, 4.231365f, 3.754947f, 0.0f));
    sceneGraph->createTriangleMesh(basePath + "Dining-room\\models\\chrome.obj", chromeId, chrome);*/

    std::shared_ptr<BSDF> emissiveBSDF;
    int emissiveIdx = materialPool->getEmissiveMaterial(emissiveBSDF);
    lightPool->createQuadLight(
        (make_float3(-0.5f, 0.f, 1.5f)),make_float3(0.f),make_float3(0.5f,0.5f,1.f),
        make_float3(1.f, 0.f, 0.f), 4.0f, emissiveIdx, emissiveBSDF, true);
    /*lightPool->createQuadLight(
        (make_float3(0.5f, 0.f, 1.5f)), make_float3(0.f), make_float3(0.5f, 0.5f, 1.f),
        make_float3(0.f, 0.f, 1.f), 4.0f, emissiveIdx, emissiveBSDF, true);*/
}

/* Create Dining Room scene. */
void create_DiningRoom(std::shared_ptr<SceneGraph> &sceneGraph, std::shared_ptr<LightPool> &lightPool, std::shared_ptr<MaterialPool> &materialPool, const std::string & basePath)
{
    /* Specify Integrator. */
    //sceneGraph->changeIntegrator(IntegratorType::PathTracing);
    /* Create HDRProbe. */
    lightPool->createHDRILight(basePath + "Dining-room\\textures\\Skydome.hdr", optix::make_float3(0.f, 0.f, 0.f));

    

    /* Create Sampler. */
    sceneGraph->createSampler(CommonStructs::SamplerType::SobolQMCSampler);

    /* Create BSDFs and Triangle Meshes. */
    /*std::shared_ptr<BSDF> whiteplastic;
    int whiteplasticId = materialPool->createPlasticMaterial(0.1f, 1.5f, make_float4(1.0f), make_float4(0.04f), whiteplastic);
    sceneGraph->createTriangleMesh(basePath + "Dining-room\\models\\whiteplastic.obj", whiteplasticId, whiteplastic);*/

    /*std::shared_ptr<BSDF> chrome;
    int chromeId = materialPool->createRoughMetalMaterial(chrome, 0.05f, make_float4(4.369683f, 2.916703f, 1.654701f, 0.0f), make_float4(5.206434f, 4.231365f, 3.754947f, 0.0f));
    sceneGraph->createTriangleMesh(basePath + "Dining-room\\models\\chrome.obj", chromeId, chrome);*/

    std::shared_ptr<BSDF> blackrubber;
    int blackrubberId = materialPool->createPlasticMaterial(0.1f, 1.5f, make_float4(0.05f), make_float4(0.04f), blackrubber);
    sceneGraph->createTriangleMesh(basePath + "Dining-room\\models\\blackrubber.obj", blackrubberId, blackrubber);

    std::shared_ptr<BSDF> walls;
    int wallsId = materialPool->createLambertMaterial(make_float4(0.2f), walls);
    sceneGraph->createTriangleMesh(basePath + "Dining-room\\models\\walls.obj", wallsId, walls);

    std::shared_ptr<BSDF> artwork;
    int artworkId = materialPool->createLambertMaterial(basePath + "Dining-room\\textures\\picture3.tga", artwork);
    sceneGraph->createTriangleMesh(basePath + "Dining-room\\models\\artwork.obj", artworkId, artwork);

    std::shared_ptr<BSDF> none;
    int noneId = materialPool->createLambertMaterial(make_float4(0.0f), none);
    sceneGraph->createTriangleMesh(basePath + "Dining-room\\models\\none.obj", noneId, none);

    std::shared_ptr<BSDF> floortiles;
    int floortilesId = materialPool->createLambertMaterial(basePath + "Dining-room\\textures\\tiles.tga", floortiles);
    sceneGraph->createTriangleMesh(basePath + "Dining-room\\models\\floortiles.obj", floortilesId, floortiles);

    std::shared_ptr<BSDF> blackpaint;
    int blackpaintId = materialPool->createPlasticMaterial(0.2f, 1.5f, make_float4(0.01f), make_float4(0.04f), blackpaint);
    sceneGraph->createTriangleMesh(basePath + "Dining-room\\models\\blackpaint.obj", blackpaintId, blackpaint);

    std::shared_ptr<BSDF> whitemarble;
    int whitemarbleId = materialPool->createLambertMaterial(make_float4(0.325037f), whitemarble);
    sceneGraph->createTriangleMesh(basePath + "Dining-room\\models\\whitemarble.obj", whitemarbleId, whitemarble);

    std::shared_ptr<BSDF> gold;
    int goldId = materialPool->createRoughMetalMaterial(gold, 0.1f, make_float4(0.143119f, 0.374957f, 1.442479f, 0.0f), make_float4(3.983160f, 2.385721f, 1.603215f, 0.0f));
    sceneGraph->createTriangleMesh(basePath + "Dining-room\\models\\gold.obj", goldId, gold);

    std::shared_ptr<BSDF> ceramic;
    int ceramicId = materialPool->createPlasticMaterial(0.01f, 1.5f, make_float4(1.0f), make_float4(0.04f), ceramic);
    sceneGraph->createTriangleMesh(basePath + "Dining-room\\models\\ceramic.obj", ceramicId, ceramic);

    std::shared_ptr<BSDF> roughmetal;
    int roughmetalId = materialPool->createRoughMetalMaterial(roughmetal, 0.1f, make_float4(1.657460f, 0.880369f, 0.521229f, 0.0f), make_float4(9.223869f, 6.269523f, 4.837001f, 0.0f));
    sceneGraph->createTriangleMesh(basePath + "Dining-room\\models\\roughmetal.obj", roughmetalId, roughmetal);

    std::shared_ptr<BSDF> paintedceramic;
    int paintedceramicId = materialPool->createPlasticMaterial(0.01f, 1.5f, basePath + "Dining-room\\textures\\teacup.tga", make_float4(0.04f), paintedceramic);
    sceneGraph->createTriangleMesh(basePath + "Dining-room\\models\\paintedceramic.obj", paintedceramicId, paintedceramic);

    std::shared_ptr<BSDF> skirtwood;
    int skirtwoodId = materialPool->createPlasticMaterial(0.01f, 1.5f, make_float4(0.684615f), make_float4(0.04f), skirtwood);
    sceneGraph->createTriangleMesh(basePath + "Dining-room\\models\\skirtwood.obj", skirtwoodId, skirtwood);

    std::shared_ptr<BSDF> frostedglass;
    int frostedglassId = materialPool->createPlasticMaterial(0.01f, 1.5f, make_float4(0.793110f), make_float4(0.04f), frostedglass);
    sceneGraph->createTriangleMesh(basePath + "Dining-room\\models\\frostedglass.obj", frostedglassId, frostedglass);

    /* Create light. */
    std::shared_ptr<BSDF> emissiveBSDF;
    int emissiveIdx = materialPool->getEmissiveMaterial(emissiveBSDF);
    lightPool->createQuadLight(
        make_float3(1.0f, 0.0f, 7.0f), make_float3(0.f),
        make_float3(3.0f, 3.0f, 1.f),
        make_float3(1.f, 1.f, 1.f), 18.f, emissiveIdx, emissiveBSDF, true);
}

int main(int argc, char *argv[])
{
/*    QuadLight::TestZHRecurrence();*/
    /*_CrtSetBreakAlloc(208);*/
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

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
    const uint32_t filmWidth = 1280, filmHeight = 720;
    //uint32_t filmWidth = 600, filmHeight = 600;
    //const uint32_t filmWidth = 1024, filmHeight = 1024;
    /*const float fov = 60.f;*/
    const float fov = 45.f;

    /* Create application instance first.*/
    std::unique_ptr<Application> application = std::make_unique<Application>(glfwWindow, filmWidth, filmHeight, 0);

    /* Create sceneGraph instance. */
    std::shared_ptr<SceneGraph> sceneGraph = std::make_unique<SceneGraph>(application.get(), application->getProgramsMap(), application->getContext(), filmWidth, filmHeight);
    
    /* Create materialPool instance. */
    std::shared_ptr<MaterialPool> materialPool = MaterialPool::createMaterialPool(application.get(), application->getProgramsMap(), application->getContext());

    /* Create lightPool instance. */
    std::shared_ptr<LightPool> lightPool = LightPool::createLightPool(application.get(), application->getProgramsMap(), application->getContext(), sceneGraph);


    /* Create scene using sceneGraph::createXXX methods. */
    sceneGraph->createCamera(Matrix4x4::identity(), fov, filmWidth, filmHeight, 0.06f, 0.0f);
    create_CornellBoxScene(sceneGraph, lightPool, materialPool, examplesBasePath); /* left scene configurations are created... */
    //create_TestScene(sceneGraph, lightPool, materialPool, examplesBasePath);
    //create_DiningRoom(sceneGraph, lightPool, materialPool, examplesBasePath);

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

    //_CrtDumpMemoryLeaks();
    return 0;
}