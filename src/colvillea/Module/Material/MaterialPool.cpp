#include "colvillea/Module/Material/MaterialPool.h"

#include "colvillea/Application/Application.h"

std::shared_ptr<MaterialPool> MaterialPool::createMaterialPool(Application *application, const std::map<std::string, optix::Program> &programsMap, const optix::Context context)
{
    std::shared_ptr<MaterialPool> materialPool = std::make_shared<MaterialPool>(programsMap, context);
    application->m_materialPool = materialPool;
    return materialPool;
}