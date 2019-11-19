#include "colvillea/Module/Camera/CameraController.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <locale>
#include <string>

void CameraController::handleInputGUIEvent(const InputMouseActionType mouseAction, optix::int2 screenPos)
{
    switch (this->m_cameraMotionType)
    {
        case CameraMotionType::None:
        {
            // todo:overloading bitwise and operator for scoped enum
            if ((toUnderlyingValue(mouseAction) & toUnderlyingValue(InputMouseActionType::Down)) == toUnderlyingValue(InputMouseActionType::Down))
            {
                this->m_cameraInfo.basePosition = screenPos;

                switch ((toUnderlyingValue(mouseAction)) & 
                    (toUnderlyingValue(InputMouseActionType::LeftMouse)|
                     toUnderlyingValue(InputMouseActionType::MiddleMouse)|
                     toUnderlyingValue(InputMouseActionType::RightMouse)))
                {
                case toUnderlyingValue(InputMouseActionType::LeftMouse):
                    this->m_cameraMotionType = CameraMotionType::Orbit;
                    break;
                case toUnderlyingValue(InputMouseActionType::MiddleMouse):
                    this->m_cameraMotionType = CameraMotionType::Dolly;
                    break;
                case toUnderlyingValue(InputMouseActionType::RightMouse):
                    this->m_cameraMotionType = CameraMotionType::Pan;
                    break;
                default:
                    std::cout << "[Error] mouseAction is not a type of LeftMouse, MiddleMouse or RightMouse!" << std::endl;
                    break;
                }
            }
            

            break;
        }

        case CameraMotionType::Orbit:
        case CameraMotionType::Dolly:
        case CameraMotionType::Pan:
        {
            if ((toUnderlyingValue(mouseAction) & toUnderlyingValue(InputMouseActionType::Release)) == toUnderlyingValue((InputMouseActionType::Release)))
            {
                this->m_cameraMotionType = CameraMotionType::None;
            }
            else
            {
                switch ((toUnderlyingValue(mouseAction)) &
                       (toUnderlyingValue(InputMouseActionType::LeftMouse) |
                        toUnderlyingValue(InputMouseActionType::MiddleMouse) |
                        toUnderlyingValue(InputMouseActionType::RightMouse)))
                {
                case toUnderlyingValue(InputMouseActionType::LeftMouse):
                    this->orbit(screenPos);
                    break;
                case toUnderlyingValue(InputMouseActionType::MiddleMouse):
                    this->pan(screenPos);
                    break;
                case toUnderlyingValue(InputMouseActionType::RightMouse):
                    this->dolly(screenPos);
                    break;
                default:
                    std::cout << "[Error] mouseAction is not a type of LeftMouse, MiddleMouse or RightMouse!" << std::endl;
                    break;
                }
            }

            break;
        }

    default:
        std::cout << "[Error] inputCameraInteraction is not a type of none, orbit, dolly or pan!" << std::endl;
        break;
    }
}

bool CameraController::cameraInfoToFile() const
{
    std::ofstream ofsFile("config.ini");
    if (!ofsFile)
    {
        std::cerr << "[Error] Couldn't open config.ini. for writing CameraInfo. " << std::endl;
        return false;
    }

    ofsFile << "# Config for CameraInfo -- specifying camera eye position and lookat destination. " << std::endl;

    /* Write |eye| and |lookat| into file. */
    ofsFile << "Eye.x = " << this->m_cameraInfo.eye.x << std::endl
            << "Eye.y = " << this->m_cameraInfo.eye.y << std::endl
            << "Eye.z = " << this->m_cameraInfo.eye.z << std::endl;
    ofsFile << "LookAt Destination.x = " << this->m_cameraInfo.lookAtDestination.x << std::endl
            << "LookAt Destination.y = " << this->m_cameraInfo.lookAtDestination.y << std::endl
            << "LookAt Destination.z = " << this->m_cameraInfo.lookAtDestination.z << std::endl;

    std::cout << "[Info] Successfully save config file." << std::endl;
    return true;
}

bool CameraController::cameraInfoFromFile()
{
    std::ifstream cFile("config.ini");
    if (cFile.is_open())
    {
        CameraInfo lastCameraInfo = this->getCameraInfo();

        std::string line;
        while (getline(cFile, line)) 
        {
            line.erase(std::remove_if(line.begin(), line.end(), isspace),
                line.end());
            if (line[0] == '#' || line.empty())
                continue;
            auto delimiterPos = line.find("=");

            auto name  = line.substr(0, delimiterPos);
            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            auto value = line.substr(delimiterPos + 1);
            std::transform(value.begin(), value.end(), value.begin(), ::tolower);
            if (name == "eye.x")
            {
                lastCameraInfo.eye.x = std::stof(value);
            }
            else if (name == "eye.y")
            {
                lastCameraInfo.eye.y = std::stof(value);
            }
            else if (name == "eye.z")
            {
                lastCameraInfo.eye.z = std::stof(value);
            }
            else if (name == "lookatdestination.x") /* we removed space between "lookat" and "destination" */
            {
                lastCameraInfo.lookAtDestination.x = std::stof(value);
            }
            else if (name == "lookatdestination.y")
            {
                lastCameraInfo.lookAtDestination.y = std::stof(value);
            }
            else if (name == "lookatdestination.z")
            {
                lastCameraInfo.lookAtDestination.z = std::stof(value);
            }
            else
            {
                std::cerr << "[Error] Unrecognized name: " << name << " value:" << value;
            }   
        }
        this->setCameraInfo(lastCameraInfo);

        std::cout << "[Info] Successfully read config file." << std::endl;
        return true;
    }
    else 
    {
        std::cerr << "[Error] Couldn't open config file for reading CameraInfo.\n";
        return false;
    }
}