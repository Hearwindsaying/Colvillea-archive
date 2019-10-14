#include "colvillea/Module/Camera/CameraController.h"


void CameraController::handleInputGUIEvent(const InputMouseActionType mouseAction, const optix::int2 screenPos)
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