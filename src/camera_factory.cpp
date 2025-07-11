#include "camera_interface.hpp"
#include "kinect_camera_adapter.hpp"
#include <memory>

// 相机工厂类实现
std::unique_ptr<CameraInterface> CameraFactory::createCamera(CameraType type) {
    switch (type) {
        case CameraType::KINECT:
            return std::make_unique<KinectCameraAdapter>();
        default:
            return nullptr;
    }
} 