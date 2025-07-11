#pragma once
#include "camera_interface.hpp"
#include <k4a/k4a.h>
#include <memory>
#include <string>

class KinectCameraAdapter : public CameraInterface {
public:
    KinectCameraAdapter();
    ~KinectCameraAdapter() override;

    // CameraInterface实现
    bool openCamera() override;
    void closeCamera() override;
    bool getFrame(cv::Mat& frame) override;
    bool isOpen() const override;
    std::string getCameraInfo() const override;

    // Kinect特有设置
    bool setColorFormat(k4a_image_format_t format = K4A_IMAGE_FORMAT_COLOR_BGRA32);
    bool setColorResolution(k4a_color_resolution_t resolution = K4A_COLOR_RESOLUTION_720P);
    bool setDeviceIndex(uint32_t device_index = 0);

private:
    k4a_device_t device_;
    k4a_capture_t capture_;
    k4a_device_configuration_t config_;
    
    bool initialized_;
    std::string serial_number_;
    uint32_t device_index_;
    
    // 禁用拷贝构造和赋值
    KinectCameraAdapter(const KinectCameraAdapter&) = delete;
    KinectCameraAdapter& operator=(const KinectCameraAdapter&) = delete;
}; 