#include "kinect_camera_adapter.hpp"
#include <iostream>
#include <k4a/k4aversion.h>

KinectCameraAdapter::KinectCameraAdapter() 
    : device_(nullptr), capture_(nullptr), initialized_(false), device_index_(0) {
    
    // 初始化默认配置
    config_.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    config_.color_resolution = K4A_COLOR_RESOLUTION_720P;
    // 默认使用 30 FPS，可根据需要改为 K4A_FRAMES_PER_SECOND_15 / 60 等
    config_.camera_fps      = K4A_FRAMES_PER_SECOND_30;
    config_.depth_delay_off_color_usec = 0;
    config_.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
    config_.subordinate_delay_off_master_usec = 0;
    config_.synchronized_images_only = false;
}

KinectCameraAdapter::~KinectCameraAdapter() {
    closeCamera();
}

bool KinectCameraAdapter::openCamera() {
    if (initialized_) {
        std::cout << "Kinect camera already initialized!" << std::endl;
        return true;
    }

    // 获取设备数量
    uint32_t device_count = k4a_device_get_installed_count();
    if (device_count == 0) {
        std::cerr << "No Azure Kinect devices found!" << std::endl;
        return false;
    }

    if (device_index_ >= device_count) {
        std::cerr << "Device index " << device_index_ << " out of range. "
                  << "Only " << device_count << " device(s) available." << std::endl;
        return false;
    }

    // 打开设备
    if (K4A_RESULT_SUCCEEDED != k4a_device_open(device_index_, &device_)) {
        std::cerr << "Failed to open Azure Kinect device " << device_index_ << std::endl;
        return false;
    }

    // 获取设备序列号
    size_t serial_number_length = 0;
    k4a_device_get_serialnum(device_, nullptr, &serial_number_length);
    if (serial_number_length > 0) {
        serial_number_.resize(serial_number_length);
        k4a_device_get_serialnum(device_, &serial_number_[0], &serial_number_length);
    }

    // 启动相机
    if (K4A_RESULT_SUCCEEDED != k4a_device_start_cameras(device_, &config_)) {
        std::cerr << "Failed to start cameras!" << std::endl;
        k4a_device_close(device_);
        device_ = nullptr;
        return false;
    }

    initialized_ = true;
    std::cout << "Azure Kinect camera initialized successfully!" << std::endl;
    std::cout << "Serial number: " << serial_number_ << std::endl;
    std::cout << "Color resolution: " << config_.color_resolution << std::endl;
    std::cout << "Color format: " << config_.color_format << std::endl;

    return true;
}

void KinectCameraAdapter::closeCamera() {
    if (initialized_) {
        if (capture_) {
            k4a_capture_release(capture_);
            capture_ = nullptr;
        }
        
        if (device_) {
            k4a_device_stop_cameras(device_);
            k4a_device_close(device_);
            device_ = nullptr;
        }
        
        initialized_ = false;
        std::cout << "Azure Kinect camera shutdown complete." << std::endl;
    }
}

bool KinectCameraAdapter::getFrame(cv::Mat& frame) {
    if (!initialized_) {
        std::cerr << "Kinect camera not initialized!" << std::endl;
        return false;
    }

    // 获取捕获
    k4a_wait_result_t wait_result = k4a_device_get_capture(device_, &capture_, 1000);
    if (wait_result == K4A_WAIT_RESULT_TIMEOUT) {
        std::cerr << "Timeout waiting for capture!" << std::endl;
        return false;
    } else if (wait_result == K4A_WAIT_RESULT_FAILED) {
        std::cerr << "Failed to get capture!" << std::endl;
        return false;
    }

    // 获取颜色图像
    k4a_image_t color_k4a_image = k4a_capture_get_color_image(capture_);
    if (color_k4a_image == nullptr) {
        std::cerr << "Failed to get color image from capture!" << std::endl;
        return false;
    }

    // 获取图像信息
    int width = k4a_image_get_width_pixels(color_k4a_image);
    int height = k4a_image_get_height_pixels(color_k4a_image);
    k4a_image_format_t format = k4a_image_get_format(color_k4a_image);
    uint8_t* buffer = k4a_image_get_buffer(color_k4a_image);
    size_t buffer_size = k4a_image_get_size(color_k4a_image);

    // 转换为OpenCV格式
    if (format == K4A_IMAGE_FORMAT_COLOR_BGRA32) {
        // BGRA32格式
        frame = cv::Mat(height, width, CV_8UC4, buffer);
        // 转换为BGR格式（OpenCV默认格式）
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
    } else if (format == K4A_IMAGE_FORMAT_COLOR_MJPG) {
        // MJPG格式需要解码
        std::vector<uint8_t> buffer_vec(buffer, buffer + buffer_size);
        frame = cv::imdecode(buffer_vec, cv::IMREAD_COLOR);
    } else if (format == K4A_IMAGE_FORMAT_COLOR_YUY2) {
        // YUY2格式
        cv::Mat yuy2_image(height, width, CV_8UC2, buffer);
        cv::cvtColor(yuy2_image, frame, cv::COLOR_YUV2BGR_YUY2);
    } else if (format == K4A_IMAGE_FORMAT_COLOR_NV12) {
        // NV12格式
        cv::Mat nv12_image(height + height/2, width, CV_8UC1, buffer);
        cv::cvtColor(nv12_image, frame, cv::COLOR_YUV2BGR_NV12);
    } else {
        std::cerr << "Unsupported color format: " << format << std::endl;
        k4a_image_release(color_k4a_image);
        return false;
    }

    // 释放K4A图像
    k4a_image_release(color_k4a_image);
    
    return true;
}

bool KinectCameraAdapter::isOpen() const {
    return initialized_;
}

std::string KinectCameraAdapter::getCameraInfo() const {
    if (!initialized_) {
        return "Kinect Camera: Not initialized";
    }
    return "Kinect Camera: " + serial_number_ + " (Device " + std::to_string(device_index_) + ")";
}

bool KinectCameraAdapter::setColorFormat(k4a_image_format_t format) {
    if (initialized_) {
        std::cerr << "Cannot change format after initialization!" << std::endl;
        return false;
    }
    
    config_.color_format = format;
    return true;
}

bool KinectCameraAdapter::setColorResolution(k4a_color_resolution_t resolution) {
    if (initialized_) {
        std::cerr << "Cannot change resolution after initialization!" << std::endl;
        return false;
    }
    
    config_.color_resolution = resolution;
    return true;
}

bool KinectCameraAdapter::setDeviceIndex(uint32_t device_index) {
    if (initialized_) {
        std::cerr << "Cannot change device index after initialization!" << std::endl;
        return false;
    }
    
    device_index_ = device_index;
    return true;
} 