#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

// 相机类型枚举
enum class CameraType {
    HIK,    // 海康威视相机
    KINECT  // Azure Kinect相机
};

// 抽象相机接口
class CameraInterface {
public:
    virtual ~CameraInterface() = default;
    
    // 打开相机
    virtual bool openCamera() = 0;
    
    // 关闭相机
    virtual void closeCamera() = 0;
    
    // 获取一帧图像
    virtual bool getFrame(cv::Mat& frame) = 0;
    
    // 检查相机是否已打开
    virtual bool isOpen() const = 0;
    
    // 获取相机信息
    virtual std::string getCameraInfo() const = 0;
};

// 相机工厂类
class CameraFactory {
public:
    static std::unique_ptr<CameraInterface> createCamera(CameraType type);
}; 