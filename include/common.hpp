#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <string>
#include "camera_interface.hpp"

struct Plane {
    cv::Vec3d point{0.0, 0.0, 0.0};      // 平面上一点 (世界坐标系)
    cv::Vec3d normal{0.0, 1.0, 0.0};     // 平面法向量 (需归一化)
};

struct Config {
    // 相机类型选择
    CameraType cameraType = CameraType::KINECT;  // 默认使用 Kinect 相机
    
    // 相机参数
    cv::Mat K;              // 内参矩阵 3x3
    cv::Mat distCoeffs;     // 畸变系数

    // ArUco 参数
    int   arucoDictId   = cv::aruco::DICT_4X4_50;
    float arucoMarkerLength = 0.05f;  // 单位: m
    double H_marker = 0.0;            // ArUco 中心点离地高度 (m)

    // --- 可选：使用固定的 ArUco -> 相机旋转矩阵 ---
    // 若 useFixedRotation 为 true 且 fixedR 为 3x3 矩阵，则算法将跳过 estimatePoseSingleMarkers
    // 直接使用 fixedR（ArUco 坐标系 → 相机坐标系）进行坐标转换。
    bool useFixedRotation = false;   // 默认为 false（使用 OpenCV 求得的 R）
    cv::Mat fixedR;                  // 3x3, CV_64F

    // 数据记录相关
    bool  recordEnabled = false;      // 是否启用记录功能
    cv::Vec3d originOffset{0.0, 0.0, 0.0}; // 新坐标系原点相对 ArUco 坐标系的偏移
    double launchRPM = 0.0;           // 发射篮球时的转速 (RPM)

    // 记录文件保存目录（可选，默认当前目录）
    std::string recordDir{"."};

    // 轨迹绘制：若连续两帧篮球中心距离超过该值，将拆分轨迹 (像素)
    double maxBallGap = 120.0;

    // HSV 阈值
    cv::Scalar hsvLow{0, 167, 37};
    cv::Scalar hsvHigh{19, 240, 147};

    // 运动平面定义
    Plane motionPlane;

    // ROI 边缘比例 (0~0.5)。可分别设置左右 / 上下。
    double roiLeftMarginRatio   = 0.1;   // 默认左侧去除 10%
    double roiRightMarginRatio  = 0.1;   // 默认右侧去除 10%
    double roiTopMarginRatio    = 0.0;   // 默认不去除顶部
    double roiBottomMarginRatio = 0.0;   // 默认不去除底部

    // ROI 视频录制保存目录
    std::string roiVideoDir{"."};
};

// 从 YAML 文件加载所有配置参数
// 成功返回 true，失败返回 false
bool loadConfig(const std::string &filePath, Config &cfg); 