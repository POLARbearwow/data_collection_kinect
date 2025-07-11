#include "common.hpp"
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>

bool loadConfig(const std::string &filePath, Config &cfg)
{
    cv::FileStorage fs(filePath, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "无法打开配置文件: " << filePath << std::endl;
        return false;
    }

    // 读取相机类型配置
    std::string cameraTypeStr;
    if (!fs["camera_type"].empty()) {
        fs["camera_type"] >> cameraTypeStr;
        if (cameraTypeStr == "kinect" || cameraTypeStr == "KINECT") {
            cfg.cameraType = CameraType::KINECT;
        } else {
            std::cerr << "警告: 未知的相机类型 '" << cameraTypeStr << "', 使用默认值 KINECT" << std::endl;
            cfg.cameraType = CameraType::KINECT;
        }
    }

    fs["camera_matrix"] >> cfg.K;
    fs["dist_coeffs"] >> cfg.distCoeffs;
    fs["aruco_dict_id"] >> cfg.arucoDictId;
    fs["aruco_marker_length"] >> cfg.arucoMarkerLength;
    fs["H_marker"] >> cfg.H_marker;

    // 可选：固定 ArUco -> 相机旋转矩阵
    if (!fs["use_fixed_rotation"].empty()) {
        int flag = 0;
        fs["use_fixed_rotation"] >> flag;
        cfg.useFixedRotation = (flag != 0);
    }

    // 读取固定旋转矩阵 (3x3)
    cv::Mat fixedRMat;
    fs["fixed_R"] >> fixedRMat;
    if (!fixedRMat.empty()) {
        cfg.fixedR = fixedRMat.clone();
    }

    // 数据记录相关参数（可选）
    if (!fs["record_enabled"].empty()) {
        int recEnabledInt = 0;
        fs["record_enabled"] >> recEnabledInt;
        cfg.recordEnabled = (recEnabledInt != 0);
    }

    cv::FileNode offsetNode = fs["origin_offset"];
    if (!offsetNode.empty()) {
        cv::Mat offsetMat;
        offsetNode >> offsetMat;
        if (offsetMat.total() == 3) {
            cfg.originOffset = cv::Vec3d(offsetMat.at<double>(0), offsetMat.at<double>(1), offsetMat.at<double>(2));
        }
    }

    if (!fs["launch_rpm"].empty()) {
        fs["launch_rpm"] >> cfg.launchRPM;
    }

    cv::FileNode hsv = fs["hsv_range"];
    if (!hsv.empty())
    {
        hsv["low"] >> cfg.hsvLow;
        hsv["high"] >> cfg.hsvHigh;
    }
    else {
        // 兼容新格式：hsv_low / hsv_high 直接在根节点
        cv::Mat lowMat, highMat;
        fs["hsv_low"] >> lowMat;
        fs["hsv_high"] >> highMat;

        if (!lowMat.empty() && lowMat.total() == 3)
        {
            // 支持 int 或 double 类型
            if (lowMat.type() == CV_64F)
            {
                cfg.hsvLow = cv::Scalar(lowMat.at<double>(0), lowMat.at<double>(1), lowMat.at<double>(2));
            }
            else
            {
                cfg.hsvLow = cv::Scalar(lowMat.at<int>(0), lowMat.at<int>(1), lowMat.at<int>(2));
            }
        }

        if (!highMat.empty() && highMat.total() == 3)
        {
            if (highMat.type() == CV_64F)
            {
                cfg.hsvHigh = cv::Scalar(highMat.at<double>(0), highMat.at<double>(1), highMat.at<double>(2));
            }
            else
            {
                cfg.hsvHigh = cv::Scalar(highMat.at<int>(0), highMat.at<int>(1), highMat.at<int>(2));
            }
        }
    }

    // 尝试读取旧格式的运动平面参数
    cv::Mat planePoint, planeNormal;
    fs["motion_plane_point"] >> planePoint;
    fs["motion_plane_normal"] >> planeNormal;
    
    if (!planePoint.empty() && !planeNormal.empty())
    {
        // 使用旧格式的参数
        cfg.motionPlane.point = cv::Vec3d(planePoint.at<double>(0), planePoint.at<double>(1), planePoint.at<double>(2));
        cfg.motionPlane.normal = cv::normalize(cv::Vec3d(planeNormal.at<double>(0), planeNormal.at<double>(1), planeNormal.at<double>(2)));
    }
    else
    {
        // 尝试读取新格式的运动平面参数
        cv::FileNode plane = fs["motion_plane"];
        if (!plane.empty())
        {
            cv::Mat p, n;
            plane["point"] >> p;
            plane["normal"] >> n;
            
            if (!p.empty() && !n.empty())
            {
                cfg.motionPlane.point = cv::Vec3d(p.at<double>(0), p.at<double>(1), p.at<double>(2));
                cv::Vec3d normal(n.at<double>(0), n.at<double>(1), n.at<double>(2));
                double norm = cv::norm(normal);
                if (norm < 1e-6)
                {
                    std::cerr << "警告: 平面法向量接近零向量，使用默认值 [0,0,1]" << std::endl;
                    cfg.motionPlane.normal = cv::Vec3d(0, 0, 1);
                }
                else
                {
                    cfg.motionPlane.normal = normal / norm;
                }
            }
            else
            {
                std::cerr << "警告: motion_plane 参数不完整，使用默认值" << std::endl;
            }
        }
        else
        {
            std::cerr << "警告: 未找到运动平面参数，使用默认值" << std::endl;
        }
    }

    if (!fs["record_dir"].empty()) {
        fs["record_dir"] >> cfg.recordDir;
    }

    if (!fs["max_ball_gap"].empty()) {
        fs["max_ball_gap"] >> cfg.maxBallGap;
    }

    // ROI 视频保存目录
    if (!fs["roi_video_dir"].empty()) {
        fs["roi_video_dir"] >> cfg.roiVideoDir;
    }

    // ROI 边缘比例 (可选) —— 新格式
    auto clampRatio = [](double &v){
        if (v < 0.0) v = 0.0;
        else if (v > 0.49) v = 0.49;
    };

    if (!fs["roi_left_margin_ratio"].empty())  fs["roi_left_margin_ratio"]  >> cfg.roiLeftMarginRatio;
    if (!fs["roi_right_margin_ratio"].empty()) fs["roi_right_margin_ratio"] >> cfg.roiRightMarginRatio;
    if (!fs["roi_top_margin_ratio"].empty())   fs["roi_top_margin_ratio"]   >> cfg.roiTopMarginRatio;
    if (!fs["roi_bottom_margin_ratio"].empty())fs["roi_bottom_margin_ratio"]>> cfg.roiBottomMarginRatio;

    clampRatio(cfg.roiLeftMarginRatio);
    clampRatio(cfg.roiRightMarginRatio);
    clampRatio(cfg.roiTopMarginRatio);
    clampRatio(cfg.roiBottomMarginRatio);

    // 兼容旧格式 roi_side_margin_ratio：若新键未指定，则使用旧值
    if (!fs["roi_side_margin_ratio"].empty()) {
        double oldRatio = 0.0;
        fs["roi_side_margin_ratio"] >> oldRatio;
        clampRatio(oldRatio);
        // 仅当左右未在新格式中显式给出时，才使用旧值
        if (fs["roi_left_margin_ratio"].empty())  cfg.roiLeftMarginRatio  = oldRatio;
        if (fs["roi_right_margin_ratio"].empty()) cfg.roiRightMarginRatio = oldRatio;
    }

    fs.release();
    return true;
} 