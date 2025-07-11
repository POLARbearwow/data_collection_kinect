#include "solver.hpp"
#include "camera_interface.hpp"

#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>  // 新增：用于计时
#include <deque>  // 用于存储轨迹点
#include <fstream>
#include <ctime>
#include <filesystem>
#include <opencv2/video.hpp>
#include <thread>
#include <atomic>
#include "safe_queue.hpp"

using namespace cv;
using namespace std;
using namespace std::chrono;  // 新增：用于计时

// 定义一组不同的轨迹颜色
const std::vector<cv::Scalar> TRAJECTORY_COLORS = {
    cv::Scalar(0, 255, 255),   // 黄色
    cv::Scalar(255, 0, 255),   // 洋红
    cv::Scalar(0, 255, 0),     // 绿色
    cv::Scalar(255, 128, 0),   // 橙色
    cv::Scalar(255, 255, 0),   // 青色
    cv::Scalar(128, 0, 255)    // 紫色
};

// 像素坐标 -> 相机坐标系射线
static Vec3d pixelToCameraRay(const Point2f &uv, const Mat &K)
{
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    Vec3d dir((uv.x - cx) / fx, (uv.y - cy) / fy, 1.0);
    return normalize(dir);
}

// 射线与平面求交
static bool intersectRayPlane(const Vec3d &origin, const Vec3d &dir,
                              const Plane &plane, Vec3d &intersection)
{
    double denom = plane.normal.dot(dir);
    if (fabs(denom) < 1e-6) return false;
    double t = plane.normal.dot(plane.point - origin) / denom;
    if (t < 0) return false;
    intersection = origin + t * dir;
    return true;
}

// ===== 数据包定义 =====
struct PixelPkt { uint64_t id; int64_t ts; cv::Point2f ballUV; cv::Point2f arucoUV; double rpm; };
struct WorldPkt { uint64_t id; int64_t ts; cv::Vec3d xyz; double height; };

static SafeQueue<PixelPkt> gPixQueue;
static SafeQueue<WorldPkt> gWorldQueue;

//================= 实时相机接口 =================//
void runTrajectorySolverCamera(const Config &cfg)
{
    // 根据配置创建相机实例
    auto camera = CameraFactory::createCamera(cfg.cameraType);
    if (!camera) {
        std::cerr << "[ERROR] Failed to create camera instance" << std::endl;
        return;
    }
    
    if(!camera->openCamera())
    {
        std::cerr << "[ERROR] Failed to open camera" << std::endl;
        return;
    }
    
    std::cout << "[INFO] Camera info: " << camera->getCameraInfo() << std::endl;

    // 背景减除器，用于滤除静态背景，只保留运动目标
    cv::Ptr<cv::BackgroundSubtractor> bgSub = cv::createBackgroundSubtractorMOG2(500, /*varThreshold*/16, /*detectShadows*/true);

    cv::Mat frame, undistorted;
    int frameIdx = 0;
    
    // 记录相关
    std::ofstream recordFile;
    bool recording = false;
    int  sessionIndex = 0;  // 用于文件命名

    // ROI 视频录制相关
    bool roiRec = false;
    cv::VideoWriter roiWriter;

    // 用于控制日志输出频率
    auto lastLogTime = steady_clock::now();
    const auto logInterval = milliseconds(500);  // 0.5秒
    bool shouldLog = false;

    // 检测功能开关
    bool detectEnabled = true;  // 默认开启检测
    const string windowName = "Basketball Detection";
    cv::namedWindow(windowName);
    // 添加新的窗口用于显示处理过程
    // const string binaryWindowName = "Binary Process"; // 已禁用
    // cv::namedWindow(binaryWindowName);

    // 新增：运动掩码窗口
    const string motionWindowName = "Motion Mask";
    cv::namedWindow(motionWindowName);

    // 存储轨迹点和对应的颜色索引
    struct TrajectorySegment {
        std::deque<cv::Point2f> points;
        size_t colorIndex;
    };
    std::vector<TrajectorySegment> trajectorySegments;
    trajectorySegments.push_back({std::deque<cv::Point2f>(), 0});
    const size_t maxTrajectoryPoints = 1000;

    // 添加面积阈值
    const double MIN_CONTOUR_AREA = 80.0;  // 最小轮廓面积阈值
    double currentMaxArea = 0.0;  // 用于显示当前最大轮廓面积

    // 创建形态学操作的核
    cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    
    // 创建一个用于显示处理过程的图像
    cv::Mat processViz;

    // === 多线程写 CSV ===
    std::atomic<bool> stopFlag{false};

    // CSV 文件  —— pixels 带时间戳命名
    std::shared_ptr<std::ofstream> pixelCsvPtr; // 按检测会话创建
    std::mutex pixelCsvMtx;
    std::ofstream worldCsv(cfg.recordDir + "/world_coords.csv");
    worldCsv << "frame_id,timestamp_ms,X,Y,Z,H\n";

    // pixel 写线程
    std::thread pixWriter([&]{
        PixelPkt pkt;
        while(!stopFlag){
            if(gPixQueue.pop(pkt)){
                std::lock_guard<std::mutex> lock(pixelCsvMtx);
                if (pixelCsvPtr) {
                    (*pixelCsvPtr) << pkt.id << "," << pkt.ts << "," << pkt.ballUV.x << "," << pkt.ballUV.y << "," << pkt.arucoUV.x << "," << pkt.arucoUV.y << "," << pkt.rpm << "\n";
                }
            }
        }
    });

    // world 写线程
    std::thread worldWriter([&]{
        WorldPkt wp;
        while(!stopFlag){
            if(gWorldQueue.pop(wp)){
                worldCsv << wp.id << "," << wp.ts << "," << wp.xyz[0] << "," << wp.xyz[1] << "," << wp.xyz[2] << "," << wp.height << "\n";
            }
        }
    });

    while (true)
    {
        if(!camera->getFrame(frame))
        {
            std::cerr << "[WARN] Failed to get camera frame, retrying..." << std::endl;
            continue;
        }

        if (frameIdx == 0)
        {
            std::cout << "[DEBUG] Actual camera frame size: "
                      << frame.cols << "x" << frame.rows << std::endl;
        }

        ++frameIdx;
        
        auto now = steady_clock::now();
        if (now - lastLogTime >= logInterval) {
            shouldLog = true;
            lastLogTime = now;
        } else {
            shouldLog = false;
        }

        cv::undistort(frame, undistorted, cfg.K, cfg.distCoeffs);

        // 本帧时间戳(ms)
        int64_t frameTs = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count();

        // after frameTs declaration
        cv::Point2f arucoUV(-999.f,-999.f);
        cv::Point2f ballUV(-999.f,-999.f);

        // ===== ROI 计算：根据配置左右边缘比例 =====
        const int frameW = undistorted.cols;
        const int frameH = undistorted.rows;
        const int leftX   = static_cast<int>(frameW * cfg.roiLeftMarginRatio   + 0.5);
        const int rightX  = static_cast<int>(frameW * cfg.roiRightMarginRatio  + 0.5);
        const int topY    = static_cast<int>(frameH * cfg.roiTopMarginRatio    + 0.5);
        const int bottomY = static_cast<int>(frameH * cfg.roiBottomMarginRatio + 0.5);

        const int roiW = frameW - leftX - rightX;
        const int roiH = frameH - topY - bottomY;

        if (roiW <= 0 || roiH <= 0) {
            std::cerr << "[ERROR] ROI 尺寸为非正值，请检查 margin 配置" << std::endl;
            break;
        }

        const cv::Rect roiRect(leftX, topY, roiW, roiH);
        cv::Mat roiFrame = undistorted(roiRect);

        // ===== 背景减除，获取前景运动区域 (仅处理 ROI) =====
        cv::Mat fgMask;
        bgSub->apply(roiFrame, fgMask);
        // 对前景掩码进行简单形态学处理，去除噪声
        cv::erode(fgMask, fgMask, morphKernel, cv::Point(-1,-1), 1);
        cv::dilate(fgMask, fgMask, morphKernel, cv::Point(-1,-1), 2);

        // 显示检测状态
        string statusText = detectEnabled ? "Detection: ON [Space]" : "Detection: OFF [Space]";
        cv::putText(undistorted, statusText, 
                   cv::Point(15, undistorted.rows - 90),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                   detectEnabled ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

        // 显示清除轨迹的提示
        cv::putText(undistorted, "Press 'C' to clear trajectory", 
                   cv::Point(15, undistorted.rows - 120),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                   cv::Scalar(255, 255, 255), 2);

        // 显示 ROI 录制状态
        string recText = roiRec ? "ROI Rec: ON [V]" : "ROI Rec: OFF [V]";
        cv::putText(undistorted, recText,
                   cv::Point(15, undistorted.rows - 60),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6,
                   roiRec ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), 2);

        // 持续进行ArUco检测
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> corners;
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cfg.arucoDictId);
        cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();
        cv::aruco::detectMarkers(undistorted, dictionary, corners, markerIds, detectorParams);

        bool hasValidAruco = false;
        std::vector<cv::Vec3d> rvecs, tvecs;
        
        if (!markerIds.empty())
        {
            hasValidAruco = true;
            cv::aruco::estimatePoseSingleMarkers(corners, cfg.arucoMarkerLength, cfg.K, cfg.distCoeffs, rvecs, tvecs);
            
            // 绘制检测到的ArUco标记
            cv::aruco::drawDetectedMarkers(undistorted, corners, markerIds);
            
            for (size_t i = 0; i < markerIds.size(); ++i)
            {
                // 绘制坐标轴
                cv::aruco::drawAxis(undistorted, cfg.K, cfg.distCoeffs, rvecs[i], tvecs[i], cfg.arucoMarkerLength);
                
                // 在标记上显示ID
                cv::putText(undistorted, 
                          cv::format("ArUco ID: %d", markerIds[i]), 
                          corners[i][0], 
                          cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                          cv::Scalar(0,255,255), 2);

                // 显示ArUco标记的3D位置
                cv::Vec3d tvec = tvecs[i];
                cv::putText(undistorted,
                          cv::format("ArUco %d Pos: (%.2f, %.2f, %.2f)m", 
                                   markerIds[i], tvec[0], tvec[1], tvec[2]),
                          cv::Point(15, 30 + i * 30),
                          cv::FONT_HERSHEY_SIMPLEX, 0.6,
                          cv::Scalar(255,255,0), 2);
            }
        }

        // ------ 每帧写入像素数据（ArUco 优先） ------
        if(hasValidAruco){
            cv::Point2f sum(0,0);
            for(const auto &pt: corners[0]) sum += pt;
            arucoUV = sum * 0.25f;
        }

        // 篮球检测
        bool hasValidBall = false;
        cv::Point2f center2D;
        cv::Mat hsv, mask, morphed;
        cv::cvtColor(roiFrame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cfg.hsvLow, cfg.hsvHigh, mask);

        // 形态学操作
        cv::erode(mask, morphed, morphKernel, cv::Point(-1,-1), 2);
        cv::dilate(morphed, morphed, morphKernel, cv::Point(-1,-1), 2);

        // 创建可视化图像
        const int vizWidth = mask.cols * 2;
        const int vizHeight = mask.rows;
        processViz = cv::Mat::zeros(vizHeight, vizWidth, CV_8UC3);

        // 转换掩码为彩色图像以便显示
        cv::Mat maskViz, morphedViz;
        cv::cvtColor(mask, maskViz, cv::COLOR_GRAY2BGR);
        cv::cvtColor(morphed, morphedViz, cv::COLOR_GRAY2BGR);

        // 在可视化图像中并排显示原始掩码和处理后的图像
        maskViz.copyTo(processViz(cv::Rect(0, 0, mask.cols, mask.rows)));
        morphedViz.copyTo(processViz(cv::Rect(mask.cols, 0, mask.cols, mask.rows)));

        // ===== 颜色掩码与运动前景掩码合并 =====
        cv::Mat movingMask;
        cv::bitwise_and(morphed, fgMask, movingMask);
        // 显示运动掩码
        cv::imshow(motionWindowName, movingMask);

        // 添加标题
        cv::putText(processViz, "Original Mask", 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 
                   0.8, cv::Scalar(0,255,0), 2);
        cv::putText(processViz, "After Morphology", 
                   cv::Point(mask.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 
                   0.8, cv::Scalar(0,255,0), 2);

        // 轮廓检测（始终进行）
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(movingMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        double maxArea = 0; 
        int maxIdx = -1;
        
        // 计算最大面积的轮廓
        for (size_t i = 0; i < contours.size(); ++i)
        {
            double area = cv::contourArea(contours[i]);
            if (area > maxArea) {
                maxArea = area;
                maxIdx = (int)i;
            }
        }

        currentMaxArea = maxArea;  // 更新当前最大面积

        // 显示当前最大轮廓面积
        cv::putText(undistorted, 
                   cv::format("Max Contour Area: %.1f", currentMaxArea),
                   cv::Point(15, undistorted.rows - 150),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6,
                   currentMaxArea >= MIN_CONTOUR_AREA ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
                   2);

        // 显示面积阈值
        cv::putText(undistorted,
                   cv::format("Area Threshold: %.1f", MIN_CONTOUR_AREA),
                   cv::Point(15, undistorted.rows - 180),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6,
                   cv::Scalar(255, 255, 255),
                   2);

        // ===== 定义检测 ROI: 根据配置左右各去除一定比例 =====
        // (ROI 参数已在本循环开始处计算, 此处无需重复定义)

        if (detectEnabled && maxIdx >= 0 && maxArea >= MIN_CONTOUR_AREA)
        {
            hasValidBall = true;
            // 计算篮球中心点
            cv::Moments m = cv::moments(contours[maxIdx]);
            center2D = cv::Point2f(static_cast<float>(m.m10/m.m00) + roiRect.x,
                                   static_cast<float>(m.m01/m.m00) + roiRect.y);
            ballUV = center2D;
            
            // 添加新的轨迹点到当前段（带距离检测）
            auto& currentSegment = trajectorySegments.back();
            bool startNewSegment = false;
            if (!currentSegment.points.empty()) {
                double distGap = cv::norm(center2D - currentSegment.points.back());
                if (distGap > cfg.maxBallGap) {
                    startNewSegment = true;
                }
            }

            if (startNewSegment) {
                // 仅切分轨迹，不改变颜色
                trajectorySegments.push_back({std::deque<cv::Point2f>(), currentSegment.colorIndex});
                if (shouldLog) {
                    std::cout << "[INFO] Large ball gap " << cfg.maxBallGap << "px exceeded. Start new segment (same color)." << std::endl;
                }
            }

            auto& segToAdd = trajectorySegments.back();
            segToAdd.points.push_back(center2D);
            if (segToAdd.points.size() > maxTrajectoryPoints) {
                segToAdd.points.pop_front();
            }

            // 绘制当前篮球位置
            cv::circle(undistorted, center2D, 5, cv::Scalar(0,0,255), -1);
            // 绘制外接矩形
            cv::Rect bbox = cv::boundingRect(contours[maxIdx]);
            bbox.x += roiRect.x;
            bbox.y += roiRect.y;
            cv::rectangle(undistorted, bbox, cv::Scalar(0, 255, 255), 2);

            // 将轮廓坐标平移后绘制
            std::vector<std::vector<cv::Point>> contoursShifted = contours;
            for (auto &c : contoursShifted) for (auto &pt : c) {
                pt.x += roiRect.x;
                pt.y += roiRect.y;
            }
            cv::drawContours(undistorted, contoursShifted, maxIdx, cv::Scalar(0,255,0), 2);
            
            // 显示篮球像素坐标
            cv::putText(undistorted, 
                       cv::format("Ball Pixel Pos: (%.1f, %.1f)", center2D.x, center2D.y),
                       cv::Point(15, undistorted.rows - 150),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,255), 2);
        }

        // 当同时检测到篮球和ArUco标记时，进行坐标解算
        if (hasValidBall && hasValidAruco)
        {
            // 计算并显示世界坐标
            cv::Matx33d R;

            // 使用固定旋转矩阵
            if(cfg.useFixedRotation && !cfg.fixedR.empty()) {
                cv::Mat tmp = cfg.fixedR;
                if(tmp.rows == 3 && tmp.cols == 3) {
                    // 将 cv::Mat 转为 Matx33d
                    R = cv::Matx33d(
                        tmp.at<double>(0,0), tmp.at<double>(0,1), tmp.at<double>(0,2),
                        tmp.at<double>(1,0), tmp.at<double>(1,1), tmp.at<double>(1,2),
                        tmp.at<double>(2,0), tmp.at<double>(2,1), tmp.at<double>(2,2)
                    );
                } else {
                    // 维度不符，回退到动态求解
                    cv::Mat Rmat;
                    cv::Rodrigues(rvecs[0], Rmat);
                    R = cv::Matx33d(Rmat);
                }
            } else {
                cv::Mat Rmat;
                cv::Rodrigues(rvecs[0], Rmat);
                R = cv::Matx33d(Rmat);
            }
            cv::Vec3d rayWorld = R.t()*pixelToCameraRay(center2D, cfg.K);
            cv::Vec3d tvecVec(tvecs[0][0], tvecs[0][1], tvecs[0][2]);
            cv::Vec3d camPosWorld = -(R.t()*tvecVec);

            cv::Vec3d intersection;
            if(intersectRayPlane(camPosWorld, rayWorld, cfg.motionPlane, intersection))
            {
                double height = cfg.H_marker + intersection[0];
                if (shouldLog) {
                    std::cout << "[LOG] World coordinates = (" 
                             << intersection[0] << ", " 
                             << intersection[1] << ", " 
                             << intersection[2] << ") Height = " 
                             << height << " m" << std::endl;
                }
                
                // 在图像上标注坐标解算结果（使用 ArUco 坐标系）
                cv::putText(undistorted, 
                          cv::format("Ball ArUco Pos: (%.2f, %.2f, %.2f)m", 
                                   intersection[0], intersection[1], intersection[2]),
                          cv::Point(15, undistorted.rows - 300),
                          cv::FONT_HERSHEY_SIMPLEX, 1.2,
                          cv::Scalar(255,0,0), 2);
                cv::putText(undistorted, 
                          cv::format("Ball Height H: %.2f m", height),
                          cv::Point(15, undistorted.rows - 270),
                          cv::FONT_HERSHEY_SIMPLEX, 1.2,
                          cv::Scalar(255,0,0), 2);

                // 添加状态指示
                cv::putText(undistorted,
                          "Status: Ball & ArUco detected - Computing coordinates",
                          cv::Point(15, undistorted.rows - 240),
                          cv::FONT_HERSHEY_SIMPLEX, 0.6,
                          cv::Scalar(0,255,0), 2);

                // ---- 数据记录 ----
                if (recording && recordFile.is_open()) {
                    // 获取时间戳 (ms)
                    auto now_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();

                    // 计算新坐标系下的坐标 (减去偏移量)
                    cv::Vec3d newPos = intersection - cfg.originOffset;

                    recordFile << frameIdx << "," << now_ts << ","
                               << intersection[0] << "," << intersection[1] << "," << intersection[2] << ","
                               << newPos[0]       << "," << newPos[1]       << "," << newPos[2] << ","
                               << height << ","
                               << cfg.launchRPM << std::endl;
                }

                // 推送 world 数据包供写线程
                gWorldQueue.push(WorldPkt{static_cast<uint64_t>(frameIdx), frameTs, intersection, height});
            }
        }
        else
        {
            // 显示当前检测状态
            std::string status = "Status: ";
            if (!hasValidBall && !hasValidAruco) status += "No Ball & No ArUco detected";
            else if (!hasValidBall) status += "No Ball detected";
            else if (!hasValidAruco) status += "No ArUco detected";
            
            cv::putText(undistorted,
                      status,
                      cv::Point(15, undistorted.rows - 240),
                      cv::FONT_HERSHEY_SIMPLEX, 0.6,
                      cv::Scalar(0,0,255), 2);
        }

        // ---- 始终绘制所有轨迹段，以便丢失目标后仍显示历史轨迹 ----
        for (const auto& segment : trajectorySegments) {
            const cv::Scalar& color = TRAJECTORY_COLORS[segment.colorIndex];
            for (size_t i = 1; i < segment.points.size(); ++i) {
                cv::line(undistorted,
                        segment.points[i-1],
                        segment.points[i],
                        color,
                        2);
            }
        }

        // ===== 在非检测区域覆盖半透明蒙版 =====
        {
            cv::Mat overlay = undistorted.clone();
            // 左右遮罩
            cv::rectangle(overlay, cv::Rect(0, 0, leftX, frameH), cv::Scalar(0,0,0), -1);
            cv::rectangle(overlay, cv::Rect(frameW - rightX, 0, rightX, frameH), cv::Scalar(0,0,0), -1);
            // 上下遮罩
            cv::rectangle(overlay, cv::Rect(leftX, 0, roiW, topY), cv::Scalar(0,0,0), -1);
            cv::rectangle(overlay, cv::Rect(leftX, frameH - bottomY, roiW, bottomY), cv::Scalar(0,0,0), -1);

            double alpha = 0.5;
            cv::addWeighted(overlay, alpha, undistorted, 1 - alpha, 0, undistorted);
        }

        // ---- ROI 视频录制 ----
        if (roiRec && roiWriter.isOpened()) {
            roiWriter.write(frame); // 录制摄像头原始画面
        }

        // 显示图像和检查按键
        cv::imshow(windowName, undistorted);
        int key = cv::waitKey(1);
        if (key == 27) {
            // cv::destroyWindow(binaryWindowName); // 已禁用二值窗口
            cv::destroyWindow(motionWindowName);
            break;        // ESC: exit program
        }
        if (key == 32) {
            detectEnabled = !detectEnabled;  // Space: toggle detection only

            // ---- pixels.csv 会话管理 ----
            {
                std::lock_guard<std::mutex> lock(pixelCsvMtx);
                if (detectEnabled) {
                    // 开启新 pixels.csv
                    if (pixelCsvPtr && pixelCsvPtr->is_open()) pixelCsvPtr->close();
                    char fnameP[128];
                    std::time_t tP = std::time(nullptr);
                    std::tm *tmPtrP = std::localtime(&tP);
                    std::strftime(fnameP, sizeof(fnameP), "pixels_%Y%m%d_%H%M%S.csv", tmPtrP);
                    std::string ppath = cfg.recordDir + "/" + fnameP;
                    pixelCsvPtr = std::make_shared<std::ofstream>(ppath);
                    if (pixelCsvPtr->is_open()) {
                        (*pixelCsvPtr) << "frame_id,timestamp_ms,ball_u,ball_v,aruco_u,aruco_v,rpm\n";
                        std::cout << "[INFO] Start pixels csv: " << ppath << std::endl;
                    }
                } else {
                    if (pixelCsvPtr && pixelCsvPtr->is_open()) {
                        pixelCsvPtr->close();
                        std::cout << "[INFO] pixels csv closed." << std::endl;
                    }
                    pixelCsvPtr.reset();
                }
            }

            // 切换检测状态时，处理记录文件开关
            if (cfg.recordEnabled)
            {
                if (detectEnabled) {
                    // 开始新的记录会话
                    if (recordFile.is_open()) recordFile.close();

                    namespace fs = std::filesystem;
                    try {
                        fs::create_directories(cfg.recordDir);
                    } catch (const std::exception &e) {
                        std::cerr << "[ERROR] Failed to create record directory: " << e.what() << std::endl;
                    }

                    char filename[128];
                    std::time_t t = std::time(nullptr);
                    std::tm *tm_ptr = std::localtime(&t);
                    std::strftime(filename, sizeof(filename), "record_%Y%m%d_%H%M%S.csv", tm_ptr);
                    std::string filepath = cfg.recordDir + "/" + filename;

                    recordFile.open(filepath);
                    if (recordFile.is_open()) {
                        recording = true;
                        std::cout << "[INFO] Start recording to " << filepath << std::endl;
                        // 写入表头
                        recordFile << "frame_id,timestamp_ms,x,y,z,new_x,new_y,new_z,height_m,rpm" << std::endl;
                    } else {
                        std::cerr << "[ERROR] Failed to open record file: " << filepath << std::endl;
                    }
                    ++sessionIndex;
                }
                else {
                    if (recordFile.is_open()) {
                        std::cout << "[INFO] Stop recording, file closed." << std::endl;
                        recordFile.close();
                    }
                    recording = false;
                }
            }

            if (detectEnabled) {
                size_t newColorIndex = (trajectorySegments.back().colorIndex + 1) % TRAJECTORY_COLORS.size();
                trajectorySegments.push_back({std::deque<cv::Point2f>(), newColorIndex});
            }
        }
        if (key == 'c' || key == 'C') {
            trajectorySegments.clear();
            trajectorySegments.push_back({std::deque<cv::Point2f>(), 0});
            if (shouldLog) {
                std::cout << "[INFO] Trajectory cleared" << std::endl;
            }
        }
        if (key == 'v' || key == 'V') {
            roiRec = !roiRec;
            if (roiRec) {
                // 开启录制
                char fname[128];
                std::time_t t = std::time(nullptr);
                std::tm *tm_ptr = std::localtime(&t);
                std::strftime(fname, sizeof(fname), "roi_%Y%m%d_%H%M%S.avi", tm_ptr);
                std::string filepath = cfg.roiVideoDir + "/" + fname;
                // 确保目录存在
                namespace fs = std::filesystem;
                try { fs::create_directories(cfg.roiVideoDir); } catch (...) {}
                int fourcc = cv::VideoWriter::fourcc('F','F','V','1');
                double fps = 30; // 0 让容器决定; 若需固定可设置实际摄像头帧率
                roiWriter.open(filepath, fourcc, fps, cv::Size(frame.cols, frame.rows));
                if (!roiWriter.isOpened()) {
                    // 尝试 MJPG + 质量 100 作为退备
                    fourcc = cv::VideoWriter::fourcc('M','J','P','G');
                    fps = 30.0;
                    roiWriter.open(filepath, fourcc, fps, cv::Size(frame.cols, frame.rows));
                    if (roiWriter.isOpened()) {
                        roiWriter.set(cv::VIDEOWRITER_PROP_QUALITY, 100);
                    }
                }
                if (!roiWriter.isOpened()) {
                    std::cerr << "[ERROR] 无法打开 ROI 视频文件 " << filepath << std::endl;
                    roiRec = false;
                } else {
                    std::cout << "[INFO] 开始 ROI 录制: " << filepath << std::endl;
                }
            } else {
                // 关闭录制
                if (roiWriter.isOpened()) {
                    roiWriter.release();
                    std::cout << "[INFO] 停止 ROI 录制" << std::endl;
                }
            }
        }

        // ---- push PixelPkt at end of frame ----
        if(detectEnabled){
            gPixQueue.push(PixelPkt{static_cast<uint64_t>(frameIdx), frameTs, ballUV, arucoUV, cfg.launchRPM});
        }
    }

    camera->closeCamera();

    if (roiWriter.isOpened()) {
        roiWriter.release();
        std::cout << "[INFO] ROI 录制文件已关闭" << std::endl;
    }

    if (recordFile.is_open()) {
        std::cout << "[INFO] Stop recording, file closed." << std::endl;
        recordFile.close();
    }

    // 通知写线程退出
    stopFlag = true;
    gPixQueue.push(PixelPkt{});   // 唤醒
    gWorldQueue.push(WorldPkt{});
    pixWriter.join();
    worldWriter.join();
    pixelCsvPtr.reset();
    worldCsv.close();
} 