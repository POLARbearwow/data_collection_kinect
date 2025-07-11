#include "solver.hpp"

#include <opencv2/aruco.hpp>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace cv;
using namespace std;

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

void runTrajectorySolver(const std::string &videoPath, const Config &cfg)
{
    VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        cerr << "无法打开视频文件: " << videoPath << endl;
        return;
    }

    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(cfg.arucoDictId);
    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();

    int frameIdx = 0;
    Mat frame, undistorted;

    while (cap.read(frame))
    {
        ++frameIdx;
        undistort(frame, undistorted, cfg.K, cfg.distCoeffs);

        // ===== 新增: 定义检测 ROI =====
        const int frameW = undistorted.cols;
        const int frameH = undistorted.rows;
        const int leftX   = static_cast<int>(frameW * cfg.roiLeftMarginRatio   + 0.5);
        const int rightX  = static_cast<int>(frameW * cfg.roiRightMarginRatio  + 0.5);
        const int topY    = static_cast<int>(frameH * cfg.roiTopMarginRatio    + 0.5);
        const int bottomY = static_cast<int>(frameH * cfg.roiBottomMarginRatio + 0.5);
        const int roiW = frameW - leftX - rightX;
        const int roiH = frameH - topY - bottomY;
        if (roiW <=0 || roiH <=0) { cerr << "ROI尺寸错误" << endl; break; }
        const cv::Rect roiRect(leftX, topY, roiW, roiH);
        cv::Mat roiFrame = undistorted(roiRect);

        cout << "[LOG] 处理帧 " << frameIdx << endl;

        // ArUco 检测
        vector<int> markerIds;
        vector<vector<Point2f>> corners;
        aruco::detectMarkers(undistorted, dictionary, corners, markerIds, detectorParams);

        cout << "[LOG] ArUco 检测到 " << markerIds.size() << " 个标记" << endl;
        if (!markerIds.empty())
        {
            cout << "[LOG] 标记 ID: ";
            for (size_t i = 0; i < markerIds.size(); ++i)
            {
                cout << markerIds[i] << (i + 1 == markerIds.size() ? "\n" : ", ");
            }
        }

        Mat rvec, tvec;
        bool arucoValid = false;
        if (!markerIds.empty())
        {
            vector<Vec3d> rvecs, tvecs;
            aruco::estimatePoseSingleMarkers(corners, cfg.arucoMarkerLength, cfg.K, cfg.distCoeffs, rvecs, tvecs);

            aruco::drawDetectedMarkers(undistorted, corners, markerIds);

            // 为每个检测到的 ArUco 标记绘制坐标轴并标注其 ID
            for (size_t i = 0; i < markerIds.size(); ++i)
            {
                aruco::drawAxis(undistorted, cfg.K, cfg.distCoeffs, rvecs[i], tvecs[i], cfg.arucoMarkerLength);
                // 在标记左上角位置绘制其 ID
                putText(undistorted,
                        format("ID:%d", markerIds[i]),
                        corners[i][0],
                        FONT_HERSHEY_SIMPLEX,
                        0.6,
                        Scalar(0, 255, 255),
                        2);
            }

            // 新增功能：显示第一个ArUco标记相对于相机的信息
            Vec3d tvec_cam = tvecs[0];
            double dist_cam = norm(tvec_cam);
            String cam_info_pos = format("Aruco Pos @Cam: (%.2f, %.2f, %.2f)m", tvec_cam[0], tvec_cam[1], tvec_cam[2]);
            String cam_info_dist = format("Aruco Dist @Cam: %.2f m", dist_cam);
            putText(undistorted, cam_info_pos, Point(15, undistorted.rows - 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);
            putText(undistorted, cam_info_dist, Point(15, undistorted.rows - 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);

            // 使用第一个标记的位姿进行后续计算
            rvec = (Mat)rvecs[0];
            tvec = (Mat)tvecs[0];
            arucoValid = true;
        }
        if (!arucoValid)
        {
            cout << "[WARN] 本帧未检测到有效 ArUco 标记，跳过后续计算" << endl;
            {
                cv::Mat overlay = undistorted.clone();
                cv::rectangle(overlay, cv::Rect(0,0,leftX, frameH), cv::Scalar(0,0,0), -1);
                cv::rectangle(overlay, cv::Rect(frameW-rightX,0,rightX,frameH), cv::Scalar(0,0,0), -1);
                cv::rectangle(overlay, cv::Rect(leftX,0,roiW, topY), cv::Scalar(0,0,0), -1);
                cv::rectangle(overlay, cv::Rect(leftX, frameH-bottomY, roiW, bottomY), cv::Scalar(0,0,0), -1);
                double alpha=0.5; cv::addWeighted(overlay,alpha,undistorted,1-alpha,0,undistorted);
            }
            imshow("debug", undistorted);
            if (waitKey(1) == 27) break;
            continue;
        }

        // 篮球 HSV 分割 (仅在 ROI 内)
        Mat hsv, mask;
        cvtColor(roiFrame, hsv, COLOR_BGR2HSV);
        inRange(hsv, cfg.hsvLow, cfg.hsvHigh, mask);
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        // 最大轮廓
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        cout << "[LOG] 共检测到 " << contours.size() << " 个候选轮廓" << endl;
        double maxArea = 0; int maxIdx = -1;
        for (size_t i=0;i<contours.size();++i)
        {
            double a = contourArea(contours[i]);
            if (a>maxArea){maxArea=a;maxIdx=i;}
        }
        if (maxIdx<0)
        {
            cout << "[WARN] 未找到满足面积要求的轮廓，跳过" << endl;
            {
                cv::Mat overlay = undistorted.clone();
                cv::rectangle(overlay, cv::Rect(0,0,leftX, frameH), cv::Scalar(0,0,0), -1);
                cv::rectangle(overlay, cv::Rect(frameW-rightX,0,rightX,frameH), cv::Scalar(0,0,0), -1);
                cv::rectangle(overlay, cv::Rect(leftX,0,roiW, topY), cv::Scalar(0,0,0), -1);
                cv::rectangle(overlay, cv::Rect(leftX, frameH-bottomY, roiW, bottomY), cv::Scalar(0,0,0), -1);
                double alpha=0.5; cv::addWeighted(overlay,alpha,undistorted,1-alpha,0,undistorted);
            }
            imshow("debug", undistorted);
            if (waitKey(1)==27) break;
            continue;
        }

        Moments m = moments(contours[maxIdx]);
        Point2f center2D(static_cast<float>(m.m10/m.m00) + roiRect.x,
                         static_cast<float>(m.m01/m.m00) + roiRect.y);
        circle(undistorted, center2D, 5, Scalar(0,0,255), -1);

        cout << fixed << setprecision(2);
        cout << "[LOG] 篮球像素坐标 = (" << center2D.x << ", " << center2D.y << ")" << endl;

        // 坐标转换
        Vec3d rayCam = pixelToCameraRay(center2D, cfg.K);
        Mat Rmat; Rodrigues(rvec, Rmat);
        Matx33d R(Rmat);
        Vec3d rayWorld = R.t()*rayCam;
        Vec3d tvecVec(tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0));
        Vec3d camPosWorld = -(R.t()*tvecVec);

        Vec3d intersection;
        if(intersectRayPlane(camPosWorld, rayWorld, cfg.motionPlane, intersection))
        {
            double height = cfg.H_marker + intersection[0];
            cout << "[LOG] 世界坐标 = (" << intersection[0] << ", " << intersection[1] << ", " << intersection[2] << ") 高度 = " << height << " m" << endl;
            putText(undistorted, format("(%.2f,%.2f,%.2f)m", intersection[0], intersection[1], intersection[2]),
                    Point(15,30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,0,0),2);
            putText(undistorted, format("H=%.2f m", height), Point(15,60), FONT_HERSHEY_SIMPLEX,0.8,Scalar(255,0,0),2);
        }

        // 轮廓偏移后再绘制
        std::vector<std::vector<Point>> contoursShifted = contours;
        for(auto &c: contoursShifted){ for(auto &pt: c) { pt.x += roiRect.x; pt.y += roiRect.y; } }
        drawContours(undistorted, contoursShifted, maxIdx, Scalar(0,255,0),2);

        // ====== 显示前覆盖非检测区域蒙版 ======
        {
            cv::Mat overlay = undistorted.clone();
            cv::rectangle(overlay, cv::Rect(0,0,leftX, frameH), cv::Scalar(0,0,0), -1);
            cv::rectangle(overlay, cv::Rect(frameW-rightX,0,rightX,frameH), cv::Scalar(0,0,0), -1);
            cv::rectangle(overlay, cv::Rect(leftX,0,roiW, topY), cv::Scalar(0,0,0), -1);
            cv::rectangle(overlay, cv::Rect(leftX, frameH-bottomY, roiW, bottomY), cv::Scalar(0,0,0), -1);
            double alpha=0.5; cv::addWeighted(overlay,alpha,undistorted,1-alpha,0,undistorted);
        }
        imshow("debug", undistorted);
        if (waitKey(1)==27) break;
    }
}

//================= 单张图像接口 =================//
void runTrajectorySolverImage(const std::string &imagePath, const Config &cfg)
{
    cv::Mat frame = cv::imread(imagePath);
    if (frame.empty())
    {
        std::cerr << "无法加载图像: " << imagePath << std::endl;
        return;
    }

    cv::Mat undistorted;
    cv::undistort(frame, undistorted, cfg.K, cfg.distCoeffs);

    // ===== 新增: 定义检测 ROI =====
    const int frameW = undistorted.cols;
    const int frameH = undistorted.rows;
    const int leftX   = static_cast<int>(frameW * cfg.roiLeftMarginRatio   + 0.5);
    const int rightX  = static_cast<int>(frameW * cfg.roiRightMarginRatio  + 0.5);
    const int topY    = static_cast<int>(frameH * cfg.roiTopMarginRatio    + 0.5);
    const int bottomY = static_cast<int>(frameH * cfg.roiBottomMarginRatio + 0.5);
    const int roiW = frameW - leftX - rightX;
    const int roiH = frameH - topY - bottomY;
    if (roiW <=0 || roiH <=0) { cerr << "ROI尺寸错误" << endl; return; }
    const cv::Rect roiRect(leftX, topY, roiW, roiH);
    cv::Mat roiFrame = undistorted(roiRect);

    // 复用视频版本中的主要逻辑，只是不循环
    int dummyFrameIdx = 1;
    std::string tmpVideoPath = ""; // 未使用
    std::cout << "[LOG] 处理单张图像" << std::endl;

    // ArUco 检测
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cfg.arucoDictId);
    cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> corners;
    cv::aruco::detectMarkers(undistorted, dictionary, corners, markerIds, detectorParams);
    std::cout << "[LOG] ArUco 检测到 " << markerIds.size() << " 个标记" << std::endl;

    cv::Mat rvec, tvec;
    bool arucoValid = false;
    if (!markerIds.empty())
    {
        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::aruco::estimatePoseSingleMarkers(corners, cfg.arucoMarkerLength, cfg.K, cfg.distCoeffs, rvecs, tvecs);
        cv::aruco::drawDetectedMarkers(undistorted, corners, markerIds);
        for (size_t i = 0; i < markerIds.size(); ++i)
        {
            cv::aruco::drawAxis(undistorted, cfg.K, cfg.distCoeffs, rvecs[i], tvecs[i], cfg.arucoMarkerLength);
            cv::putText(undistorted, cv::format("ID:%d", markerIds[i]), corners[i][0], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,255), 2);
        }

        // 新增功能：显示第一个ArUco标记相对于相机的信息
        cv::Vec3d tvec_cam = tvecs[0];
        double dist_cam = cv::norm(tvec_cam);
        cv::String cam_info_pos = cv::format("Aruco Pos @Cam: (%.2f, %.2f, %.2f)m", tvec_cam[0], tvec_cam[1], tvec_cam[2]);
        cv::String cam_info_dist = cv::format("Aruco Dist @Cam: %.2f m", dist_cam);
        cv::putText(undistorted, cam_info_pos, cv::Point(15, undistorted.rows - 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        cv::putText(undistorted, cam_info_dist, cv::Point(15, undistorted.rows - 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        
        rvec = (cv::Mat)rvecs[0];
        tvec = (cv::Mat)tvecs[0];
        arucoValid = true;
    }
    if (!arucoValid)
    {
        std::cout << "[WARN] 未检测到有效 ArUco 标记，跳过三维解算" << std::endl;
        cv::imshow("result", undistorted);
        cv::waitKey(0);
        return;
    }

    // HSV 分割
    cv::Mat hsv, mask;
    cv::cvtColor(roiFrame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cfg.hsvLow, cfg.hsvHigh, mask);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double maxArea = 0; int maxIdx = -1;
    for (size_t i = 0; i < contours.size(); ++i)
    {
        double a = cv::contourArea(contours[i]);
        if (a > maxArea) { maxArea = a; maxIdx = (int)i; }
    }
    if (maxIdx < 0)
    {
        std::cout << "[WARN] 未找到满足面积要求的轮廓" << std::endl;
        cv::imshow("result", undistorted);
        cv::waitKey(0);
        return;
    }

    cv::Moments m = cv::moments(contours[maxIdx]);
    cv::Point2f center2D(static_cast<float>(m.m10/m.m00) + roiRect.x,
                         static_cast<float>(m.m01/m.m00) + roiRect.y);
    cv::circle(undistorted, center2D, 5, cv::Scalar(0,0,255), -1);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[LOG] 篮球像素坐标 = (" << center2D.x << ", " << center2D.y << ")" << std::endl;

    // 坐标转换
    cv::Vec3d rayCam = pixelToCameraRay(center2D, cfg.K);
    cv::Mat Rmat; cv::Rodrigues(rvec, Rmat);
    cv::Matx33d R(Rmat);
    cv::Vec3d rayWorld = R.t()*rayCam;
    cv::Vec3d tvecVec(tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0));
    cv::Vec3d camPosWorld = -(R.t()*tvecVec);

    cv::Vec3d intersection;
    if (intersectRayPlane(camPosWorld, rayWorld, cfg.motionPlane, intersection))
    {
        double height = cfg.H_marker + intersection[0]; // 默认为 X 轴高度
        std::cout << "[LOG] 世界坐标 = (" << intersection[0] << ", " << intersection[1] << ", " << intersection[2] << ") 高度=" << height << " m" << std::endl;
        cv::putText(undistorted, cv::format("(%.2f,%.2f,%.2f)m", intersection[0], intersection[1], intersection[2]), cv::Point(15,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,0,0),2);
        cv::putText(undistorted, cv::format("H=%.2f m", height), cv::Point(15,60), cv::FONT_HERSHEY_SIMPLEX,0.8, cv::Scalar(255,0,0),2);
    }

    // 轮廓偏移后再绘制
    std::vector<std::vector<cv::Point>> contoursShifted = contours;
    for(auto &c: contoursShifted){ for(auto &pt: c) { pt.x += roiRect.x; pt.y += roiRect.y; } }
    cv::drawContours(undistorted, contoursShifted, maxIdx, cv::Scalar(0,255,0), 2);

    // ====== 显示前覆盖非检测区域蒙版 ======
    {
        cv::Mat overlay = undistorted.clone();
        cv::rectangle(overlay, cv::Rect(0,0,leftX, frameH), cv::Scalar(0,0,0), -1);
        cv::rectangle(overlay, cv::Rect(frameW-rightX,0,rightX,frameH), cv::Scalar(0,0,0), -1);
        cv::rectangle(overlay, cv::Rect(leftX,0,roiW, topY), cv::Scalar(0,0,0), -1);
        cv::rectangle(overlay, cv::Rect(leftX, frameH-bottomY, roiW, bottomY), cv::Scalar(0,0,0), -1);
        double alpha=0.5; cv::addWeighted(overlay,alpha,undistorted,1-alpha,0,undistorted);
    }
    imshow("result", undistorted);
    cv::waitKey(0);
} 