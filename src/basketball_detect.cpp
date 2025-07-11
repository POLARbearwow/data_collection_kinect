#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <string>
#include "common.hpp"
#include "utils.hpp"
#include "detect.hpp"
#include <unistd.h>
#include <filesystem>

using namespace cv;
using namespace std;

void runBasketballDetect(const std::string &videoPath, const Config &cfg)
{
    VideoCapture cap(videoPath);
    if(!cap.isOpened())
    {
        cerr << "无法打开视频文件: " << videoPath << endl;
        return;
    }

    int frameIdx = 0;
    Mat frame, undistorted;

    while(cap.read(frame))
    {
        ++frameIdx;
        undistort(frame, undistorted, cfg.K, cfg.distCoeffs);

        // HSV 分割
        Mat hsv, mask;
        cvtColor(undistorted, hsv, COLOR_BGR2HSV);
        inRange(hsv, cfg.hsvLow, cfg.hsvHigh, mask);
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        // 寻找最大轮廓
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        double maxArea = 0; int maxIdx = -1;
        for(size_t i=0;i<contours.size();++i)
        {
            double a = contourArea(contours[i]);
            if(a>maxArea){maxArea=a; maxIdx=i;}
        }

        Mat visMask; cvtColor(mask, visMask, COLOR_GRAY2BGR);
        if(maxIdx>=0)
        {
            Moments m = moments(contours[maxIdx]);
            Point2f center(static_cast<float>(m.m10/m.m00), static_cast<float>(m.m01/m.m00));
            drawContours(visMask, contours, maxIdx, Scalar(0,255,0),2);
            circle(undistorted, center, 5, Scalar(0,0,255), -1);
            putText(undistorted, format("u=%.1f v=%.1f", center.x, center.y), Point(15,30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,0,0),2);
            cout << frameIdx << ", " << center.x << ", " << center.y << endl;
        }

        imshow("mask", visMask);
        imshow("detect", undistorted);
        if(waitKey(1)==27) break;
    }
}

int main(int argc, char **argv)
{
    if(argc < 2)
    {
        cerr << "用法: " << argv[0] << " <video_path> [config.yaml]" << endl;
        return -1;
    }
    string videoPath = argv[1];
    string configPath;
    if(argc >= 3) {
        configPath = argv[2];
    } else {
        // 获取当前可执行文件所在目录
        char result[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
        string execPath = string(result, (count > 0) ? count : 0);
        string execDir = execPath.substr(0, execPath.find_last_of("/"));
        
        // 尝试不同的相对路径
        vector<string> candidates = {
            execDir + "/../config/kinect_config.yaml",  // build目录执行
            execDir + "/config/kinect_config.yaml",     // 安装目录执行
            "../config/kinect_config.yaml",             // 相对于build目录
            "config/kinect_config.yaml",                // 相对于源码根目录
            "./kinect_config.yaml"                      // 当前目录
        };
        
        for(const auto& path : candidates) {
            if(filesystem::exists(path)) {
                configPath = path;
                break;
            }
        }
        
        if(configPath.empty()) {
            cerr << "错误：无法找到配置文件。请确保config/example_config.yaml存在" << endl;
            return -1;
        }
    }

    cout << "加载配置文件: " << configPath << endl;

    Config cfg;
    if(!loadConfig(configPath, cfg)) return -1;

    runBasketballDetect(videoPath, cfg);
    return 0;
} 