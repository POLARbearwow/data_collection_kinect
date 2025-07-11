#pragma once

#include <string>
#include "common.hpp"

// 运行轨迹解算主循环
void runTrajectorySolver(const std::string &videoPath, const Config &cfg); 

// 运行单张图像的轨迹解算（主要用于调试）
void runTrajectorySolverImage(const std::string &imagePath, const Config &cfg); 

// 运行实时相机的轨迹解算（海康相机）
void runTrajectorySolverCamera(const Config &cfg); 