#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <iomanip>
#include <string>
#include "common.hpp"
#include "solver.hpp"
#include <vector>
#include "utils.hpp"
#include <filesystem>
namespace fs = std::filesystem;

using namespace std;

int main(int argc, char **argv)
{
    // 基本用法提示
    const std::string usage =
        std::string("用法: ") + argv[0] + " <mode> [path] [config.yaml]\n" +
        "  mode: video | image | camera\n" +
        "  当 mode 为 video 时, [path] 为视频文件路径\n" +
        "  当 mode 为 image 时, [path] 为图片文件路径\n" +
        "  当 mode 为 camera 时, 不需要 [path] 参数\n" +
        "示例:\n" +
        "  " + argv[0] + " video myvideo.mp4 config.yaml\n" +
        "  " + argv[0] + " image img.png\n" +
        "  " + argv[0] + " camera config.yaml";

    // 至少需要一个 mode 参数
    if (argc < 2)
    {
        cerr << usage << endl;
        return -1;
    }

    std::string mode = argv[1];
    std::string inputPath;          // 对 camera 模式可为空
    std::string configPath;

    if (mode == "camera")
    {
        // camera 模式下, 第 2 个参数若存在则视为 config.yaml
        if (argc >= 3)
            configPath = argv[2];
    }
    else
    {
        // video / image 模式下必须提供路径
        if (argc < 3)
        {
            cerr << usage << endl;
            return -1;
        }
        inputPath = argv[2];
        if (argc >= 4)
            configPath = argv[3];
    }

    // 如果未显式给出 configPath, 尝试默认搜索路径
    if (configPath.empty())
    {
        std::vector<std::string> candidates = {
            "../config/kinect_config.yaml",   // build 目录执行
            "config/kinect_config.yaml",      // 源码根目录执行
            "./kinect_config.yaml"            // 当前目录
        };

        for(const auto& path : candidates){
            if(fs::exists(path)){
                configPath = path;
                break;
            }
        }
    }

    cout << "加载配置文件: " << configPath << endl;

    Config cfg;
    if (!loadConfig(configPath, cfg))
        return -1;

    if (mode == "video")
    {
        runTrajectorySolver(inputPath, cfg);
    }
    else if (mode == "image")
    {
        runTrajectorySolverImage(inputPath, cfg);
    }
    else if (mode == "camera")
    {
        runTrajectorySolverCamera(cfg);
    }
    else
    {
        cerr << "未知模式: " << mode << " (需要 video | image | camera)" << endl;
        return -1;
    }

    return 0;
} 