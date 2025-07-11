cmake_minimum_required(VERSION 3.10)

# 项目名称
project(BasketballTrajectorySolver LANGUAGES CXX)

# C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED core calib3d imgproc highgui video aruco)
# 查找 Kinect SDK
find_package(k4a REQUIRED)

# 公共源文件 (通用配置加载等)
set(COMMON_SRC src/utils.cpp)

# 相机适配器源码
set(CAMERA_ADAPTER_SRC 
    src/kinect_camera_adapter.cpp 
    src/camera_factory.cpp
)

# 头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)
# Kinect SDK 头文件路径 (如果安装了)
if(TARGET k4a::k4a)
    message(STATUS "Found Azure Kinect SDK via k4a::k4a target.")
endif()

# 主轨迹求解器
add_executable(trajectory_solver src/main.cpp src/solver.cpp src/solver_camera.cpp ${COMMON_SRC} ${CAMERA_ADAPTER_SRC})

# 篮球检测调试器
add_executable(basketball_detect src/basketball_detect.cpp ${COMMON_SRC})

# 相机测试程序
add_executable(test_camera src/test_camera.cpp ${COMMON_SRC} ${CAMERA_ADAPTER_SRC})

# 链接
# 海康相机SDK和OpenCV
target_link_libraries(trajectory_solver PRIVATE ${OpenCV_LIBS} k4a::k4a)
target_link_libraries(test_camera PRIVATE ${OpenCV_LIBS} k4a::k4a)
target_link_libraries(basketball_detect PRIVATE ${OpenCV_LIBS}) 