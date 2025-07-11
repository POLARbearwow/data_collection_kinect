#pragma once

#include <string>
#include "common.hpp"

// 从 YAML 文件加载配置，成功返回 true
bool loadConfig(const std::string &filePath, Config &cfg); 