import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------- 配置区域 ----------------------
# 请在此处修改要分析的目录和 RPM
BASE_DIR = './records'
TARGET_RPM = 7  # 要分析的 RPM 值
# 每个文件对（水平方向一列）的宽度（英寸）
FIG_WIDTH_PER_PAIR = 5
# 整体图的高度（英寸）
FIG_HEIGHT = 8
# 图像尺寸
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
# -----------------------------------------------------

def get_timestamp_from_filename(filename: str):
    """根据文件名解析时间戳，格式 record_YYYYMMDD_HHMMSS.csv / pixels_YYYYMMDD_HHMMSS.csv"""
    try:
        if not filename.endswith('.csv'):
            return None
        parts = filename.split('_')
        if len(parts) < 3:
            return None
        date_part, time_part_with_ext = parts[-2], parts[-1]
        time_part = time_part_with_ext.split('.')[0]
        dt_str = f"{date_part}_{time_part}"
        return datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def load_files(base_dir: str):
    """扫描base_dir，返回record_files, pixel_files字典，key为timestamp"""
    record_files, pixel_files = {}, {}
    for fpath in glob.glob(os.path.join(base_dir, '*.csv')):
        fname = os.path.basename(fpath)
        ts = get_timestamp_from_filename(fname)
        if ts is None:
            continue
        if fname.startswith('record_'):
            record_files[ts] = fpath
        elif fname.startswith('pixels_'):
            pixel_files[ts] = fpath
    return record_files, pixel_files


def main():
    base_dir = BASE_DIR
    target_rpm = TARGET_RPM

    if not os.path.isdir(base_dir):
        print(f'目录不存在: {base_dir}')
        return

    record_map, pixel_map = load_files(base_dir)

    # 找到同时存在 record 与 pixel 的时间戳
    common_ts = sorted(set(record_map.keys()) & set(pixel_map.keys()))

    pairs = []
    for ts in common_ts:
        record_file = record_map[ts]
        # 只快速读取 header 和第一行确定 rpm
        try:
            record_df_head = pd.read_csv(record_file, nrows=1)
        except Exception as e:
            print(f'读取 {record_file} 失败: {e}')
            continue
        if record_df_head.empty or 'rpm' not in record_df_head.columns:
            continue
        rpm_val = int(record_df_head['rpm'].iloc[0])
        if rpm_val == target_rpm:
            pairs.append((record_file, pixel_map[ts]))

    if not pairs:
        print(f'未找到 RPM={target_rpm} 的匹配文件对')
        return

    n = len(pairs)
    fig_width = max(FIG_WIDTH_PER_PAIR * n, 6)  # 最小 6 英寸宽，防止过小
    fig, axes = plt.subplots(2, n, figsize=(fig_width, FIG_HEIGHT), squeeze=False)
    fig.suptitle(f'RPM {target_rpm} World vs Pixel', fontsize=16)

    for idx, (record_file, pixel_file) in enumerate(pairs):
        # 处理 record
        record_df = pd.read_csv(record_file)
        x_record = -record_df['new_y']  # 水平坐标
        y_record = record_df['new_x']   # 垂直坐标
        axes[0, idx].scatter(x_record, y_record, s=10, c='tab:blue')
        axes[0, idx].set_title(f'World\n{os.path.basename(record_file)}')
        axes[0, idx].set_xlabel('X (world)  = -new_y')
        axes[0, idx].set_ylabel('Y (world)  = new_x')
        axes[0, idx].grid(True, linestyle='--', alpha=0.5)

        # 处理 pixel
        pixel_df = pd.read_csv(pixel_file)
        # 过滤掉 -999
        pixel_df = pixel_df[(pixel_df['ball_u'] != -999) & (pixel_df['ball_v'] != -999)]
        if pixel_df.empty:
            print(f'{pixel_file} 没有有效像素坐标')
            continue

        # ball 坐标
        x_ball = pixel_df['ball_u']
        y_ball = IMAGE_HEIGHT - pixel_df['ball_v']  # 使用固定的图像高度
        axes[1, idx].scatter(x_ball, y_ball, s=10, c='tab:orange', label='ball')

        # aruco 坐标（过滤 -999）
        aruco_df = pixel_df[(pixel_df['aruco_u'] != -999) & (pixel_df['aruco_v'] != -999)]
        if not aruco_df.empty:
            x_aruco = aruco_df['aruco_u']
            y_aruco = IMAGE_HEIGHT - aruco_df['aruco_v']  # 使用固定的图像高度
            axes[1, idx].scatter(x_aruco, y_aruco, s=12, c='tab:green', marker='x', label='aruco')
        
        # 设置像素坐标系的显示范围
        axes[1, idx].set_xlim(0, IMAGE_WIDTH)
        axes[1, idx].set_ylim(0, IMAGE_HEIGHT)
        
        axes[1, idx].set_title(f'Pixel\n{os.path.basename(pixel_file)}')
        axes[1, idx].set_xlabel('u')
        axes[1, idx].set_ylabel('v (bottom origin, up)')
        axes[1, idx].grid(True, linestyle='--', alpha=0.5)
        axes[1, idx].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    main() 