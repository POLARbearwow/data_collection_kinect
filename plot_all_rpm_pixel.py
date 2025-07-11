import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.colors import to_rgba

# ----------------- 配置 -----------------
# 像素文件所在目录(含 pixels_*.csv)
BASE_DIR = './records'
# 每种RPM使用的色图
COLOR_MAP = plt.cm.tab20  # 最多支持 20 种颜色
POINT_SIZE = 8
LINE_WIDTH = 1.0
# 折线透明度
POLYLINE_ALPHA = 0.3
# 线宽设置
CURVE_WIDTH = 2.0          # 拟合曲线线宽
POLYLINE_WIDTH = 3.0       # 折线线宽

def lighten_color(color, amount: float = 0.5):
    r, g, b, a = to_rgba(color)
    r = r + (1 - r) * amount
    g = g + (1 - g) * amount
    b = b + (1 - b) * amount
    return (r, g, b, a)

def get_timestamp_from_filename(filename: str):
    """解析文件名中的时间戳，pixels_YYYYMMDD_HHMMSS.csv"""
    try:
        if not filename.startswith('pixels_'):
            return None
        date_part, time_part_with_ext = filename.split('_')[-2], filename.split('_')[-1]
        time_part = time_part_with_ext.split('.')[0]
        dt_str = f"{date_part}_{time_part}"
        return datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
    except Exception:
        return None

def load_pixel_files(base_dir):
    pixels = []
    for fpath in glob.glob(os.path.join(base_dir, 'pixels_*.csv')):
        fname = os.path.basename(fpath)
        ts = get_timestamp_from_filename(fname)
        if ts is None:
            continue
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f'读取 {fpath} 失败: {e}')
            continue
        # 过滤无效点
        df = df[(df['ball_u'] != -999) & (df['ball_v'] != -999)]
        if df.empty:
            continue
        df['timestamp'] = df['timestamp_ms'] if 'timestamp_ms' in df.columns else np.arange(len(df))
        df['file_ts'] = ts
        df['source_file'] = fname
        pixels.append(df)
    if not pixels:
        return pd.DataFrame()
    return pd.concat(pixels, ignore_index=True)

def main():
    if not os.path.isdir(BASE_DIR):
        print(f'目录不存在: {BASE_DIR}')
        return
    all_pixels = load_pixel_files(BASE_DIR)
    if all_pixels.empty:
        print('未找到有效的 pixel 数据')
        return

    rpm_values = sorted(all_pixels['rpm'].unique())
    colors = COLOR_MAP(np.linspace(0, 1, len(rpm_values)))
    rpm_color = dict(zip(rpm_values, colors))

    plt.figure(figsize=(10, 8))

    for rpm in rpm_values:
        df_rpm_all = all_pixels[all_pixels['rpm'] == rpm]

        file_groups = [(fname, grp.copy()) for fname, grp in df_rpm_all.groupby('source_file')]
        if not file_groups:
            continue

        file_groups_sorted = sorted(file_groups, key=lambda g: len(g[1]))
        # file_groups_use = file_groups_sorted[1:] if len(file_groups_sorted) > 1 else file_groups_sorted
        file_groups_use = file_groups_sorted
        first_curve_plotted = False  # 控制图例只出现一次

        for fname, df_file in file_groups_use:
            if len(df_file) < 20:
                continue  # 该文件点数不足，跳过

            df_file.sort_values('timestamp', inplace=True)

            img_h = df_file['ball_v'].max() + 10
            x_vals = df_file['ball_u'].values
            y_vals = img_h - df_file['ball_v'].values

            if len(y_vals) == 0:
                continue

            idx_min = int(np.argmin(y_vals))
            x_plot = x_vals[:idx_min + 1]
            y_plot = y_vals[:idx_min + 1]

            if len(x_plot) < 20:
                continue

            # 生成浅色用于折线
            light_col = lighten_color(rpm_color[rpm], 0.5)

            # 折线连接（浅色，加粗）
            plt.plot(x_plot, y_plot, color=light_col, linewidth=POLYLINE_WIDTH, alpha=POLYLINE_ALPHA)

            # 二次多项式拟合
            coeffs = np.polyfit(x_plot, y_plot, deg=2)
            poly = np.poly1d(coeffs)
            x_fit = np.linspace(min(x_plot), max(x_plot), 200)
            y_fit = poly(x_fit)

            label = f'RPM {rpm}' if not first_curve_plotted else None
            plt.plot(x_fit, y_fit, color=rpm_color[rpm], linewidth=CURVE_WIDTH, label=label)
            first_curve_plotted = True

    plt.title('Pixel Trajectories for All RPMs')
    plt.xlabel('u')
    plt.ylabel('v (bottom origin, up)')
    plt.legend(title='RPM')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main() 