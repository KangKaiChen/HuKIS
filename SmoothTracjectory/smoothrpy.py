import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# 读取CSV文件，手动指定列标头
column_names = ['qx', 'qy', 'qz', 'qw', 'tx', 'ty', 'tz']
df = pd.read_csv('D:\SmoothTracjectory\poses.csv', header=None, names=column_names)

# 检查读取的数据
print(df.head())

# 提取四元数值
quaternions = df[['qx', 'qy', 'qz', 'qw']].values

# 定义 Savitzky-Golay 滤波器平滑函数
def savgol_smooth(data, window_size, poly_order):
    return savgol_filter(data, window_size, poly_order, axis=0)

# 平滑处理，注意窗口大小和多项式阶数
window_size = 11
poly_order = 2
quaternions_smooth = savgol_smooth(quaternions, window_size, poly_order)

# 将四元数转换为欧拉角
euler_angles = R.from_quat(quaternions).as_euler('xyz', degrees=True)
euler_angles_smooth = R.from_quat(quaternions_smooth).as_euler('xyz', degrees=True)

df['roll'] = euler_angles[:, 0]
df['pitch'] = euler_angles[:, 1]
df['yaw'] = euler_angles[:, 2]

df['roll_smooth'] = euler_angles_smooth[:, 0]
df['pitch_smooth'] = euler_angles_smooth[:, 1]
df['yaw_smooth'] = euler_angles_smooth[:, 2]

# 检查平滑后的数据
print(df[['roll', 'roll_smooth', 'pitch', 'pitch_smooth', 'yaw', 'yaw_smooth']].head())
plt.figure(figsize=(8, 2))
plt.plot(df['roll'], label='Original roll')
plt.plot(df['roll_smooth'], label='Smoothed roll')
plt.xlabel('Sample')
plt.ylabel('Roll (degrees)')
plt.legend()
plt.title('Original and Smoothed Roll')
plt.show()
plt.figure(figsize=(8, 2))
plt.plot(df['pitch'], label='Original pitch')
plt.plot(df['pitch_smooth'], label='Smoothed pitch')
plt.xlabel('Sample')
plt.ylabel('Pitch (degrees)')
plt.legend()
plt.title('Original and Smoothed Pitch')
plt.show()
plt.figure(figsize=(8, 2))
plt.plot(df['yaw'], label='Original yaw')
plt.plot(df['yaw_smooth'], label='Smoothed yaw')
plt.xlabel('Sample')
plt.ylabel('Yaw (degrees)')
plt.legend()
plt.title('Original and Smoothed Yaw')
plt.show()



# 可视化 roll, pitch, yaw 的结果
plt.figure(figsize=(12, 12))

# roll
plt.subplot(3, 1, 1)
plt.plot(df['roll'], label='Original roll')
plt.plot(df['roll_smooth'], label='Smoothed roll')
plt.xlabel('Sample')
plt.ylabel('Roll (degrees)')
plt.legend()

# pitch
plt.subplot(3, 1, 2)
plt.plot(df['pitch'], label='Original pitch')
plt.plot(df['pitch_smooth'], label='Smoothed pitch')
plt.xlabel('Sample')
plt.ylabel('Pitch (degrees)')
plt.legend()

# yaw
plt.subplot(3, 1, 3)
plt.plot(df['yaw'], label='Original yaw')
plt.plot(df['yaw_smooth'], label='Smoothed yaw')
plt.xlabel('Sample')
plt.ylabel('Yaw (degrees)')
plt.legend()

plt.tight_layout()
plt.show()
