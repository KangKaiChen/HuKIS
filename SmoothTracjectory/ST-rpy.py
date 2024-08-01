import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# 读取CSV文件，手动指定列标头
column_names = ['qx', 'qy', 'qz', 'qw', 'tx', 'ty', 'tz']
df = pd.read_csv('D:\SmoothTracjectory\pose3.csv', header=None, names=column_names)

# 检查读取的数据
print(df.head())

# 提取四元数值
quaternions = df[['qx', 'qy', 'qz', 'qw']].values

# 定义 Savitzky-Golay 滤波器平滑函数
def savgol_smooth(data, window_size, poly_order):
    return savgol_filter(data, window_size, poly_order, axis=0)

# 平滑处理四元数，注意窗口大小和多项式阶数
window_size = 15
poly_order = 2
quaternions_smooth = savgol_smooth(quaternions, window_size, poly_order)

# 将平滑后的四元数转换为欧拉角
euler_angles = R.from_quat(quaternions).as_euler('xyz', degrees=True)
euler_angles_smooth = R.from_quat(quaternions_smooth).as_euler('xyz', degrees=True)

# 将平滑后的欧拉角存入DataFrame
df['roll'] = euler_angles[:, 0]
df['pitch'] = euler_angles[:, 1]
df['yaw'] = euler_angles[:, 2]

df['roll_smooth'] = euler_angles_smooth[:, 0]
df['pitch_smooth'] = euler_angles_smooth[:, 1]
df['yaw_smooth'] = euler_angles_smooth[:, 2]

# 可视化原始和平滑后的角度变化
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(df.index, df['roll'], label='Roll (original)')
plt.plot(df.index, df['roll_smooth'], label='Roll (smoothed)')
plt.xlabel('Index')
plt.ylabel('Roll Angle (degrees)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(df.index, df['pitch'], label='Pitch (original)')
plt.plot(df.index, df['pitch_smooth'], label='Pitch (smoothed)')
plt.xlabel('Index')
plt.ylabel('Pitch Angle (degrees)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(df.index, df['yaw'], label='Yaw (original)')
plt.plot(df.index, df['yaw_smooth'], label='Yaw (smoothed)')
plt.xlabel('Index')
plt.ylabel('Yaw Angle (degrees)')
plt.legend()

plt.tight_layout()
plt.show()


# ... [前面的代碼保持不變] ...

# 計算角速度
def calculate_angular_velocity(angles, time_step=1):
    return np.diff(angles, axis=0) / time_step

# 計算原始和平滑後的角速度
angular_velocity = calculate_angular_velocity(euler_angles)
angular_velocity_smooth = calculate_angular_velocity(euler_angles_smooth)

# 將角速度添加到 DataFrame
df['roll_velocity'] = np.pad(angular_velocity[:, 0], (1, 0), 'constant')
df['pitch_velocity'] = np.pad(angular_velocity[:, 1], (1, 0), 'constant')
df['yaw_velocity'] = np.pad(angular_velocity[:, 2], (1, 0), 'constant')

df['roll_velocity_smooth'] = np.pad(angular_velocity_smooth[:, 0], (1, 0), 'constant')
df['pitch_velocity_smooth'] = np.pad(angular_velocity_smooth[:, 1], (1, 0), 'constant')
df['yaw_velocity_smooth'] = np.pad(angular_velocity_smooth[:, 2], (1, 0), 'constant')

# # 可視化角速度
# plt.figure(figsize=(12, 9))

# plt.subplot(3, 1, 1)
# plt.plot(df.index[1:], df['roll_velocity'][1:], label='Roll Velocity (original)')
# plt.plot(df.index[1:], df['roll_velocity_smooth'][1:], label='Roll Velocity (smoothed)')
# plt.xlabel('Index')
# plt.ylabel('Roll Angular Velocity (degrees/step)')
# plt.legend()

# plt.subplot(3, 1, 2)
# plt.plot(df.index[1:], df['pitch_velocity'][1:], label='Pitch Velocity (original)')
# plt.plot(df.index[1:], df['pitch_velocity_smooth'][1:], label='Pitch Velocity (smoothed)')
# plt.xlabel('Index')
# plt.ylabel('Pitch Angular Velocity (degrees/step)')
# plt.legend()

# plt.subplot(3, 1, 3)
# plt.plot(df.index[1:], df['yaw_velocity'][1:], label='Yaw Velocity (original)')
# plt.plot(df.index[1:], df['yaw_velocity_smooth'][1:], label='Yaw Velocity (smoothed)')
# plt.xlabel('Index')
# plt.ylabel('Yaw Angular Velocity (degrees/step)')
# plt.legend()

# plt.tight_layout()
# plt.show()

# 評估角速度的平滑效果
def angular_velocity_smoothness(velocities):
    return np.mean(np.abs(np.diff(velocities, axis=0)))

original_smoothness = angular_velocity_smoothness(angular_velocity)
smoothed_smoothness = angular_velocity_smoothness(angular_velocity_smooth)

print(f'Original Angular Velocity Smoothness: {original_smoothness}')
print(f'Smoothed Angular Velocity Smoothness: {smoothed_smoothness}')

# 計算角速度的均方根誤差（RMSE）
def rmse_angular_velocity(original, smoothed):
    return np.sqrt(np.mean((original - smoothed) ** 2))

rmse_roll = rmse_angular_velocity(df['roll_velocity'][1:], df['roll_velocity_smooth'][1:])
rmse_pitch = rmse_angular_velocity(df['pitch_velocity'][1:], df['pitch_velocity_smooth'][1:])
rmse_yaw = rmse_angular_velocity(df['yaw_velocity'][1:], df['yaw_velocity_smooth'][1:])

print(f'RMSE Roll Angular Velocity: {rmse_roll}')
print(f'RMSE Pitch Angular Velocity: {rmse_pitch}')
print(f'RMSE Yaw Angular Velocity: {rmse_yaw}')