import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# 讀取CSV文件，手動指定列標頭
column_names = ['qx', 'qy', 'qz', 'qw', 'tx', 'ty', 'tz']
df = pd.read_csv('D:\SmoothTracjectory\poses1.csv', header=None, names=column_names)
 
# 檢查讀取的數據
print(df.head())

# 提取t的值
t_values = df[['tx', 'ty', 'tz']].values

# 定義移動平均平滑函數
def moving_average(data, window_size):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec

# 平滑處理，注意移動平均的窗口大小
window_size = 7
t_smooth = np.zeros_like(t_values)

for i in range(3):
    smoothed_data = moving_average(t_values[:, i], window_size)
    # 用前面的數據進行填充
    t_smooth[:, i] = np.concatenate((t_values[:window_size-1, i], smoothed_data))

# 將平滑後的值存回DataFrame
df['tx_smooth'] = t_smooth[:, 0]
df['ty_smooth'] = t_smooth[:, 1]
df['tz_smooth'] = t_smooth[:, 2]

# 檢查平滑後的數據
print(df[['tx', 'tx_smooth', 'ty', 'ty_smooth', 'tz', 'tz_smooth']].head())

# 儲存平滑後的數據到新的CSV文件，包含標頭
output_columns = column_names + ['tx_smooth', 'ty_smooth', 'tz_smooth']
df.to_csv('smoothed_output.csv', index=False, columns=output_columns)

# 可視化結果

# 原始和平滑後的 tx
plt.subplot(3, 1, 1)
plt.plot(df['tx'], label='Original tx')
plt.plot(df['tx_smooth'], label='Smoothed tx')
plt.xlabel('Sample')
plt.ylabel('tx')
plt.legend()
plt.title('Original and Smoothed tx')
# 原始和平滑後的 ty
plt.subplot(3, 1, 2)
plt.plot(df['ty'], label='Original ty')
plt.plot(df['ty_smooth'], label='Smoothed ty')
plt.xlabel('Sample')
plt.ylabel('ty')
plt.legend()
plt.title('Original and Smoothed ty')

# 原始和平滑後的 tz
plt.subplot(3, 1, 3)
plt.plot(df['tz'], label='Original tz')
plt.plot(df['tz_smooth'], label='Smoothed tz')
plt.xlabel('Sample')
plt.ylabel('tz')
plt.legend()
plt.title('Original and Smoothed tz')

plt.tight_layout()
plt.show()

# 可視化結果：繪製 tx 和 ty 的原始軌跡和平滑後的軌跡
# 繪製原始軌跡
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 繪製原始軌跡
ax.plot(df['tx'], df['ty'], df['tz'], label='Original trajectory')

# 繪製平滑後的軌跡
ax.plot(df['tx_smooth'], df['ty_smooth'], df['tz_smooth'], label='Smoothed trajectory')

ax.set_xlabel('tx')
ax.set_ylabel('ty')
ax.set_zlabel('tz')
ax.legend()
ax.set_title('Original and Smoothed 3D Trajectories')
