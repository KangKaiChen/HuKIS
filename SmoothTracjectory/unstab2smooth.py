import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 生成不穩定軌跡
n_points = 250
x = np.linspace(0, 10, n_points)
y_unstable = np.sin(2*x) + np.random.normal(0, 0.5, n_points)

# # 將 y 值限制在 -5 到 5 之間
# y_unstable = np.clip(y_unstable, , 10)

# 使用 Savitzky-Golay 濾波器平滑軌跡
window_length = 19  # 必須是奇數
poly_order = 3  # 多項式階數
y_smooth = savgol_filter(y_unstable, window_length, poly_order)

# 繪製圖形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(x, y_unstable)
ax1.set_title('Unstable trajectory')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_xlim(0 ,5)
ax1.set_ylim(-5, 5)

ax2.plot(x, y_smooth)
ax2.set_title('Smooth trajectory')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_xlim(0, 5)
ax2.set_ylim(-5, 5)

plt.tight_layout()
plt.show()