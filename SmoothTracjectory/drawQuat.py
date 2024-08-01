import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# 加载保存的 Numpy 数组
quaternions = np.load('quaternions.npy', allow_pickle=True)

# 创建一个新的图形窗口
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 遍历四元数数组
for epoch in range(quaternions.shape[0]):
    epoch_quaternions = quaternions[epoch]
    
    # 遍历当前 epoch 的四元数
    for i in range(epoch_quaternions.shape[0]):
        quaternion = epoch_quaternions[i]
        
        # 将四元数转换为 Numpy 数组，并且脱离计算图
        quaternion = quaternion.detach().numpy()
        
        # 将四元数转换为旋转矩阵
        r = R.from_quat(quaternion)
        rotation_matrix = r.as_matrix()
        
        # 获取旋转矩阵的旋转向量
        rotation_vector = R.from_matrix(rotation_matrix).as_rotvec()
        
        # 绘制当前四元数的方向
        ax.quiver(0, 0, 0, rotation_vector[0], rotation_vector[1], rotation_vector[2], color=np.random.rand(3,))
    
# 设置图形属性
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Quaternion Directions')

# 显示图形
plt.show()

