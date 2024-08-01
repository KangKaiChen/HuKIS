import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
def quaternion_to_vector(quaternion):
    q1, q2, q3, q4 = quaternion
    return np.array([2*(q2*q4 + q1*q3), 2*(q3*q4 - q1*q2), 1 - 2*(q1**2 + q2**2)])


def visual_rotation(rotations_real, rotations_input, rotations_virtual2=None, path=None):
    plt.clf()
    plt.figure(figsize=(8, 16))
    
    # 绘制 x 轴陀螺仪数据曲线
    plt.subplot(3, 1, 1)
    rotations_real_x = np.concatenate(rotations_real[:,:,0])
    rotations_real_x_input = rotations_input[:,0] 

    print(rotations_real_x_input)
    plt.plot(rotations_real_x, "g", label='Real X')
    plt.plot(rotations_real_x_input, "b")
    
    x_min = min(np.min(rotations_real_x), np.min(rotations_input[:, 0]))
    x_max = max(np.max(rotations_real_x), np.max(rotations_input[:, 0]))
    plt.ylim(x_min - 0.1, x_max + 0.1)
   # plt.ylim(np.min(rotations_real_x) - 0.1, np.max(rotations_real_x) + 0.1)
    plt.xlabel('Frame ID')
    plt.ylabel('Gyro X')
    plt.legend()

    # 绘制 y 轴陀螺仪数据曲线
    plt.subplot(3, 1, 2)
    rotations_real_y = np.concatenate(rotations_real[:,:,1])
    rotations_real_y_input = rotations_input[:,1]
   # print("rotations_real_y",rotations_real_y)
    plt.plot(rotations_real_y, "g", label='Real Y')
    plt.plot(rotations_real_y_input, "b")
    y_min = min(np.min(rotations_real_y), np.min(rotations_input[:, 1]))
    y_max = max(np.max(rotations_real_y), np.max(rotations_input[:, 1]))
    plt.ylim(y_min - 0.1, y_max + 0.1)
    plt.xlabel('Frame ID')
    plt.ylabel('Gyro Y')
    plt.legend()

    # 绘制 z 轴陀螺仪数据曲线
    plt.subplot(3, 1, 3)
    rotations_real_z = np.concatenate(rotations_real[:,:,2])
    rotations_real_z_input = rotations_input[:,2]
   # print("rotations_real_z",rotations_real_z)
    plt.plot(rotations_real_z, "g", label='Real Z')
    plt.plot(rotations_real_z_input, "b")
    z_min = min(np.min(rotations_real_z), np.min(rotations_input[:, 2]))
    z_max = max(np.max(rotations_real_z), np.max(rotations_input[:, 2]))
    plt.ylim(z_min - 0.1, z_max + 0.1)
    plt.xlabel('Frame ID')
    plt.ylabel('Gyro Z')
    plt.legend()

    if path:
        plt.savefig(path[:-4]+".jpg")
    plt.show()

tensor_array = np.load('inference_results.npy')
# 从CSV文件中读取数据
df = pd.read_csv('/home/kang/SmoothTracjectory/dataset/eval/IMG7_0001/frame_6dof.csv',usecols=range(4))
# 转换为NumPy数组
data_array = df.to_numpy()
visual_rotation(tensor_array[:, :, :3], rotations_input=data_array)

