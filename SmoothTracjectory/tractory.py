import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

def visual_rotation(rotations_real, path,rotations_virtual):
    plt.clf()
    plt.figure(figsize=(8, 16))
    
    plt.subplot(5, 1, 1)
    plt.plot(rotations_real[:, 0], "g", label='Original')
    if rotations_virtual is not None:
        plt.plot(rotations_virtual[:, 0], "b",label='Smoothed')
    plt.ylim(-0.5,0.5)
    plt.xlabel('frame ID')
    plt.ylabel('Quaternion x')

    plt.subplot(5, 1, 2)
    plt.plot(rotations_real[:, 1], "g",label='Original')
    if rotations_virtual is not None:
        plt.plot(rotations_virtual[:, 1], "b", label='Smoothed')
    plt.ylim(-0.5,0.5)
    plt.xlabel('frame ID')
    plt.ylabel('Quaternion y')

    plt.subplot(5, 1, 3)
    plt.plot(rotations_real[:, 2], "g",label='Original')
    if rotations_virtual is not None:
        plt.plot(rotations_virtual[:, 2], "b", label='Smoothed')
    plt.ylim(-0.5,0.5)
    plt.xlabel('frame ID')
    plt.ylabel('Quaternion z')


    plt.savefig(path + ".png")
    return

# 讀取CSV檔案並提取資訊
def read_csv(filename):
    quaternions = []
    positions = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            quaternion = [float(val) for val in row[:4]]
            position = [float(val) for val in row[4:]]
            quaternions.append(quaternion)
            positions.append(position)
    return np.array(quaternions), np.array(positions)

# 將四元數轉換成旋轉矩陣
def quaternion_to_rotation_matrix(quaternion):
    q0, q1, q2, q3 = quaternion
    return np.array([[1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q0*q2 + q1*q3)],
                     [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
                     [2*(q1*q3 - q0*q2), 2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2)]])

# 計算旋轉後的向量
def rotate_vector(rotation_matrix, vector):
    return np.dot(rotation_matrix, vector)

# 讀取資料
quaternions, positions = read_csv('/home/kang/SmoothTracjectory/frame_6dof.csv')
#print(quaternions)
data = np.load('/home/kang/SmoothTracjectory/inference_results.npy')
#print(data)
#print(data)
quaternions_flat = data.reshape(-1, 4)
path= '/home/kang/SmoothTracjectory/Tracjectory1'
visual_rotation(quaternions,path,quaternions_flat)



