import numpy as np
from scipy.linalg import block_diag
import csv
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
    
class QuaternionKalmanFilter:
    def __init__(self, dt, process_noise_covariance, measurement_noise_covariance):
        self.dt = dt
        self.Q = process_noise_covariance
        self.R = measurement_noise_covariance
        self.reset()

    def reset(self):
        self.x = np.array([1, 0, 0, 0])  # Initial state
        self.P = np.eye(4)  # Initial covariance matrix

    def predict(self):
        # State transition matrix (constant velocity model)
        F = np.array([[1, -self.dt/2, -self.dt/2, -self.dt/2],
                      [self.dt/2, 1, -self.dt/2, self.dt/2],
                      [self.dt/2, self.dt/2, 1, -self.dt/2],
                      [self.dt/2, -self.dt/2, self.dt/2, 1]])

        # Predicted state
        self.x = np.dot(F, self.x)
        
        # Predicted covariance
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

    def update(self, z):
        # Observation matrix
        H = np.eye(4)

        # Kalman gain
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        # Update state estimate
        y = z - np.dot(H, self.x)
        self.x = self.x + np.dot(K, y)

        # Update covariance
        self.P = self.P - np.dot(np.dot(K, H), self.P)

# Example usage
dt = 1  # Time step (assuming constant)
process_noise_covariance = 0.001 * np.eye(4)  # Process noise covariance (adjust as needed)
measurement_noise_covariance = 0.1 * np.eye(4)  # Measurement noise covariance (adjust as needed)

kf = QuaternionKalmanFilter(dt, process_noise_covariance, measurement_noise_covariance)

# Quaternion measurements

measurements, positions = read_csv('/home/kang/SmoothTracjectory/frame_6dof.csv')
# Smoothed quaternion estimates
smoothed_quaternions = []
for i, measurement in enumerate(measurements):
    kf.predict()
    kf.update(measurement)
    if i >= 2:  # Start saving after the first three measurements
        smoothed_quaternions.append(kf.x)
np.save('inference_results.npy', smoothed_quaternions)

# smoothed_quaternions now contains the smoothed quaternion estimates

