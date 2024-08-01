import torch
import torch.nn as nn

def quaternion_distance(q1, q2):
    """
    计算两个四元数之间的欧几里得距离
    """
    return torch.norm(q1 - q2)

def quaternion_smoothness_loss(predicted_quaternions):
    """
    计算四元数序列的平滑损失
    """
    smooth_loss = 0.0
    for i in range(1, len(predicted_quaternions)):
        smooth_loss += quaternion_distance(predicted_quaternions[i], predicted_quaternions[i-1])
    return smooth_loss

def C1_Smooth_loss(quat,inputs):
    quat_zero = torch.zeros_like(quat)
    quat_zero[:, :2] = 0.001  # 将x和y分量设置为0
    quat_zero[:, 2] = inputs[:, 2]  # 保持 z 分量与原始 quat 相同
    quat_zero[:, 3] = 1  # 将w分量设置为1
    
 #   print("quat_zero",quat_zero)
    
    return nn.MSELoss()(quat, quat_zero)


def angle_continuity_loss(quat):
    angle_diff = torch.acos(torch.abs(torch.sum(quat[:, 1:] * quat[:, :-1], dim=-1)))  # 计算相邻四元数之间的角度差异
    angle_continuity_loss = torch.mean(angle_diff)  # 角度连续性损失，角度差异的平均值
    return angle_continuity_loss
    
def quaternion_real_imag_loss(outputs,inputs):


    return torch.mean(torch.abs(outputs - inputs))


# 定义新的损失函数
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        
        self.c1_weight = 5.0
        self.smooth = 5.0

    def forward(self, outputs, inputs):
        # 计算 MSE 损失
        mse_loss = nn.MSELoss()(outputs, inputs)
     
        c1_loss = C1_Smooth_loss(outputs,inputs)
        c1_loss = self.c1_weight * c1_loss
        # 计算角度损失
        angle_loss = angle_continuity_loss(outputs)
        
      #  smooth_loss_value = smooth_loss(outputs) * self.smooth
        smooth_loss1 = quaternion_smoothness_loss(outputs)
        meanloss = quaternion_real_imag_loss(outputs, inputs)

        
        # 返回加权损失
        return  smooth_loss1 + c1_loss
