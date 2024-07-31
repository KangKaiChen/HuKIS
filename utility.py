import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.autograd import Variable


def rotation_matrix_to_euler_angles(rotation_matrix):
    """
    將旋轉矩陣轉換為歐拉角
    Args:
        rotation_matrix (torch.Tensor): 3x3 的旋轉矩陣
    Returns:
        torch.Tensor: 包含旋轉角度的張量，格式為 [roll, pitch, yaw]
    """
    sy = torch.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = torch.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = torch.atan2(-rotation_matrix[2, 0], sy)
        yaw = torch.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = torch.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = torch.atan2(-rotation_matrix[2, 0], sy)
        yaw = 0

    return torch.tensor([roll, pitch, yaw])


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q1[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.stack((w, x, y, z), dim=1)


# 定義四元數逆操作
def quaternion_inverse(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    norm_squared = w**2 + x**2 + y**2 + z**2
    return torch.stack(
        (w / norm_squared, -x / norm_squared, -y / norm_squared, -z / norm_squared),
        dim=1,
    )


def quaternion_to_rotation_matrix(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r11 = 1 - 2 * (y**2 + z**2)
    r12 = 2 * (x * y - w * z)
    r13 = 2 * (x * z + w * y)
    r21 = 2 * (x * y + w * z)
    r22 = 1 - 2 * (x**2 + z**2)
    r23 = 2 * (y * z - w * x)
    r31 = 2 * (x * z - w * y)
    r32 = 2 * (y * z + w * x)
    r33 = 1 - 2 * (x**2 + y**2)
    return torch.stack(
        (
            torch.stack((r11, r12, r13), dim=1),
            torch.stack((r21, r22, r23), dim=1),
            torch.stack((r31, r32, r33), dim=1),
        ),
        dim=1,
    )


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == "SGD":
        optimizer_function = optim.SGD
        kwargs = {"momentum": 0.9}
    elif args.optimizer == "ADAM":
        optimizer_function = optim.Adam
        kwargs = {"betas": (0.9, 0.9999), "eps": 1e-08}
    elif args.optimizer == "ADAMax":
        optimizer_function = optim.Adamax
        kwargs = {"betas": (0.9, 0.999), "eps": 1e-08}
    elif args.optimizer == "RMSprop":
        optimizer_function = optim.RMSprop
        kwargs = {"eps": 1e-08}

    kwargs["lr"] = args.lr
    # kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args.decay_type == "step":
        scheduler = lrs.StepLR(my_optimizer, step_size=args.lr_decay, gamma=args.gamma)
    elif args.decay_type.find("step") >= 0:
        milestones = args.decay_type.split("_")
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer, milestones=milestones, gamma=args.gamma
        )

    return scheduler


def CharbonnierFunc(data, epsilon=0.001):
    return torch.mean(torch.sqrt(data**2 + epsilon**2))


class Module_CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=0.001):
        super(Module_CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, gt):
        return torch.mean(torch.sqrt((output - gt) ** 2 + self.epsilon**2))


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def moduleNormalize(frame):
    return torch.cat(
        [
            (frame[:, 0:1, :, :] - 0.4631),
            (frame[:, 1:2, :, :] - 0.4352),
            (frame[:, 2:3, :, :] - 0.3990),
        ],
        1,
    )


def normalize(x):
    return x * 2.0 - 1.0


def denormalize(x):
    return (x + 1.0) / 2.0


def meshgrid(height, width, grid_min, grid_max):
    x_t = torch.matmul(
        torch.ones(height, 1), torch.linspace(grid_min, grid_max, width).view(1, width)
    )
    y_t = torch.matmul(
        torch.linspace(grid_min, grid_max, height).view(height, 1), torch.ones(1, width)
    )

    grid_x = x_t.view(1, height, width)
    grid_y = y_t.view(1, height, width)
    return grid_x, grid_y


def position_difference(p1, p2):

    p1 = p1[0][4:]
    p2 = p2[0][4:]
    # 计算三维坐标之间的差异

    #   print(p1-p2)

    return p1 - p2
