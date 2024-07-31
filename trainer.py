import sys

sys.path.append("core")

import argparse
import os

import cv2
import torch
from raft import RAFT
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image as imwrite
from utils import flow_viz
from utils.utils import InputPadder

import softsplat
import utility
from models_arbitrary import dvsnet
from utility import (
    CharbonnierFunc,
    moduleNormalize,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_euler_angles,
    to_variable,
)

backwarp_tenGrid = {}
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToPILImage, ToTensor


def torch_QuaternionProduct(q1, q2, USE_CUDA=True):
    x1 = q1[:, 0]
    y1 = q1[:, 1]
    z1 = q1[:, 2]
    w1 = q1[:, 3]

    x2 = q2[:, 0]
    y2 = q2[:, 1]
    z2 = q2[:, 2]
    w2 = q2[:, 3]

    batch_size = q1.size()[0]
    quat = Variable(torch.zeros((batch_size, 4), requires_grad=True))
    if USE_CUDA == True:
        quat = quat.cuda()

    quat[:, 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    quat[:, 0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    quat[:, 1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    quat[:, 2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    quat = torch_norm_quat(quat)

    return quat


def torch_norm_quat(quat, USE_CUDA=True):
    # Method 1:
    batch_size = quat.size()[0]
    quat_out = Variable(torch.zeros((batch_size, 4), requires_grad=True))
    if USE_CUDA == True:
        quat_out = quat_out.cuda()
    for i in range(batch_size):
        norm_quat = torch.norm(quat[i])
        if norm_quat > 1e-6:
            quat_out[i] = quat[i] / norm_quat
            #     [0 norm_quat norm_quat - 1e-6]
        else:
            quat_out[i, :3] = quat[i, :3] * 0
            quat_out[i, 3] = quat[i, 3] / quat[i, 3]

    # Method 2:
    # quat = quat / (torch.unsqueeze(torch.norm(quat, dim = 1), 1) + 1e-6) # check norm
    return quat_out


def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = (
            torch.linspace(
                -1.0 + (1.0 / tenFlow.shape[3]),
                1.0 - (1.0 / tenFlow.shape[3]),
                tenFlow.shape[3],
            )
            .view(1, 1, 1, -1)
            .expand(-1, -1, tenFlow.shape[2], -1)
        )
        tenVer = (
            torch.linspace(
                -1.0 + (1.0 / tenFlow.shape[2]),
                1.0 - (1.0 / tenFlow.shape[2]),
                tenFlow.shape[2],
            )
            .view(1, 1, -1, 1)
            .expand(-1, -1, -1, tenFlow.shape[3])
        )

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()

        print("backwarp_tenGrid", backwarp_tenGrid[str(tenFlow.shape)])
        print(
            "backwarp_tenGrid+flow", backwarp_tenGrid[str(tenFlow.shape)] + tenFlow
        ).permute(0, 2, 3, 1)
    # end

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def quaternion_average(quaternions):
    # 将四元数转换为张量
    quaternions = torch.stack(quaternions)

    # 迭代求解平均四元数
    average_quaternion = quaternions[0]
    for q in quaternions[1:]:
        dot_product = torch.sum(average_quaternion * q)
        if dot_product < 0.0:
            q = -q
            dot_product = -dot_product
        if dot_product > 0.9995:  # 如果两个四元数接近平行，直接线性插值
            average_quaternion = (average_quaternion + q) / 2.0
        else:
            omega = torch.acos(dot_product)
            sin_omega = torch.sin(omega)
            weight1 = torch.sin((1.0 - 0.5) * omega) / sin_omega
            weight2 = torch.sin(0.5 * omega) / sin_omega
            average_quaternion = weight1 * average_quaternion + weight2 * q

    return average_quaternion


def norm_flow(flow, h, w):
    if flow.shape[2] == 2:
        flow[:, :, 0] /= h
        flow[:, :, 1] /= w
    else:
        flow[:, :, :, 0] /= h
        flow[:, :, :, 1] /= w
    return flow


class Trainer:
    def __init__(
        self, args, train_loader, test_loader, my_model, my_loss, start_epoch=0
    ):
        # 初始化训练器类
        self.args = args  # 存储参数
        self.train_loader = train_loader  # 存储训练数据加载器
        self.max_step = self.train_loader.__len__()  # 获取训练数据集的大小
        self.test_loader = test_loader  # 存储测试数据加载器
        self.model = my_model  # 存储模型
        self.loss = my_loss  # 存储损失函数
        self.current_epoch = start_epoch  # 存储当前训练的轮次
        self.iteration = 0
        self.writer = SummaryWriter("./log")
        self.iteration = 0
        self.optimizer = utility.make_optimizer(args, self.model)
        if not os.path.exists(args.out_dir):  # 如果输出目录不存在则创建
            os.makedirs(args.out_dir)
        self.result_dir = args.out_dir + "/result"  # 设置结果目录
        self.ckpt_dir = args.out_dir + "/checkpoint"  # 设置检查点目录

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.degree = []
        self.logfile = open(args.out_dir + "/log.txt", "w")
        self.firsr_time = True
        # Initial Test
        self.model.eval()  # 设置模型为评估模式

        # 创建模型实例

        input_size = 4  # 输入特征数
        hidden_size = 128  # 隐藏层大小
        output_size = 4  # 输出特征数，与输入相同
        num_layers = 2  # Transformer 层数
        num_heads = 4  # 注意力头的数量
        self.test_loader.Test(
            self.model,
            self.result_dir,
            self.current_epoch,
            self.logfile,
            str(self.current_epoch).zfill(3) + ".png",
        )  # 进行初始测试
        self.mymodel = dvsnet.TransformerModel(
            input_size, hidden_size, output_size, num_layers, num_heads
        )
        # 加载检查点
        self.checkpoint_path = (
            "checkpoints/epoch_29_checkpoint.pth"  # 假设是您最后一个检查点的路径
        )
        self.checkpoint = torch.load(self.checkpoint_path)
        self.mymodel.load_state_dict(self.checkpoint["model_state_dict"])
        self.mymodel.eval()
        # RAFT
        self.flow_model = torch.nn.DataParallel(RAFT(args))
        self.flow_model.load_state_dict(torch.load("raft_models/raft-things.pth"))
        self.flow_model = self.flow_model.module
        self.flow_model.to("cuda")
        self.flow_model.eval()

    def train(self):
        # Train
        print("Training start --------------------------")
        self.model.train()

        for batch_idx, (
            input_frames,
            base_frame,
            GT,
            frame_dofs,
            gt_frame_dof,
        ) in enumerate(self.train_loader):
            is_training = True
            #  input_frames: [7] (base_frame included)
            #  base_frame: 1
            #  GT: 1
            #  frame_dofs 7
            #  gt_frame_dof
            assert len(input_frames) % 2 == 1
            base_frame = to_variable(base_frame).detach()
            GT = to_variable(GT).detach()
            for i in range(len(input_frames)):
                input_frames[i] = to_variable(input_frames[i]).detach()

            # RAFT
            with torch.no_grad():
                if self.args.bundle_forward_flow > 0:
                    _, F_kprime_to_k = self.flow_model(
                        base_frame.detach(), GT.detach(), iters=20, test_mode=True
                    )
                else:
                    # 跑這裡
                    _, F_kprime_to_k = self.flow_model(
                        GT.detach(), base_frame.detach(), iters=20, test_mode=True
                    )

                F_n_to_k_s = []
                F_k_to_n_s = []

                # 提取每个子列表的前四个元素
                one_quat = [sublist[0, :4] for sublist in frame_dofs]
                one_quat_stacked = torch.stack(one_quat, dim=0)
                one_quat_stacked = one_quat_stacked.unsqueeze(0)
                one_quat_stacked_squeezed = torch.squeeze(one_quat_stacked, dim=0)
                smooth_out = self.mymodel(one_quat_stacked)
                # 計算四元數的差異
                q_diff = quaternion_multiply(
                    one_quat_stacked_squeezed, quaternion_inverse(smooth_out)
                )

                #     print(one_quat_stacked)

                # 將四元數的差異轉換為旋轉矩陣
                rotation_matrices = quaternion_to_rotation_matrix(q_diff)
                rotated_images = []
                disparity_image = []
                for i, rot_matrix in enumerate(rotation_matrices):
                    euler_angles = rotation_matrix_to_euler_angles(rot_matrix)
                    # 提取 yaw 角度
                    yaw_angle_degrees = torch.rad2deg(euler_angles[2])
                    self.degree.append(yaw_angle_degrees)
                    #   print("degree", yaw_angle_degrees)

                    # 选择批次中的一个图像
                    input_frame = (
                        input_frames[i].squeeze(0).cpu()
                    )  # 将张量移到 CPU 上并去除批次维度
                    # 将 PyTorch 张量转换为 NumPy 数组，并将其乘以 255 并转换为 uint8 类型
                    np_image = (input_frame.numpy() * 255).astype(np.uint8)
                    # 将 NumPy 数组转换为 PIL 图像
                    pil_image = Image.fromarray(
                        np_image.transpose(1, 2, 0)
                    )  # 转换为 HWC 格式

                    # 将图像逆时针旋转 yaw 角度
                    rotated_pil_image = TF.rotate(
                        pil_image, yaw_angle_degrees.item(), resample=Image.BICUBIC
                    )

                    # 自动裁剪旋转后的图像
                    # rotated_pil_image = rotated_pil_image.crop(rotated_pil_image.getbbox())
                    # 将裁剪后的图像转换为张量
                    rotated_tensor = ToTensor()(rotated_pil_image).unsqueeze(0)
                    rotated_tensor = rotated_tensor.to(input_frames[i].device)
                    rotated_images.append(rotated_tensor)
                    disparity_image.append(input_frames[i] - rotated_tensor)

                for idx, img in enumerate(input_frames):
                    _, ref_frame_flow = self.flow_model(
                        img.detach(), base_frame.detach(), iters=20, test_mode=True
                    )

                    F_n_to_k_s.append(ref_frame_flow)
                    _, ref_frame_flow2 = self.flow_model(
                        base_frame.detach(), img.detach(), iters=20, test_mode=True
                    )
                    F_k_to_n_s.append(ref_frame_flow2)
            # TODO: detach

            self.optimizer.zero_grad()

            output = self.model(
                input_frames, F_kprime_to_k, F_n_to_k_s, F_k_to_n_s, disparity_image
            )
            loss = self.loss(output, GT, input_frames)

            loss.backward()
            self.optimizer.step()
            #  self.writer.add_scalar("loss", loss.item(), self.iteration)
            #   self.writer.add_scalar("lossMSE", avg_loss, self.iteration)
            self.writer.add_scalar("loss", loss.item(), self.iteration)
            self.iteration += 1
            if batch_idx % 100 == 0:
                print(
                    "{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f}".format(
                        "Train Epoch: ",
                        "["
                        + str(self.current_epoch)
                        + "/"
                        + str(self.args.epochs)
                        + "]",
                        "Step: ",
                        "[" + str(batch_idx) + "/" + str(self.max_step) + "]",
                        "train loss: ",
                        loss.item(),
                    )
                )
        self.current_epoch += 1
        torch.save(
            {"epoch": self.current_epoch, "state_dict": self.model.get_state_dict()},
            self.ckpt_dir + "/model_epoch" + str(self.current_epoch).zfill(3) + ".pth",
        )
        # self.scheduler.step()

    def test(self):
        # Test
        # torch.save({'epoch': self.current_epoch, 'state_dict': self.model.get_state_dict()}, self.ckpt_dir + '/model_epoch' + str(self.current_epoch).zfill(3) + '.pth')
        self.model.eval()
        self.test_loader.Test(
            self.model,
            self.result_dir,
            self.current_epoch,
            self.logfile,
            str(self.current_epoch).zfill(3) + ".png",
        )
        self.logfile.write("\n")

    def terminate(self):
        return self.current_epoch >= self.args.epochs

    def close(self):
        self.logfile.close()
