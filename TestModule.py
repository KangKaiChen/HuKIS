import sys

sys.path.append("core")

import argparse

# from utility import quaternion_multiply,quaternion_to_rotation_matrix,quaternion_inverse
import glob
import os
import random
from math import log10

import cv2
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from raft import RAFT
from torchvision import transforms
from torchvision.utils import save_image as imwrite
from utils import flow_viz
from utils.utils import InputPadder

import softsplat
from models_arbitrary import dvsnet
from utility import (
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_euler_angles,
    to_variable,
)

backwarp_tenGrid = {}
import math

import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import ToTensor


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


def norm_flow(flow, h, w):
    if flow.shape[2] == 2:
        flow[:, :, 0] /= h
        flow[:, :, 1] /= w
    else:
        flow[:, :, :, 0] /= h
        flow[:, :, :, 1] /= w
    return flow


# end

# def create_forward_proxy(forward_flow, input_frames, base_frame, flow_model, out_path):
#     GAUSSIAN_FILTER_KSIZE = len(input_frames)
#     gaussian_filter = cv2.getGaussianKernel(GAUSSIAN_FILTER_KSIZE, -1)
#     with torch.no_grad():
#         summed_imgs = torch.zeros_like(base_frame)
#         summed_masks = torch.zeros_like(base_frame)
#         for idx, img in enumerate(input_frames):
#             _, ref_frame_flow = flow_model(img.detach(), base_frame.detach(), iters=20, test_mode=True)
#             tenOnes = torch.ones_like(img)
#
#             tenWarped = softsplat.FunctionSoftsplat(tenInput=img, tenFlow=ref_frame_flow + forward_flow, tenMetric=None, strType='average')
#             tenMask = softsplat.FunctionSoftsplat(tenInput=tenOnes, tenFlow=ref_frame_flow + forward_flow, tenMetric=None, strType='average')
#             # imwrite(img, out_path[:-4]+'_img_'+str(idx)+'.png', range=(0, 1))
#             # imwrite(tenWarped, out_path[:-4] + '_warped_' + str(idx) + '.png', range=(0, 1))
#             # imwrite(base_frame, out_path[:-4] + '_base_.png', range=(0, 1))
# quaternion_difference_rotation_matrix
#
#             summed_imgs += (tenWarped * gaussian_filter[idx, 0] * tenMask)
#             summed_masks += (gaussian_filter[idx, 0] * tenMask)
#         output = (summed_imgs / (torch.clamp(summed_masks, min=1e-6)))
#         output_mask = (summed_masks / (torch.clamp(summed_masks, min=1e-6)))
#     return output, output_mask
#
# def score_max(x, dim, score):
#     _tmp = [1] * len(x.size())
#     _tmp[dim] = x.size(dim)
#     return torch.gather(x, dim, score.max(dim)[1].unsqueeze(dim).repeat(tuple(_tmp))).select(dim, 0)
#
# def create_forward_max_proxy(forward_flow, input_frames, base_frame, flow_model, out_path):
#     GAUSSIAN_FILTER_KSIZE = len(input_frames)
#     gaussian_filter = cv2.getGaussianKernel(GAUSSIAN_FILTER_KSIZE, -1)
#     with torch.no_grad():
#         color_tensor = []
#         weight_tensor = []
#         for idx, img in enumerate(input_frames):
#             _, ref_frame_flow = flow_model(img.detach(), base_frame.detach(), iters=20, test_mode=True)
#             tenOnes = torch.ones_like(img)
#
#             tenWarped = softsplat.FunctionSoftsplat(tenInput=img, tenFlow=ref_frame_flow + forward_flow, tenMetric=None, strType='average')
#             tenMask = softsplat.FunctionSoftsplat(tenInput=tenOnes, tenFlow=ref_frame_flow + forward_flow, tenMetric=None, strType='average')
#
#             color_tensor.append(tenWarped)
#             weight_tensor.append(gaussian_filter[idx, 0] * tenMask)
#
#         color_tensor = torch.stack(color_tensor, 0)
#         weight_tensor = torch.stack(weight_tensor, 0)
#         output_mask = torch.sum(weight_tensor, dim=0) / torch.clamp(torch.sum(weight_tensor, dim=0), 1e-6)
#         output = score_max(color_tensor, 0, weight_tensor)
#     return output, output_mask


class adobe240fps:
    def __init__(self, input_dir, args):
        self.args = args
        #  self.testing_set_name = ['debug']
        # self.testing_set_name = ['IMG_0030', 'IMG_0049', 'IMG_0021', '720p_240fps_2', 'IMG_0032', 'IMG_0033', 'IMG_0031', 'IMG_0003', 'IMG_0039', 'IMG_0037']
        self.testing_set_name = [
            "IMG10_0002",
            "IMG11_0002",
            "IMG14_0003",
            "IMG15_0003",
            "IMG16_0002",
            "IMG18_0002",
            "IMG19_0003",
            "IMG19_0004",
        ]

        self.list_first_frame = []
        for sq in self.testing_set_name:
            self.list_first_frame.append(os.path.join(input_dir, sq))

        # 獲取每個影片的所有幀的路徑，並排除每個影片的最後六幀
        self.triplet_list = []
        self.frame_dof_list = []
        for sq in self.list_first_frame:
            all_frames = sorted(glob.glob(os.path.join(sq, "GT/*.jpg")))
            all_quaternions = pd.read_csv(
                os.path.join(sq, "frame_6dof.csv"), header=None
            )
            all_quaternions_np = all_quaternions.values
            self.triplet_list.extend(
                all_frames[:-6]
            )  #  移除最後六幀，僅保留列表中的第一幀
            self.frame_dof_list.extend(all_quaternions_np)

        self.transform = transforms.Compose(
            [transforms.RandomCrop((576, 1024)), transforms.ToTensor()]
        )

        self.transform_long = transforms.Compose(
            [transforms.RandomCrop((576, 1024)), transforms.ToTensor()]
        )
        self.firsr_time = True
        self.frame_dofs = []
        self.gt_frame_dof = []
        self.input0_list = []
        self.input1_list = []
        self.input2_list = []
        self.input3_list = []
        self.input4_list = []
        self.input5_list = []
        self.input6_list = []
        self.input0_dof_list = []
        self.input1_dof_list = []
        self.input2_dof_list = []
        self.input3_dof_list = []
        self.input4_dof_list = []
        self.input5_dof_list = []
        self.input6_dof_list = []
        self.gt_list = []
        self.im_list = []
        self.degree = []
        for frame in self.triplet_list:
            name = os.path.split(frame)[-1][:-4]
            # print(name)
            dir_name = os.path.dirname(frame)
            self.input0_list.append(
                os.path.join(dir_name, str(int(name)).zfill(5) + ".jpg")
            )
            self.input1_list.append(
                os.path.join(dir_name, str(int(name) + 1).zfill(5) + ".jpg")
            )
            self.input2_list.append(
                os.path.join(dir_name, str(int(name) + 2).zfill(5) + ".jpg")
            )
            self.input3_list.append(
                os.path.join(dir_name, str(int(name) + 3).zfill(5) + ".jpg")
            )
            self.input4_list.append(
                os.path.join(dir_name, str(int(name) + 4).zfill(5) + ".jpg")
            )
            self.input5_list.append(
                os.path.join(dir_name, str(int(name) + 5).zfill(5) + ".jpg")
            )
            self.input6_list.append(
                os.path.join(dir_name, str(int(name) + 6).zfill(5) + ".jpg")
            )
            self.gt_list.append(
                os.path.join(dir_name, str(int(name) + 3).zfill(5) + ".jpg")
            )
            self.im_list.append(
                dir_name.split("/")[-2]
                + "/"
                + dir_name.split("/")[-1]
                + "/"
                + str(int(name) + 3).zfill(5)
            )
        #   print(dir_name, str(int(name)).zfill(5)+'.jpg')

        for idx in range(len(self.frame_dof_list) - 6):
            self.input0_dof_list.append(self.frame_dof_list[idx])
            self.input1_dof_list.append(self.frame_dof_list[idx + 1])
            self.input2_dof_list.append(self.frame_dof_list[idx + 2])
            self.input3_dof_list.append(self.frame_dof_list[idx + 3])
            self.input4_dof_list.append(self.frame_dof_list[idx + 4])
            self.input5_dof_list.append(self.frame_dof_list[idx + 5])
            self.input6_dof_list.append(self.frame_dof_list[idx + 6])

        #    print(len(self.input6_dof_list),len(self.input1_dof_list),len(self.input2_list),len(self.input3_list),len(self.input4_list),len(self.input5_list),len(self.input6_list),len(self.triplet_list))

        input_size = 4  # 输入特征数
        hidden_size = 128  # 隐藏层大小
        output_size = 4  # 输出特征数，与输入相同
        num_layers = 2  # Transformer 层数
        num_heads = 4  # 注意力头的数量
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
        # parser = argparse.ArgumentParser()
        # args = parser.parse_args()

        self.optimizer = None
        self.flow_model = torch.nn.DataParallel(RAFT(args))
        self.flow_model.load_state_dict(torch.load("raft_models/raft-things.pth"))
        self.flow_model = self.flow_model.module
        self.flow_model.to("cuda")
        self.flow_model.eval()

    def Test(
        self, model, output_dir, current_epoch, logfile=None, output_name="output.png"
    ):
        torch.manual_seed(1)
        random.seed(1)
        is_training = False
        model.eval()
        av_psnr = 0
        if logfile is not None:
            logfile.write("{:<7s}{:<3d}".format("Epoch: ", current_epoch) + "\n")
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + "/" + self.im_list[idx]):
                os.makedirs(output_dir + "/" + self.im_list[idx])

            if "IMG_0049" in self.input0_list[idx]:
                _transform = self.transform_long
            else:
                _transform = self.transform

            input_frames = []
            input_frames.append(
                to_variable(_transform(Image.open(self.input0_list[idx])).unsqueeze(0))
            )
            input_frames.append(
                to_variable(_transform(Image.open(self.input1_list[idx])).unsqueeze(0))
            )
            input_frames.append(
                to_variable(_transform(Image.open(self.input2_list[idx])).unsqueeze(0))
            )
            base_frame = to_variable(
                _transform(Image.open(self.input3_list[idx])).unsqueeze(0)
            )
            # input_frames.append(to_variable(_transform(Image.open(self.input3_list[idx])).unsqueeze(0)))
            input_frames.append(base_frame)
            input_frames.append(
                to_variable(_transform(Image.open(self.input4_list[idx])).unsqueeze(0))
            )
            input_frames.append(
                to_variable(_transform(Image.open(self.input5_list[idx])).unsqueeze(0))
            )
            input_frames.append(
                to_variable(_transform(Image.open(self.input6_list[idx])).unsqueeze(0))
            )
            gt = to_variable(_transform(Image.open(self.gt_list[idx])).unsqueeze(0))

            self.frame_dofs = []
            self.frame_dofs.append(self.input0_dof_list[idx])
            self.frame_dofs.append(self.input1_dof_list[idx])
            self.frame_dofs.append(self.input2_dof_list[idx])
            self.frame_dofs.append(self.input3_dof_list[idx])
            self.frame_dofs.append(self.input4_dof_list[idx])
            self.frame_dofs.append(self.input5_dof_list[idx])
            self.frame_dofs.append(self.input6_dof_list[idx])

            # RAFT
            with torch.no_grad():
                if self.args.bundle_forward_flow > 0:
                    _, F_kprime_to_k = self.flow_model(
                        base_frame.detach(), gt.detach(), iters=20, test_mode=True
                    )
                else:
                    _, F_kprime_to_k = self.flow_model(
                        gt.detach(), base_frame.detach(), iters=20, test_mode=True
                    )
                F_n_to_k_s = []
                F_k_to_n_s = []
                # print(F_kprime_to_k)

                self.one_quat = [
                    torch.tensor(sublist[:4]) for sublist in self.frame_dofs
                ]
                one_quat_stacked = torch.stack(self.one_quat, dim=0)
                one_quat_stacked = one_quat_stacked.unsqueeze(0)
                one_quat_stacked_squeezed = torch.squeeze(one_quat_stacked, dim=0)
                #     print(one_quat_stacked.shape)
                smooth_out = self.mymodel(one_quat_stacked)
                # 計算四元數的差異
                q_diff = quaternion_multiply(
                    one_quat_stacked_squeezed, quaternion_inverse(smooth_out)
                )
                #    print(smooth_out)
                # 將四元數的差異轉換為旋轉矩陣
                rotation_matrices = quaternion_to_rotation_matrix(q_diff)

                # 移除添加的批次维度
                # rotated_image_tensor = rotated_image_tensor.squeeze(0)  # 移除批次维度
                rotated_images = []
                self.degree = []
                disparity_image = []
                for i, rot_matrix in enumerate(rotation_matrices):
                    euler_angles = rotation_matrix_to_euler_angles(rot_matrix)
                    # 提取 yaw 角度
                    yaw_angle_degrees = torch.rad2deg(euler_angles[2])
                    self.degree.append(yaw_angle_degrees)
                    #     print("degree", yaw_angle_degrees)

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

                    # 計算缺失像素的mask
                    mask = np.array(rotated_pil_image)
                    mask = np.where(mask == 0, 255, 0).astype(np.uint8)

                    # 將mask轉換為PIL Image對象
                    mask_pil_image = Image.fromarray(mask)
                    # mask_pil_image.save("mask_pil_image.jpg")
                    # print(rotated_pil_image)
                    # 将裁剪后的图像转换为张量
                    rotated_tensor = ToTensor()(mask_pil_image).unsqueeze(0)
                    rotated_tensor = rotated_tensor.to(input_frames[i].device)
                    rotated_images.append(rotated_tensor)

                    disparity_image.append(input_frames[i] - rotated_tensor)

                #      print(disparity_image[i].shape)
                # if i == 2:
                #     base_frame = base_frame.squeeze(
                #         0
                #     ).cpu()  # 将张量移到 CPU 上并去除批次维度
                #     gt = gt.squeeze(0).cpu()  # 将张量移到 CPU 上并去除批次维度
                #     # 将 PyTorch 张量转换为 NumPy 数组，并将其乘以 255 并转换为 uint8 类型
                #     np_image = (base_frame.numpy() * 255).astype(np.uint8)
                #     np_image_gt = (gt.numpy() * 255).astype(np.uint8)
                #     # 将 NumPy 数组转换为 PIL 图像
                #     pil_image = Image.fromarray(
                #         np_image.transpose(1, 2, 0)
                #     )  # 转换为 HWC 格式
                #     pil_image_gt = Image.fromarray(
                #         np_image_gt.transpose(1, 2, 0)
                #     )  # 转换为 HWC 格式
                #     # 将图像逆时针旋转 yaw 角度
                #     rotated_pil_image = TF.rotate(pil_image, yaw_angle_degrees.item())
                #     rotated_pil_image_gt = TF.rotate(
                #         pil_image_gt, yaw_angle_degrees.item()
                #     )
                #     base_frame = ToTensor()(rotated_pil_image).unsqueeze(0)
                #     gt = ToTensor()(rotated_pil_image_gt).unsqueeze(0)
                # #   print(rotated_images[0].shape)
                # #   imwrite(rotated_images[0], output_dir + '/' + self.im_list[idx] + '/' + "112233" +output_name, range=(0, 1))
                # #  rotated_image_tensor = TF.rotate(processed_image_tensor, angle)
                # base_frame = base_frame.to("cuda:0")
                # gt = gt.to("cuda:0")

                # input_frames = [frame.to("cuda:0") for frame in rotated_images]
                # base_frame = input_frames[3]
                # base_frame = base_frame.to("cuda:0")
                for img in input_frames:
                    _, ref_frame_flow = self.flow_model(
                        img.detach(), base_frame.detach(), iters=20, test_mode=True
                    )
                    F_n_to_k_s.append(ref_frame_flow)
                    _, ref_frame_flow2 = self.flow_model(
                        base_frame.detach(), img.detach(), iters=20, test_mode=True
                    )
                    F_k_to_n_s.append(ref_frame_flow2)

                #        rotation_matrix = quaternion_difference_rotation_matrix(one_quat_stacked,smooth_out)
                #   print("Test",rotation_matrix)
                frame_out = model(
                    input_frames,
                    F_kprime_to_k.detach(),
                    F_n_to_k_s,
                    F_k_to_n_s,
                    disparity_image,
                )
                # gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr
            imwrite(
                frame_out,
                output_dir + "/" + self.im_list[idx] + "/" + output_name,
                range=(0, 1),
            )
            imwrite(
                gt,
                output_dir
                + "/"
                + self.im_list[idx]
                + "/"
                + output_name[:-4]
                + "_GT.png",
                range=(0, 1),
            )
            msg = "{:<15s}{:<20.16f}".format(self.im_list[idx] + ": ", psnr) + "\n"
            print(msg, end="")
            if logfile is not None:
                logfile.write(msg)
        av_psnr /= len(self.im_list)
        msg = "{:<15s}{:<20.16f}".format("Average: ", av_psnr) + "\n"
        print(msg, end="")
        if logfile is not None:
            logfile.write(msg)
