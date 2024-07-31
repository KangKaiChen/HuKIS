import glob
import os
import random
from os import listdir
from os.path import isdir, join

import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import core.utils.data_transforms


def cointoss(p):
    return random.random() < p


class DBreader_Vimeo90k(Dataset):
    def __init__(
        self, db_dir, random_crop=None, resize=None, augment_s=True, augment_t=True
    ):
        # db_dir += '/sequences'
        self.random_crop = random_crop
        self.augment_s = augment_s
        self.augment_t = augment_t

        transform_list = []
        if resize is not None:
            transform_list += [transforms.Resize(resize)]

        transform_list += [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

        with open(os.path.join(db_dir, "sep_trainlist.txt")) as f:
            self.folder_list = f.readlines()
        self.triplet_list = [
            db_dir + "/sequences/" + x.strip() for x in self.folder_list
        ]
        # print(self.triplet_list)

        # self.folder_list = [(db_dir + '/' + f) for f in listdir(db_dir) if isdir(join(db_dir, f))]
        # self.triplet_list = []
        # for folder in self.folder_list:
        #     self.triplet_list += [(folder + '/' + f) for f in listdir(folder) if isdir(join(folder, f))]

        self.triplet_list = np.array(self.triplet_list)
        self.file_len = len(self.triplet_list)

    def __getitem__(self, index):
        rawFrame0 = Image.open(self.triplet_list[index] + "/im1.png")
        rawFrame1 = Image.open(self.triplet_list[index] + "/im2.png")
        rawFrame2 = Image.open(self.triplet_list[index] + "/im3.png")
        rawFrame3 = Image.open(self.triplet_list[index] + "/im4.png")
        rawFrame4 = Image.open(self.triplet_list[index] + "/im5.png")
        rawFrame5 = Image.open(self.triplet_list[index] + "/im6.png")
        rawFrame6 = Image.open(self.triplet_list[index] + "/im7.png")

        if self.random_crop is not None:
            i, j, h, w = transforms.RandomCrop.get_params(
                rawFrame3, output_size=self.random_crop
            )
            rawFrame0 = TF.crop(rawFrame0, i, j, h, w)
            rawFrame1 = TF.crop(rawFrame1, i, j, h, w)
            rawFrame2 = TF.crop(rawFrame2, i, j, h, w)
            rawFrame3 = TF.crop(rawFrame3, i, j, h, w)
            rawFrame4 = TF.crop(rawFrame4, i, j, h, w)
            rawFrame5 = TF.crop(rawFrame5, i, j, h, w)
            rawFrame6 = TF.crop(rawFrame6, i, j, h, w)

        if self.augment_s:
            if cointoss(0.5):
                rawFrame0 = TF.hflip(rawFrame0)
                rawFrame1 = TF.hflip(rawFrame1)
                rawFrame2 = TF.hflip(rawFrame2)
                rawFrame3 = TF.hflip(rawFrame3)
                rawFrame4 = TF.hflip(rawFrame4)
                rawFrame5 = TF.hflip(rawFrame5)
                rawFrame6 = TF.hflip(rawFrame6)
            if cointoss(0.5):
                rawFrame0 = TF.vflip(rawFrame0)
                rawFrame1 = TF.vflip(rawFrame1)
                rawFrame2 = TF.vflip(rawFrame2)
                rawFrame3 = TF.vflip(rawFrame3)
                rawFrame4 = TF.vflip(rawFrame4)
                rawFrame5 = TF.vflip(rawFrame5)
                rawFrame6 = TF.vflip(rawFrame6)

        frame0 = self.transform(rawFrame0)
        frame1 = self.transform(rawFrame1)
        frame2 = self.transform(rawFrame2)
        frame3 = self.transform(rawFrame3)
        frame4 = self.transform(rawFrame4)
        frame5 = self.transform(rawFrame5)
        frame6 = self.transform(rawFrame6)

        if self.augment_t:
            if cointoss(0.5):
                return frame6, frame5, frame4, frame3, frame2, frame1, frame0
            else:
                return frame0, frame1, frame2, frame3, frame4, frame5, frame6
        else:
            return frame0, frame1, frame2, frame3, frame4, frame5, frame6

    def __len__(self):
        return self.file_len


class DBreader_Adobe240fps(Dataset):
    def __init__(
        self, db_dir, random_crop=None, resize=None, augment_s=True, augment_t=True
    ):
        # 初始化方法，用於建立資料集物件
        # db_dir: 資料庫目錄
        # random_crop: 隨機裁切的大小
        # resize: 調整大小的參數
        # augment_s: 是否對空間進行增強 (預設為 True)
        # augment_t: 是否對時間進行增強 (預設為 True)

        # 設定測試集的影片名稱
        #  self.testing_set_name = ['IMG_0030', 'IMG_0049', 'IMG_0021', '720p_240fps_2', 'IMG_0032', 'IMG_0033', 'IMG_0031', 'IMG_0003', 'IMG_0039', 'IMG_0037']
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
        self.random_crop = random_crop
        self.augment_s = augment_s
        self.augment_t = augment_t

        # 定義圖像轉換的操作序列
        self.transform_random_crop = transforms.Compose(
            [transforms.RandomCrop(self.random_crop)]
        )

        self.transforms = core.utils.data_transforms.Compose(
            [
                # 定義圖像增強的操作序列
                core.utils.data_transforms.ColorJitter(
                    [0.2, 0.15, 0.3, 0.1]
                ),  # 這個操作是對圖像進行顏色抖動處理，參數分別代表亮度、對比度、飽和度和色調的抖動範圍。
                core.utils.data_transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]
                ),  # 這個操作是對圖像進行標準化處理，將圖像的每個通道的像素值減去均值 mean 並除以標準差 std。
                core.utils.data_transforms.RandomCrop(
                    [int(self.random_crop[0] * 1.25), int(self.random_crop[1] * 1.25)]
                ),  # 這個操作是對圖像進行隨機裁切，裁切大小為指定的 random_crop 大小的 1.25 倍。
                core.utils.data_transforms.RandomVerticalFlip(),  # 這個操作是對圖像進行隨機垂直翻轉。
                core.utils.data_transforms.RandomHorizontalFlip(),  # 這個操作是對圖像進行隨機水平翻轉。
                core.utils.data_transforms.RandomColorChannel(),  # 這個操作是對圖像的顏色通道進行隨機調整。
                core.utils.data_transforms.RandomGaussianNoise(
                    [0, 1e-4]
                ),  # 這個操作是對圖像進行隨機高斯噪聲添加，參數 0,1e−4] 是高斯噪聲的均值和標準差。
                core.utils.data_transforms.ToTensor(),
            ]
        )

        # 取得所有影片的第一張圖片的路徑
        self.list_first_frame = sorted(glob.glob(os.path.join(db_dir, "**/")))
        for test_string in self.testing_set_name:
            # 從第一張圖片的路徑中排除測試集的影片
            self.list_first_frame = [
                x for x in self.list_first_frame if test_string not in x
            ]

        # 獲取每個影片的所有幀的路徑，並排除每個影片的最後六幀
        self.triplet_list = []
        self.frame_dof_list = []
        self
        for sq in self.list_first_frame:
            all_frames = sorted(glob.glob(os.path.join(sq, "GT/*.jpg")))
            all_quaternions = pd.read_csv(
                os.path.join(sq, "frame_6dof.csv"), header=None
            )  # 读取四元数信息的 CSV 文
            all_quaternions_np = all_quaternions.values
            #       print(sq, len(all_frames), len(all_quaternions_np))
            self.triplet_list.extend(
                all_frames[:-6]
            )  #  移除最後六幀，僅保留列表中的第一幀
            self.frame_dof_list.extend(all_quaternions_np)

        # 將列表轉換為 NumPy 陣列
        self.triplet_list = np.array(self.triplet_list)
        #   self.frame_dof_list = np.array(self.frame_dof_list)
        # 計算資料集中的檔案數量
        self.file_len = len(self.triplet_list)

    def __getitem__(self, index):
        # 根據索引獲取資料集中的單個樣本，並讀取圖像序列
        name = os.path.split(self.triplet_list[index])[-1][:-4]
        dir_name = os.path.dirname(self.triplet_list[index])  # 獲取圖像所在的目錄名稱

        rawFrame0 = Image.open(
            os.path.join(dir_name, str(int(name)).zfill(5) + ".jpg")
        )  # 讀取第一幀圖像
        rawFrame1 = Image.open(
            os.path.join(dir_name, str(int(name) + 1).zfill(5) + ".jpg")
        )  # 依次讀取後續幀的圖像
        rawFrame2 = Image.open(
            os.path.join(dir_name, str(int(name) + 2).zfill(5) + ".jpg")
        )
        rawFrame3 = Image.open(
            os.path.join(dir_name, str(int(name) + 3).zfill(5) + ".jpg")
        )
        rawFrame4 = Image.open(
            os.path.join(dir_name, str(int(name) + 4).zfill(5) + ".jpg")
        )
        rawFrame5 = Image.open(
            os.path.join(dir_name, str(int(name) + 5).zfill(5) + ".jpg")
        )
        rawFrame6 = Image.open(
            os.path.join(dir_name, str(int(name) + 6).zfill(5) + ".jpg")
        )

        input_frames = [
            rawFrame0,
            rawFrame1,
            rawFrame2,
            rawFrame3,
            rawFrame4,
            rawFrame5,
            rawFrame6,
        ]
        gt_frame = [rawFrame3]
        frame_dofs = self.frame_dof_list[index : index + 7]
        gt_frame_dof = frame_dofs[3]

        input_frames, gt_frame = self.transforms(input_frames, gt_frame)

        i, j, h, w = transforms.RandomCrop.get_params(
            gt_frame[0], output_size=self.random_crop
        )
        GT = transforms.ToTensor()(
            TF.crop(transforms.ToPILImage()(gt_frame[0]), i, j, h, w)
        )
        i, j, h, w = transforms.RandomCrop.get_params(
            input_frames[0], output_size=self.random_crop
        )
        input_frames[0] = transforms.ToTensor()(
            TF.crop(transforms.ToPILImage()(input_frames[0]), i, j, h, w)
        )
        i, j, h, w = transforms.RandomCrop.get_params(
            input_frames[1], output_size=self.random_crop
        )
        input_frames[1] = transforms.ToTensor()(
            TF.crop(transforms.ToPILImage()(input_frames[1]), i, j, h, w)
        )
        i, j, h, w = transforms.RandomCrop.get_params(
            input_frames[2], output_size=self.random_crop
        )
        input_frames[2] = transforms.ToTensor()(
            TF.crop(transforms.ToPILImage()(input_frames[2]), i, j, h, w)
        )
        i, j, h, w = transforms.RandomCrop.get_params(
            input_frames[3], output_size=self.random_crop
        )
        input_frames[3] = transforms.ToTensor()(
            TF.crop(transforms.ToPILImage()(input_frames[3]), i, j, h, w)
        )
        i, j, h, w = transforms.RandomCrop.get_params(
            input_frames[4], output_size=self.random_crop
        )
        input_frames[4] = transforms.ToTensor()(
            TF.crop(transforms.ToPILImage()(input_frames[4]), i, j, h, w)
        )
        i, j, h, w = transforms.RandomCrop.get_params(
            input_frames[5], output_size=self.random_crop
        )
        input_frames[5] = transforms.ToTensor()(
            TF.crop(transforms.ToPILImage()(input_frames[5]), i, j, h, w)
        )
        i, j, h, w = transforms.RandomCrop.get_params(
            input_frames[6], output_size=self.random_crop
        )
        input_frames[6] = transforms.ToTensor()(
            TF.crop(transforms.ToPILImage()(input_frames[6]), i, j, h, w)
        )

        base_frame = input_frames[3]

        return input_frames, base_frame, GT, frame_dofs, gt_frame_dof

    def __len__(self):
        return self.file_len
