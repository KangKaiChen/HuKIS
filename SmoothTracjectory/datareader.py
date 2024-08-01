import numpy as np
from os import listdir
from PIL import Image
from os.path import join, isdir
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import pandas as pd
import random
import os
import glob



class DBreader_RVLKC(Dataset):
    def __init__(self, db_dir):

        
        # 設定測試集的影片名稱
      #  self.testing_set_name = ['IMG_0030', 'IMG_0049', 'IMG_0021', '720p_240fps_2', 'IMG_0032', 'IMG_0033', 'IMG_0031', 'IMG_0003', 'IMG_0039', 'IMG_0037']
      #  self.testing_set_name = ['IMG_0011']
        
        # 取得所有影片的第一張圖片的路徑
        self.list_first_frame = sorted(glob.glob(os.path.join(db_dir, '**/')))
     #   for test_string in self.testing_set_name:
            # 從第一張圖片的路徑中排除測試集的影片
      #      self.list_first_frame = [x for x in self.list_first_frame if test_string not in x]
            
    
        # 獲取每個影片的所有幀的路徑，並排除每個影片的最後六幀
        self.triplet_list = []
        self.frame_dof_list = []
        for sq in self.list_first_frame:
            all_quaternions = pd.read_csv(os.path.join(sq, 'frame_6dof.csv'))  # 读取四元数信息的 CSV 文
            all_quaternions_np = all_quaternions.values
            self.frame_dof_list.extend(all_quaternions_np[:-5])
            
            
     #   self.frame_dof_list = np.array(self.frame_dof_list)
        # 計算資料集中的檔案數量
        self.file_len = len(self.frame_dof_list)
        if len(self.triplet_list) != len(self.frame_dof_list):
            print("Eumber of  quaternions ",len(self.frame_dof_list))
               

    def __getitem__(self, index):
    
            # 根據索引獲取資料集中的單個樣本，並讀取圖像序列
            
                  
        frame_dofs = self.frame_dof_list[index:index+7]
    # 如果剩余样本数不足7个，则循环取前面的样本来填充到7个
        while len(frame_dofs) < 7:
            frame_dofs.append(frame_dofs[len(frame_dofs) % len(frame_dofs)])
        
    # 返回填充后的样本
        return frame_dofs

    def __len__(self):
        return self.file_len

