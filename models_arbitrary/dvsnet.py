import math
import torch
from collections import OrderedDict

import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
Activates = {"sigmoid": nn.Sigmoid, "relu": nn.ReLU, "tanh": nn.Tanh}
from torch.autograd import Variable

def torch_QuaternionProduct(q1, q2, USE_CUDA = True):
    x1 = q1[:,0]  
    y1 = q1[:,1]   
    z1 = q1[:,2]   
    w1 = q1[:,3]   

    x2 = q2[:,0]  
    y2 = q2[:,1]  
    z2 = q2[:,2]  
    w2 = q2[:,3]  

    batch_size = q1.size()[0]
    quat = Variable(torch.zeros((batch_size, 4), requires_grad=True))
    if USE_CUDA == True:
        quat = quat.cuda()
    
    quat[:,3] =  w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  
    quat[:,0] =  w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  
    quat[:,1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  
    quat[:,2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  

    quat = torch_norm_quat(quat)

    return quat



def torch_QuaternionReciprocal(q,  USE_CUDA = True):
    quat = torch.cat((-q[:,0:1], -q[:,1:2], -q[:,2:3], q[:,3:]), dim = 1) 
    batch_size = quat.size()[0]

    quat = torch_norm_quat(quat)
    return quat




def torch_norm_quat(quat, USE_CUDA = True):
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
            quat_out[i,:3] = quat[i,:3] * 0
            quat_out[i,3] = quat[i,3] / quat[i,3]

    # Method 2:
    # quat = quat / (torch.unsqueeze(torch.norm(quat, dim = 1), 1) + 1e-6) # check norm
    return quat_out

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, num_heads=2, dropout=0.1, weight_decay=0.01,noise_std=0.01):
        super(TransformerModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads),
            num_layers=num_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=input_size, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_size, output_size)
        self.batchnorm = nn.BatchNorm1d(output_size)
        self.dropout = nn.Dropout(dropout)
        self.weight_decay = weight_decay
        self.noise_std = noise_std
        
    def forward(self, x):
        x = x.float()
        
        # 编码器部分
        encoder_output = self.transformer_encoder(x)
        
        # 添加高斯噪声
  #      noise = torch.randn_like(encoder_output) * self.noise_std
  #      encoder_output = encoder_output + noise
        
        # 解码器部分
        decoder_output = self.transformer_decoder(tgt=x, memory=encoder_output)
        
        # 进行全连接和归一化
        x = self.fc(decoder_output[-1])
        x = self.batchnorm(x)
        
        return x






