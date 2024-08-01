import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datareader import DBreader_RVLKC  # 假设您的数据读取器在 datareader.py 中定义
import torch.nn.init as init
import torch.nn.functional as F
# 定义模型类
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

# 创建模型实例
input_size = 4  # 输入特征数
hidden_size = 64  # 隐藏层大小
output_size = 4  # 输出特征数，与输入相同
num_layers = 2  # Transformer 层数
num_heads = 4  # 注意力头的数量
dropout = 0.5  # Dropout 概率
model = TransformerModel(input_size, hidden_size, output_size, num_layers, num_heads)

# 加载检查点
checkpoint_path = 'checkpoints/epoch_16_checkpoint.pth'  # 假设是您最后一个检查点的路径
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

# 将模型设置为评估模式
model.eval()

# 加载数据集
batch_size = 7
dataset = DBreader_RVLKC("/home/kang/SmoothTracjectory/dataset/eval")
inference_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,drop_last=True)
total_steps = len(inference_loader)

print("total_steps",total_steps)
# 进行推理
inference_results = []
with torch.no_grad():
    for data in inference_loader:
        inputs = data[0]
        inputs = inputs.unsqueeze(0)
        inputs = inputs[:, :, :4]
      #  print("inputs",inputs)
     #   print("inputs",inputs)
        inputs = inputs.float()
        outputs = model(inputs)
        print("outputs",outputs)
        inference_results.append(outputs.detach().numpy())

# 将推理结果保存为 Numpy 数组
inference_results_array = np.array(inference_results)

# 可选择保存推理结果
np.save('inference_results.npy', inference_results_array)
