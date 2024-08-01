import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datareader import DBreader_RVLKC
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
import os
from loss import CustomLoss
import torch.nn.init as init
import matplotlib.pyplot as plt
writer = SummaryWriter("./log")
import torch.nn.functional as F

quaternions_list = []

def evaluate_model(model, test_loader):
    model.eval()  # 将模型设置为评估模式
    total_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs = data[0]
            inputs = inputs.unsqueeze(0)
            inputs = inputs[:, :, :4]  # 假设你想要使用前 4 个特征
            inputs = inputs.float()
            # 添加高斯噪声到输入数据
            outputs = model(inputs)
            loss = custom_loss(outputs, inputs.squeeze(0))
            total_loss += loss.item() * inputs.size(0)  # 累计损失
            # 可选地，你可以打印每个输入的输出
            print("当前输入的输出:", outputs)
    avg_loss = total_loss / len(test_loader.dataset)  # 计算平均损失
    return avg_loss
    
    


def collate_fn(batch):
    """
    定义用于处理不同大小样本的函数
    """
    # 过滤掉样本大小为0的样本
    batch = list(filter(lambda x: x is not None, batch))
    # 获取批次中样本的大小
    batch_sizes = [len(data) for data in batch]
    # 如果批次中的所有样本大小都相同，则返回原批次
    if all(batch_size == batch_sizes[0] for batch_size in batch_sizes):
        return default_collate(batch)
    else:
        raise RuntimeError('each element in list of batch should be of equal size')

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
        # 解码器部分
        decoder_output = self.transformer_decoder(tgt=x, memory=encoder_output)
        
        # 进行全连接和归一化
        x = self.fc(decoder_output[-1])
        x = self.batchnorm(x)
        
        return x

# 创建模型实例
input_size = 4  # 输入特征数
hidden_size = 128  # 隐藏层大小
output_size = 4  # 输出特征数，与输入相同
num_layers = 2  # Transformer 层数
num_heads = 4  # 注意力头的数量
dropout = 0.1  # Dropout 概率
weight_decay = 1e-5  # L2 正则化系数
model = TransformerModel(input_size=4, hidden_size=64, output_size=4, num_layers=2)

# 创建虚拟训练数据
# 假设你有一个名为 train_data 的张量，其形状为 (batch_size, sequence_length, input_size)
batch_size = 7
dataset = DBreader_RVLKC("/home/kang/SmoothTracjectory/dataset/train")
test_dataset = DBreader_RVLKC("/home/kang/SmoothTracjectory/dataset/test")
#"/home/kang/SmoothTracjectory/dataset/train"
#/home/kang/FuSta/DeepVideoDeblurring_Dataset/quantitative_datasets
# 定义数据加载器
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True,collate_fn=collate_fn)
test_loader =  DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True,collate_fn=collate_fn)
# 定义损失函数和优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=weight_decay)
custom_loss = CustomLoss()
# 训练模型

# 检查是否存在 "checkpoints" 文件夹，如果不存在则创建
checkpoint_dir = 'checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
num_epochs = 30
total_steps = len(train_loader)
global_step = 0  # 全局步数，用于 TensorBoard
first = False
print("total_steps",total_steps)
train_results= []
for epoch in range(num_epochs):
    for i,data in enumerate(train_loader):
    
        inputs = data[0]  # 获取输入数据
        
        if first == False and epoch == 19 :
            np.save('train_results.npy', inputs.detach().numpy())
        inputs = inputs.unsqueeze(0)  # 在第二个维度添加一个维度，变为 (7, 1, 7)
        inputs = inputs[:,:, :4]
        
     #   print("inputs",inputs.shape)
        inputs = inputs.float()  
        outputs = model(inputs)
        loss = custom_loss(outputs, inputs.squeeze(0))
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    #    print("outputs",outputs.shape)
         # 在每个步骤记录损失到 TensorBoard
        writer.add_scalar('Loss/train', loss.item(), global_step)
        
        # 更新全局步数
        global_step += 1
        
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')
            
            # Run evaluation every 2 epochs
    if (epoch + 1) % 2 == 0:
        test_loss = evaluate_model(model, test_loader)
     #   testtrain_loss = evaluate_model1(model, train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss}')
        
    if first == False and epoch == 19:
        np.save('inference_results.npy', outputs.detach().numpy())
        first = True
    # 每个 epoch 结束后保存模型的 checkpoint
    
    
# 将推理结果保存为 Numpy 数组
    train_results_array = np.array(train_results)

# 可选择保存推理结果
    #np.save('train_results.npy', train_results_array)
    torch.set_printoptions(precision=4)  
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}_checkpoint.pth')
    torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss,}, checkpoint_path)





