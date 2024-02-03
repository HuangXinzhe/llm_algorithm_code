import numpy as np
import torch

def positional_encoding(seq_len, d_model):
    # 创建一个形状为(seq_len, 1)的数组，其中的值为[0, 1, 2, ... seq_len-1]
    position = np.arange(seq_len)[:, np.newaxis]

    # 计算除数，这里的除数将用于计算正弦和余弦的频率
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # 初始化位置编码矩阵为零
    pe = np.zeros((seq_len, d_model))
    
    # 对矩阵的偶数列机型正弦函数编码
    pe[:, 0::2] = np.sin(position * div_term)

    # 对矩阵的奇数列机型余弦函数编码
    pe[:, 1::2] = np.cos(position * div_term)

    # 返回位置编码矩阵，转换为PyTorch张量
    return torch.tensor(pe, dtype=torch.float32)

if __name__ == '__main__':
    # 使用示例
    seq_len = 50  # 定义序列长度
    d_model = 512  # 定义模型的embedding维度
    pe = positional_encoding(seq_len, d_model)  # 获得位置编码
    print(pe)
    print(pe.shape)