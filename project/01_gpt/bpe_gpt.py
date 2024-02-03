"""
训练BPE级别的英文小说模型
"""
# -----------------------------------------1、导入必要的包-----------------------------------------
import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model_bpe import GPT
from mingpt.trainer_bpe import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
from mingpt.bpe import BPETokenizer

# -----------------------------------------2、获取配置信息-----------------------------------------
"""
基于字符编码的数据集类
将raw text按照block size的字节数切分，每次取一个block，基于next token prediction目标训练
next token prediction预训练数据的构建：token shift
"""


class BPEDataset(Dataset):
    """
    Emits batches of BPE
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128  # 将raw data切分为长度为block size的文本块进行next token prediction训练
        return C

    def __init__(self, config, data):
        self.config = config
        self.tokenizer = BPETokenizer()
        # 256 individual byte tokens, 50,000 merged tokens, and 1 special <|endoftext|> token
        self.vocab_size = len(self.tokenizer.encoder.encoder)
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        # 滑动窗口式构建训练数据（真正LLM Pretrain时，因为数据比较充足不需要这么做）
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        # 截取一个block-size的文本
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = self.tokenizer(chunk).squeeze(0)  # 转为int序列
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)  # 转为tensor
        y = torch.tensor(dix[1:], dtype=torch.long)  # 通过一次shift构建训练数据
        return x, y


def get_config():
    """
    获取配置信息
    """

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407  # 随机种子
    C.system.work_dir = './test_out/bpegpt'  # 输出路径

    # data
    C.data = BPEDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4  # TODO: 实践不同的LR对结果的影响

    return C


if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()  # 将所有配置信息存储在config中
    config.merge_from_args(sys.argv[1:])  # 从命令行参数初始化配置信息
    print(config)  # 打印配置信息
    setup_logging(config)  # 设置日志
    set_seed(config.system.seed)  # 设置随机种子，保证实验的可重复性

    # construct the training dataset
    # don't worry we won't run out of file handles
    text = open('./data/train.txt', 'r').read()
    train_dataset = BPEDataset(config.data, text)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):  # 每个batch结束的回调，用于打印训练loss和测试结果

        if trainer.iter_num % 10 == 0:
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                context = "O God, O God!"
                x = train_dataset.tokenizer(context).to(trainer.device)
                y = model.generate(x, 500, temperature=1.0,
                                   do_sample=True, top_k=10)[0]
                completion = train_dataset.tokenizer.decode(y)
                print(completion)
                # TODO:创建测试集，计算并打印test loss

            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
