
import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import torch

# -----------------------------------------------------------------------------

def set_seed(seed):
    """
    设置种子，确保可复现性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(config):
    """
    日志文件初始化
    """
    work_dir = config.system.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)  # 递归创建目录
    # log the args (if any)
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    # log the config itself
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))

class CfgNode:
    """ 
    配置文件节点，树状结构存储配置信息
    """
    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        # 传入的关键字与参数将会自动设置为对象的属性
        self.__dict__.update(kwargs)  # self.__dict__为当前类的所有属性及其默认值

    def __str__(self):  #  返回一个对象的描述信息
        """
        打印配置信息
        """
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ 
        提高配置信息的打印美观度
        """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):  # 判断对象是否是指定类型的实例
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ 
        将配置信息转换为字典格式
        """
        return { k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items() }

    def merge_from_dict(self, d):
        """         
        配置信息的合并
        """
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        从命令行参数初始化配置信息
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:  # 解析参数

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)  # 将参数值转换为python对象（已经是字符串的无需转换）
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:] # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)  # 迭代式生成配置文件节点
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)