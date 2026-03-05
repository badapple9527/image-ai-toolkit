#导入库
import numpy as np#数学计算库
import os #接口库
import torch#深度学习框架
import random#随机数
def seed_everything(seed):
    """
    为了保证训练过程可复现，使用确定的随机数种子。对 torch，numpy 和 random 都使用相同的种子。
    设计参数：

    参数:
    - seed: 随机数种子（整数）
    """
    random.seed(seed)#设置随机种子
    os.environ["PYTHONHASHSEED"] = str(seed) #python内部随机种子
    np.random.seed(seed) #numpy随机数种子
    torch.manual_seed(seed) #toach cpu种子
    torch.cuda.manual_seed(seed) #toach gpu种子
    torch.backends.cudnn.deterministic = True # 确保CuDNN操作确定性
    torch.backends.cudnn.benchmark = False # 禁用CuDNN性能优化


