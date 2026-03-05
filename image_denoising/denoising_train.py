import torch
import denoising_model
import denoising_engine
import torchvision.transforms as T
import denoising_data
import denoising_config
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import  torch.optim as optim

from common import  utils


def test(denoiser,val_loder,device):
    import matplotlib.pyplot as plt
    dataiter = iter(val_loder)
    noisy_images,original_images = next(dataiter)
    print("测试集 images 形状: ", noisy_images.shape)
    denoiser = denoiser.to(device)
    noisy_images = noisy_images.to(device)
    print("测试集 noisy_images 形状: ", noisy_images.shape)
    output = denoiser(noisy_images)
    print("测试集输出结果 output 形状: ", output.shape)
    noisy_images = noisy_images.cpu().numpy()
    print('noisy_imgs 转换为 numpy 数组后的形状: ', noisy_images.shape)
    noisy_images = np.moveaxis(noisy_images,1,-1)
    output = output.view(denoising_config.TEST_BATCH_SIZE,3,64,64)
    output = output.detach().cpu().numpy()
    print('output 转换为 numpy 数组后的形状: ', output.shape)
    output = np.moveaxis(output,1,-1)
    print('output 的通道维度移到最后一维后的形状: ', output.shape)
    original_images = original_images.cpu().numpy().transpose((0,2,3,1))
    print("original_image shape: ", original_images.shape)
    fig,axes = plt.subplots(nrows=3,ncols=10,sharex=True,sharey=True,figsize   =(25,4))
    for imgs,row in zip([noisy_images,original_images,output],axes):
        for img, ax in zip(imgs, row):  # 遍历每张图像和对应的子图
            ax.imshow(np.squeeze(img))  # 显示图像，并去除多余的维度
            ax.get_xaxis().set_visible(False)  # 隐藏 x 轴
            ax.get_yaxis().set_visible(False)  # 隐藏 y 轴
        plt.show()  # 显示图像
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("设置训练去噪模型的随机数种子, seed = {}".format(denoising_config.SEED))
    utils.seed_everything(denoising_config.SEED)
    transforms = T.Compose([T.Resize((64,64)),T.ToTensor()])
    print("------------ 正在创建数据集 ------------")
    full_dataset = denoising_data.ImageDataset(denoising_config.IMG_PATH,transforms)
    train_size = int(denoising_config.TRAIN_RATIO*len(full_dataset))
    val_size = len(full_dataset)-train_size
    train_dataset,val_dataset = torch.utils.data.random_split(
        full_dataset,[train_size,val_size]
    )
    print("------------ 数据集创建完成 ------------")
    print("------------ 创建数据加载器 ------------")
    train_loder = torch.utils.data.DataLoader(
        train_dataset,batch_size=denoising_config.TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last = True
    )
    val_loder = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=denoising_config.TEST_BATCH_SIZE
    )
    print("------------ 数据加载器创建完成 ------------")
    denoiser = denoising_model.ConvDenoiser()
    loss_fn = nn.MSELoss()
    denoiser.to(device)
    optimizer = torch.optim.Adam(denoiser.parameters(),lr=denoising_config.LEARNING_RATE)

    min_loss = 9999
    print("------------ 开始训练 ------------")
    for epoch in tqdm(range(denoising_config.EPOCHS)):
        train_loss = denoising_engine.train_step(
            denoiser, train_loder, loss_fn, optimizer, device=device
        )
        print(f"\n----------> Epochs = {epoch + 1}, Training Loss : {train_loss} <----------")
        val_loss = denoising_engine.val_step(

            denoiser, val_loder, loss_fn, device = device
        )
        if val_loss<min_loss:
            print("验证集的损失减小了，保存新的最好的模型。")
            min_loss = val_loss
            torch.save(denoiser.state_dict(),denoising_config.DENOISER_MODEL_NAME)
        else:
            print("验证集的损失没有减小，不保存模型。")
        print(f"Epochs = {epoch + 1}, Validation Loss : {val_loss}")
    print("\n==========> 训练结束 <==========\n")

    print("本次训练的去噪模型测试结果如下")

    test(denoiser, val_loder, device)

    print("================> 从磁盘加载模型 <================")
    load_denoiser = denoising_model.ConvDenoiser()
    load_denoiser.load_state_dict(torch.load(denoising_config.DENOISER_MODEL_NAME, map_location=device))

    load_denoiser.to(device)

    print("从磁盘加载的去噪模型测试结果如下")

    test(load_denoiser, val_loder, device)






