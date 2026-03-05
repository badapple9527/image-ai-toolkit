import torch

import similarity_data#
import similarity_model#
import similarity_config#
import similarity_engine#
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import  torch.optim as optim

from common import  utils

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("设置训练分类模型的随机数种子, seed = {}".format(similarity_config.SEED))
    utils.seed_everything(similarity_config.SEED)
    transforms = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    print("------------ 正在创建数据集 ------------")
    full_dataset = similarity_data.ImageDataset(
        similarity_config.IMG_PATH,
        transforms
    )
    train_size = int(similarity_config.TRAIN_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    print("------------ 数据集创建完成 ------------")
    print("------------ 创建数据加载器 ------------")

    train_loder = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=similarity_config.TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    val_loder = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=similarity_config.TEST_BATCH_SIZE
    )
    full_loder = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=similarity_config.TEST_BATCH_SIZE
    )
    print("------------ 数据加载器创建完成 ------------")
    loss_fn = nn.MSELoss()
    encoder = similarity_model.ConvEncoder()
    decoder = similarity_model.ConvDecoder()
    encoder.to(device)
    decoder.to(device)
    autoencoder_params = list(encoder.parameters())+list(decoder.parameters())
    optimizer = optim.AdamW(autoencoder_params,lr =similarity_config.LEARNING_RATE)

    min_loss = 9999

    print("------------ 开始训练 ------------")

    for epoch in tqdm(range(similarity_config.EPOCHS)):
        train_loss = similarity_engine.train_step(
            encoder, decoder, train_loder, loss_fn, optimizer, device=device
        )
        print(f"\n----------> Epochs = {epoch + 1}, Training Loss : {train_loss} <----------")
        val_loss = similarity_engine.val_step(
            encoder, decoder,val_loder, loss_fn, device=device
        )
        if val_loss<min_loss:
            print("验证集的损失减小了，保存新的最好的模型。")
            min_loss = val_loss
            torch.save(encoder.state_dict(),similarity_config.ENCODER_MODEL_NAME)
            torch.save(encoder.state_dict(),similarity_config.DECODER_MODEL_NAME)
        else:
            print("验证集的损失没有减小，不保存模型。")
        print(f"Epochs = {epoch + 1}, Validation Loss : {val_loss}")
    print("\n==========> 训练结束 <==========\n")







