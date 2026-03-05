__all__ = ["train_step","val_step"]

import torch

def  train_step(denoiser,train_loder,loss_fn,optimizer,device):

    total_loss = 0
    num_batches = 0
    for train_img,target_img in train_loder:
        train_img = target_img.to(device)
        target_img = target_img.to(device)
        optimizer.zero_grad()
        output = denoiser(train_img)
        loss = loss_fn(output,target_img)
        loss.backward()
        optimizer.step()
        total_loss +=loss.item()
        num_batches +=1
    return  total_loss/num_batches


def val_step(denoiser,val_loder,loss_fn,device):
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for train_img,target_img in val_loder:
            train_img = target_img.to(device)
            target_img = target_img.to(device)

            output = denoiser(train_img)
            loss = loss_fn(output,target_img)
            total_loss +=loss.item()
            num_batches +=1
    return total_loss/num_batches




