__all__ = ["train_step","val_step"]

import torch
def  train_step(encoder,decoder,train_loder,loss_fn,optimizer,device):
    total_loss = 0
    num_batches = 0
    for train_img,target_img in train_loder:
        train_img = train_img.to(device)
        target_img = target_img.to(device)
        optimizer.zero_grad()

        enc_output = encoder(train_img)
        dec_output = decoder(enc_output)
        loss = loss_fn(dec_output, target_img)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches

def val_step(encoder,decoder,val_loder,loss_fn,device):
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for train_img,target_img in val_loder:
            train_img = train_img.to(device)
            target_img = target_img.to(device)

            enc_output = encoder(train_img)
            dec_output = decoder(enc_output)
            loss = loss_fn(dec_output, target_img)
            total_loss +=loss.item()
            num_batches +=1
    return total_loss/num_batches

