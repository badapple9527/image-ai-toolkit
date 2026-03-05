__all__ = ["train_step","val_step"]

import torch

def  train_step(classifier,train_loder,loss_fn,optimizer,device):
    total_loss = 0
    num_batches = 0
    for train_img,classification in train_loder:
        train_img = train_img.to(device)
        classification = classification.to(device)
        optimizer.zero_grad()
        output = classifier(train_img)
        loss = loss_fn(output, classification)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches

def val_step(classifier,val_loder,loss_fn,device):
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for train_img,classification in val_loder:
            train_img = train_img.to(device)
            classification = classification.to(device)

            output = classifier(train_img)
            loss = loss_fn(output,classification)
            total_loss +=loss.item()
            num_batches +=1
    return total_loss/num_batches



