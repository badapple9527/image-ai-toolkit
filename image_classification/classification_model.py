__all__ = ["Classifier"]
import  torch.nn as nn
import torch.nn.functional as F
import torch

class Classifier(nn.Module):
    def __init__(self,num_class = 5):
        super(Classifier,self).__init__()
        self.conv1 = nn.Conv2d(3, 8, (3,3),(1,1) ,padding=(1,1))
        self.pool = nn.MaxPool2d((2,2), (2,2))
        self.conv2 = nn.Conv2d(8, 16, (3,3),(1,1) ,padding=(1,1))
        self.fc1 = nn.Linear(16*16*16,num_class)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return  F.log_softmax(x,dim=1)

if __name__ == "__main__":
    x = torch.randn(1,3,64,64)
    model = Classifier()
    output = model(x)
    print("output.shape: ", output.shape)





