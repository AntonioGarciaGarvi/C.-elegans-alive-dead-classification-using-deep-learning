import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch


class Resnet18Pretrained(nn.Module):
    def __init__(self, **kwargs):
        super(Resnet18Pretrained, self).__init__()
        self.seq_dim = kwargs.get("seq_dim")
        self.conv_model = models.resnet18(pretrained=True) 

        for param in self.conv_model.parameters():
            param.requires_grad = True
        self.conv_model.fc = nn.Identity()

    def forward(self, x):
        return self.conv_model(x)



class CombineResnet18Lstm(nn.Module):
    def __init__(self, **kwargs):
        super(CombineResnet18Lstm, self).__init__()
        self.seq_lenght = int(kwargs.get("seq_lenght"))
        self.seq_dim = 512 # features last resnet layer
        self.cnn = Resnet18Pretrained(seq_dim=self.seq_dim)
        self.rnn = nn.LSTM(
            input_size=self.seq_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True)
        self.linear = nn.Linear(256, 2)

    def forward(self, x):
        batch_size, timesteps, H, W = x.shape 
        c_in = x.view(batch_size*3,3, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, self.seq_lenght, -1)
        r_out, (h_n, h_c) = self.rnn(r_in)
        r_out2 = self.linear(r_out[:, -1, :]) # last hidden state

        return F.log_softmax(r_out2, dim=1)
