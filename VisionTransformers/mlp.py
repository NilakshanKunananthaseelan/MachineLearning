import torch 
import torch.nn as nn 


class MLP_Block(nn.Module):

  def __init__(self,input_dim,hidden_dim,dropout=0.1):
    super().__init__()

    self.fc1 = nn.Linear(input_dim,hidden_dim,bias=True)
    self.gelu1 = nn.GELU()
    self.dropou1 = nn.Dropout(dropout)

    self.fc2 = nn.Linear(hidden_dim,input_dim,bias=True)
    self.gelu2 = nn.GELU()
    self.dropou2 = nn.Dropout(dropout)

    self.init_weights()

  def init_weights(self):


    torch.nn.init.xavier_uniform_(self.fc1.weight)
    self.fc1.bias.data.fill_(0.01)


    torch.nn.init.xavier_uniform_(self.fc2.weight)
    self.fc2.bias.data.fill_(0.01)

  def forward(self,x):
    x = self.fc1(x)
    x = self.gelu1(x)
    x = self.dropou1(x)

    x = self.fc2(x)
    x = self.gelu1(x)
    x = self.dropou2(x)

    return x
