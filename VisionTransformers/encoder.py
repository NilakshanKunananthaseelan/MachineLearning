
#https://jalammar.github.io/illustrated-transformer/

import torch
import torch.nn as nn
import torch.nn. functional as F

import PIL 
import torchvision

from einops import rearrange, repeat

from mlp import MLP_BLOCK
from utils import LayerNorm,ResidualConnect


class Attention(nn.Module):

  def __init__(self,input_dim,heads=8,dropout=0.1):
    super().__init__()

    self.heads = heads
    self.scale = input_dim**0.5

    self.mat_calc = nn.Linear(input_dim,input_dim*3)
    self.softmax = nn.Softmax(dim=-1)

    self.fc1 = nn.Linear(input_dim,input_dim,bias=True)
    self.dropou1 = nn.Dropout(dropout)


    self.init_weights()

  def init_weights(self):


    torch.nn.init.xavier_uniform_(self.mat_calc.weight)
    self.mat_calc.bias.data.fill_(0.01)


    torch.nn.init.xavier_uniform_(self.fc1.weight)
    self.fc1.bias.data.fill_(0.01)


  def forward(self,x,mask=None):

    b, n, _= x.shape
    h = self.heads
    qkv = self.mat_calc(x)

    q,k,v = rearrange(qkv,'b n (w h d) -> w b h n d',w=3,h=8)/self.scale

    dot_product = torch.einsum('bhid,bhjd->bhij',q,k)

    if mask is not None:

      mask = F.pad(mask.flatten(1), (1, 0), value = True)
      assert mask.shape[-1] == dot_product.shape[-1], 'mask has incorrect dimensions'
      mask = mask[:, None, :] * mask[:, :, None]
      dot_product.masked_fill_(~mask, float('-inf'))
      del mask



    softmax_out = self.softmax(dot_product)

    attention_matrices = torch.einsum('bhij,bhjd->bhid',softmax_out,v)

    out = rearrange(attention_matrices,'b h n d->b n (h d)')
    out = self.fc1(out)
    out = self.dropou1(out)
    return out

class EncoderLayers(nn.Module):
  def __init__(self,input_dim,heads,mlp_dim,dropout=0.1):
    super().__init__()

    self.attention_block = ResidualConnect(LayerNorm(input_dim,Attention(input_dim,heads=heads,dropout=dropout))) 
    self.mlp_block = ResidualConnect(LayerNorm(input_dim,MLP_Block(input_dim,mlp_dim,dropout=dropout)))


  # def forward(self,x,mask=None):
  #   x = self.attention_block(x,mask=mask)
  #   x = self.mlp_block(x)

  #   return x


class TransformerEncoder(nn.Module):

  def __init__(self,input_dim,depth,heads,mlp_dim,dropout=0.1):
    super().__init__()

    self.encoder_layers = nn.ModuleList([])

    for _ in range(depth):
      self.encoder_layers.append(nn.ModuleList([ResidualConnect(LayerNorm(input_dim,Attention(input_dim,heads=heads,dropout=dropout))),
      ResidualConnect(LayerNorm(input_dim,MLP_Block(input_dim,mlp_dim,dropout=dropout)))]))

  def forward(self,x,mask=None):
      for attention_block,mlp_block in self.encoder_layers:

        x = attention_block(x,mask=mask)
        x = mlp_block(x)
      return x