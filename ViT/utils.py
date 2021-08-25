
import torch
import torch.nn as nn

from einops import rearrange, repeat


from typing import *

class LayerNorm(nn.Module):
	"""A specialized dropout layer for use with Monte Carlo dropout inference.
  Can operates exactly as tf.keras.layers.Dropout during training or inference, 
  but has the option to generate and store a collection of masks for 
  reproducible Monte Carlo dropout inference (see 'fix_masks'). Requires 
  output_shape to be set as when used in a tf.keras.Model.
  
  Attributes
  ----------
  self.fixed : Bool
    Indicates whether or not the layers have been fixed
  self.masks : array_like
    A length x output_shape 
  Methods
  -------
  call
    Can be called like a standard dropout layer, or optionally with with a 
    'sample' parameter to select which of the fixed masks to apply.
  
  fix_masks
    Generates a sequence of 'length' masks and stores them in the layer for 
    future re-use. Sets 'fixed' attribute to true.
  """




	def __init__(self,normalize_dim,function):
		super().__init__()

		self.layer_norm = nn.LayerNorm(normalize_dim)
		self.function   = function

	def forward(self,x,**kwargs):
		x = self.layer_norm(x)
		x = self.fn(x,**kwargs)

		return x



class ResidualConnect(nn.Module):

	def __init__(self,function):
		super().__init__()
		self.function = function

	def forward(x,**kwargs):
		x = self.function(x,**kwargs)
		return x

#https://jalammar.github.io/illustrated-transformer/

class Attention(nn.Module):

	def __init__(self,input_dim,heads=8,dropout=0.1):
		super().__init__()

		self.heads = heads
		self.scale = torch.sqrt(input_dim)

		self.mat_calc = nn.Linear(input_dim,input_dim*3)
		self.softmax = nn.Softmax(dim=-1)

		self.fc1 = nn.Linear(input_dim,input_dim,bias=True)
		self.dropou1 = nn.Dropout(dropout)





		self.init_weights()

	def init_weights(self):

    
        torch.nn.init.xavier_uniform(self.mat_calc.weight)
        self.mat_calc.bias.data.fill_(0.01)


        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)


    def forward(self,x,mask=None):

    	b, n, _, h = *x.shape, self.heads
    	qkv = self.mat_calc(x).chunk(3,dim=-1)

    	q,k,v = rearrange(qkv,'b n (w h d) -> w b h n d',w=3,h=8)/self.scale

    	dot_product = torch.einsum('bhid,bhjd->bhij',q,k)

    	if mask is not None:

            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dot_product.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dot_product.masked_fill_(~mask, float('-inf'))
            del mask



        softmax_out = self.softmax(dot_product)

        attention_matrices = torch.einsum('bhij,bhjd->bhid',softmax,v)

        out = rearrange(out,'b h n d->b n (h d)')
        out = self.fc1(out)
        out = self.dropou1(out)
        return out



		