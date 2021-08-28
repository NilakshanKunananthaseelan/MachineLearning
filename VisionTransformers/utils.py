
import torch
import torch.nn as nn
import torch.nn. functional as F

import PIL 
import torchvision

from einops import rearrange, repeat



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
		x = self.function(x,**kwargs)

		return x



class ResidualConnect(nn.Module):

	def __init__(self,function):
		super().__init__()
		self.function = function

	def forward(self,x,**kwargs):
		x = self.function(x,**kwargs)+x
		return x

