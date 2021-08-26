import torch
import torch.nn as nn

from einops import rearrange, repeat
from encoder import TransformerEncoder

class ViT(nn.Module):
	def __init__(self,*,image_size,patch_size,num_classes,input_dim,depth,heads,mlp_dim,channels=3,dropout=0.1,embedding_dropout=0.1):
		super.__init__()


		assert not(image_size%patch_size),'image dimension should be factor of patch dimension'

		num_patches = (image_size//patches_size)**2

		patches_dim = channels*patch_size**2

		self.patch_size = patch_size
		self.pos_embeds =  nn.Parameter(torch.empty(1, (num_patches + 1), dim))

		self.patch_conv = nn.Conv2d(3,input_dim,kernel_size=patch_size,stride=patch_size)#change input_dim name


		self.cls_embeds = nn.Parameter(torch.zeros(1,1,dim))
		self.dropout = nn.Dropout(embedding_dropout)

		self.transformer_encoder = TransformerEncoder(input_dim,depth,heads,mlp_dim,dropout)

		self.to_cls_embeds =  nn.Identity()

		self.mlp_head = nn.Linear(input_dim,num_classes)

		self.init_weights()

	def init_weights(self):

		torch.nn.init.normal_(self.pos_embeds, std = .02) # initialized based on the paper

		torch.nn.init.xavier_uniform_(self.nn1.weight)
		torch.nn.init.normal_(self.nn1.bias, std = 1e-6)


	def forward(self,img,mask=None):


		img_patches = self.patch_conv(img)

		x = rearrange(img_patches,'b c h w -> b (h w) c')
		cls_embeds = self.cls_embeds.expand(img.shape[0],-1,-1)

		x = torch.cat((cls_embeds,x),dim=1)
		x+= self.pos_embeds
		x = self.dropout(x)
		x = self.transformer_encoder(x,mask)
		x = self.to_cls_embeds(x[:,0])

		x = self.mlp_head(x)

		return x





