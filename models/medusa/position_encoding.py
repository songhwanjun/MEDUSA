""" Various positional encodings for the transformer. """

import math
from torch import nn
from util.misc import *

class PositionEmbeddingSine(nn.Module):
  """
  This is a more standard version of the position embedding, very similar to the one
  used by the Attention is all you need paper, generalized to work on images.
  """

  def __init__(self, num_pos_feats, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    # if scale is True, then eed to be normalized.
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    # defualt scaling.
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  @torch.no_grad()
  def forward(self, depths, masks):

    # this is interpolated mask: uint format
    not_mask = ~masks # inverse mask
    inverse_depths = 1.0 - depths # 0: close, 1: distant

    mask_size = not_mask.size()
    # What the mask? exclude non-used element -> cumsum -> usable element index.
    # height dim positional index
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    # width dim positional index
    x_embed = not_mask.cumsum(2, dtype=torch.float32)

    # what is the normalization? normalize index into [0,1] * 2pi
    if self.normalize:
      eps = 1e-6
      y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
      x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

    # num_pos_feats is the dimensions of each stamp.
    dim_y = torch.arange(self.num_pos_feats[0], dtype=torch.float32, device=masks.device)
    dim_x = torch.arange(self.num_pos_feats[1], dtype=torch.float32, device=masks.device)
    dim_y = self.temperature ** (2 * (torch.floor(dim_y / 2.0)) / self.num_pos_feats[0])
    dim_x = self.temperature ** (2 * (torch.floor(dim_x / 2.0)) / self.num_pos_feats[1])

    pos_y = y_embed[:, :, :, None] / dim_y # [1, 25, 34, 1] => [1, 25, 34, 128]
    pos_x = x_embed[:, :, :, None] / dim_x  # [1, 25, 34, 1] => [1, 25, 34, 128]

    # input dim postional embedding using sin and cos. Then, flattening.
    # since dim = 4, the two tensor are concatenated alternatively
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    #pos_y = pos_y.view(*mask_size, self.num_pos_feats[0])
    #pos_x = pos_x.view(*mask_size, self.num_pos_feats[1])

    if self.num_pos_feats[2] != 0:

      z_embed = inverse_depths
      # normalize [0,1] -> [0, 2pi]
      if self.normalize: # if this nomalize is "on", then the scale of depth is not important.
        eps = 1e-6
        max_value = torch.max(z_embed.view(mask_size[0], -1), dim=1)[0]
        max_value = max_value.view(mask_size[0], 1, 1)
        z_embed = z_embed / (max_value + eps) * self.scale

      dim_z = torch.arange(self.num_pos_feats[2], dtype=torch.float32, device=masks.device)
      dim_z = self.temperature ** (2 * (torch.floor(dim_z / 2.0)) / self.num_pos_feats[2])
      pos_z = z_embed[:, :, :, None] / dim_z
      pos_z = torch.stack((pos_z[:, :, :, 0::2].sin(), pos_z[:, :, :, 1::2].cos()), dim=4).flatten(3)

    # concatenation pos_y + pos_x [ 1, 25, 34, 256] => permute [1, 256, 25, 34] because torch is channel first.
    if self.num_pos_feats[2] == 0:
      pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    else:
      pos = torch.cat((pos_y, pos_x, pos_z), dim=3).permute(0, 3, 1, 2)

    return pos

def split_spatial_encoding_size(hidden_dim, h_w_ratio=1.0):

  h_w_length = math.ceil(hidden_dim * h_w_ratio)
  if h_w_length % 2 != 0:
    h_w_length -= 1

  half_hw = int(h_w_length / 2)
  if half_hw % 2 == 0:
    h_length = half_hw
    w_length = half_hw
  else:
    h_length = half_hw + 1
    w_length = half_hw - 1

  d_length = int(hidden_dim - h_w_length)

  return (h_length, w_length, d_length)

def build_position_encoding(args):

  N_steps = split_spatial_encoding_size(args.hidden_dim, h_w_ratio=1.0)
  position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

  return position_embedding
