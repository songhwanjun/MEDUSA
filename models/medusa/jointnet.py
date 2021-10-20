""" Siamese Backbone for feature extractor """
from torch import nn
import torch.nn.functional as F
from util.misc import *

from .position_encoding import build_position_encoding
from .cbam import CBAM

# down & up sampling mode
downsampling_mode = 'nearest'
upsampling_mode = 'bilinear'

# resnet structure
resnet_out_chnnels = {
    'resnet18': {
        '0': 64,
        '1': 64,
        '2': 128,
        '3': 256,
        '4': 512
    },
    'resnet34': {
        '0': 64,
        '1': 64,
        '2': 128,
        '3': 256,
        '4': 512
    },
    'resnet50': {
        '0': 64,
        '1': 256,
        '2': 512,
        '3': 1024,
        '4': 2048
    }
}

class FrozenBatchNorm2d(torch.nn.Module):
    """BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from the DETR implementation, https://github.como/facebookresearch/
    detr/blob/master/models/backbone.py
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys, unexpected_keys,
                                          error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


def resnet_backbone(name,
                    pretrained=True,
                    norm_layer=None):
    """A pretrained ResNet loader.

    Args:
        name: ResNet model name in {resnet18, resnet34, resnet50}.
        pretrained: Whether to use the pretrained model or not.
        norm_layer: Custom norm_layer to apply for the backbone

    Returns:
        A ResNet backbone.

    Raises:
        ValueError: If the given resnet model is not supported.
    """

    if name not in ['resnet18', 'resnet34', 'resnet50']:
        raise ValueError(f'the model {name} not supported')

    return getattr(torchvision.models, name)(
        pretrained=pretrained, norm_layer=norm_layer)


class JointNet(nn.Module):
    """ A siamese network to extract RGB and Depth features """

    def __init__(self, name, out_dim, refiner=True, num_b_mask=20):
        """ Initialization

        Parameters:
          name: resnet name
          out_dim: out channel dim
          refiner: whether to use refiner
          num_b_mask: the number of binary masks
        """

        super(JointNet, self).__init__()

        self.resnet = resnet_backbone(name, pretrained=True, norm_layer=FrozenBatchNorm2d)
        self.out_dim = out_dim
        self.refiner = refiner
        self.num_b_mask = num_b_mask

        if self.refiner:
            # for extracting the global saliency map
            self.saliency_conv = torch.nn.Conv2d(resnet_out_chnnels[name]['4'], 1, kernel_size=1)
            torch.nn.init.xavier_uniform_(self.saliency_conv.weight, gain=1)
            torch.nn.init.constant_(self.saliency_conv.bias, 0)

            # for pixel refinement (spatial attention)
            self.refine_conv = torch.nn.Conv2d(resnet_out_chnnels[name]['4'], 1, kernel_size=1)
            torch.nn.init.xavier_uniform_(self.refine_conv.weight, gain=1)
            torch.nn.init.constant_(self.refine_conv.bias, 0)

        dim_scale = 1
        if self.refiner:
          # for channel attention
          self.rgb_cbam = CBAM(resnet_out_chnnels[name]['4'], no_spatial=True)
          self.depth_cbam = CBAM(resnet_out_chnnels[name]['4'], no_spatial=True)
          dim_scale *= 2

        self.rgb_input_proj = nn.Conv2d(resnet_out_chnnels[name]['4'] * dim_scale,
                                        self.out_dim, kernel_size=1)
        self.depth_input_proj = nn.Conv2d(resnet_out_chnnels[name]['4'] * dim_scale,
                                          self.out_dim, kernel_size=1)

        self.num_channels = self.out_dim

    def forward(self, tensor_list: NestedTensor):
        """ The forward step of Siamese Network with Feature Refiner

            Parameters:
              tensor_list: the RGB features with padding masks

            Returns:
              rgb_features: the (refined) rgb feature maps
              depth_feature: the (refined) depth feature maps
              depth_maps: the orignal depth maps
              masks: the padding masks
        """

        rgbs = tensor_list.rgb
        depths = tensor_list.depth
        depth_maps = tensor_list.depth_map
        masks = tensor_list.mask

        cur_batch_size = rgbs.size()[0]
        depth_maps = depth_maps.unsqueeze(1)
        masks = masks.unsqueeze(1)
        rgb_features = rgbs
        depth_features = depths

        # input for the siamese resnet
        featuers = torch.cat([rgb_features, depth_features], dim=0)

        # feature extraction
        featuers = self.feature_extractor(self.resnet, featuers)

        # decomposition
        rgb_features = featuers[:cur_batch_size, :, :, :]
        depth_features = featuers[cur_batch_size:, :, :, :]

        # feature refiner
        if self.refiner:
            saliencies, r_depth_maps, spatial_attention = self.spatial_attention(depth_maps, rgb_features, self.num_b_mask)
            r_depth_maps = F.interpolate(r_depth_maps, size=featuers.size()[2:], mode=downsampling_mode)
            spatial_attention = F.interpolate(spatial_attention, size=featuers.size()[2:], mode=downsampling_mode)
            _rgb_features = self.rgb_cbam(rgb_features * spatial_attention)
            _depth_features = self.depth_cbam(depth_features * spatial_attention)
            rgb_features = torch.cat([rgb_features, _rgb_features], dim=1)
            depth_features = torch.cat([depth_features, _depth_features], dim=1)

        ## depth and mask downsampling
        depth_maps = F.interpolate(depth_maps, size=featuers.size()[2:], mode=downsampling_mode)
        masks = F.interpolate(masks.float(), size=featuers.size()[2:], mode=downsampling_mode)
        depth_maps = depth_maps.squeeze(1) # used for postional encoding
        masks = masks.squeeze(1).to(torch.bool) # used for postional encoding

        # projection
        rgb_features = self.rgb_input_proj(rgb_features)
        depth_feature = self.depth_input_proj(depth_features)

        return rgb_features, depth_feature, depth_maps, masks

    def feature_extractor(self, model, x):
        """ ResNet feature extraction module """

        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        return x

    def spatial_attention(self, depth_map, rgb_features, num_b_mask=10):
        """ Spatial attention with pixel re-weighting

            Parameters:
              depth_map: the depth map
              rgb_features: the rgb features
              num_b_mask: the number of binary masks
        """

        # obtain the global saliency map from the rgb feature map
        saliency = torch.nn.functional.relu(self.saliency_conv(rgb_features))
        saliency = F.interpolate(saliency, size=depth_map.size()[2:], mode=upsampling_mode)

        # compute pixel re-weighting map
        refine_weight = torch.nn.functional.relu(self.refine_conv(rgb_features))
        refine_weight = F.interpolate(refine_weight, size=depth_map.size()[2:], mode=upsampling_mode)

        # depth map refinement
        depth_map = depth_map * refine_weight
        max = torch.max(depth_map.reshape(depth_map.size(0), -1), dim=1)[0]
        max = max.reshape(max.size(0), 1, 1, 1)
        depth_map = depth_map / max

        # compute spatial attention map using the refined depth map
        interval = 1.0 / num_b_mask
        b_masks = []
        for num_b in range(num_b_mask):
            if num_b == 0:
                b_mask = depth_map <= ((num_b+1) * interval)
            else:
                b_mask = (depth_map > ((num_b) * interval)) * (depth_map <= ((num_b+1) * interval))
            b_mask = b_mask.to(torch.float32)
            b_masks.append(b_mask)

        attention_map = None
        for b_mask in b_masks:
            sum = torch.sum(b_mask, dim=(2,3), dtype=torch.float32) #
            weight = torch.sum(b_mask * saliency, dim=(2,3), dtype=torch.float32) / (sum + 1e-5)
            weight = weight.reshape(saliency.size()[0], 1, 1, 1)

            if attention_map is None:
                attention_map = b_mask * weight
            else:
                attention_map += b_mask * weight

        return saliency, depth_map, attention_map


class Joiner(nn.Sequential):
    """ Joiner 'Siamese Net' + 'Pos Encoder' """

    def __init__(self, feature_extractor, position_embedding):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.position_embedding = position_embedding

    def forward(self, tensor_list):
        """ The forward step

          Parameters:
            tensor_list: input (RGBs and padding masks)

          Returns:
            out: (refined) feature maps after attention
            pos: pos encodings
        """

        out: List[torch.Tensor] = []
        pos = []

        # the output of backbone
        rgb_features, depth_features, depth_maps, masks = self.feature_extractor(tensor_list)
        out.append((rgb_features, depth_features, masks))

        # position encoding
        pos.append(self[1](depth_maps, masks).to(rgb_features.dtype))

        # [(rgb, depth, rgbd)], [pos]
        return out, pos


def build_backbone(args):
    """ Build a backbone """

    backbone_name = args.backbone
    refiner = args.refiner
    num_b_mask = args.num_b_mask
    out_dim = args.hidden_dim

    feature_extractor = JointNet(name=backbone_name, out_dim=out_dim, refiner=refiner, num_b_mask=num_b_mask)
    position_embedding = build_position_encoding(args)

    model = Joiner(feature_extractor, position_embedding)
    model.num_channels = feature_extractor.num_channels

    return model


