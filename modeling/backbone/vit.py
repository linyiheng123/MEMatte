import logging
import math
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
from torch.nn import functional as F
from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.layers import DropPath, Mlp, trunc_normal_
from functools import partial
from .backbone import Backbone
from .utils import (
    PatchEmbed,
    add_decomposed_rel_pos,
    get_abs_pos,
    window_partition,
    window_unpartition,
    Router
)

logger = logging.getLogger(__name__)


__all__ = ["ViT"]


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N, _ = policy.size() # 128, 197, 1
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye # 目的是将对角线上的token计算attention
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)


    def forward(self, x, policy, H, W): # total FLOPs: 4 * B * hw * dim * dim +  2 * B * hw * hw * dim
        if x.ndim == 4:
            B, H, W, _ = x.shape
            N = H*W
        else:      
            B, N, _ = x.shape
        
        
        # qkv with shape (3, B, nHead, H * W, C) self.qkv.flops: b * hw * dim * dim * 3
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4) # 6.341787648 reshape和permute没有FLOPs
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, N, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1) # 5.637144576 (B * hw * hw * dim) 14 * 1024*1024*384

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn.reshape(B, self.num_heads, N, N), policy).reshape(B*self.num_heads, N, N)

        #  # 5.637144576 (B * hw * hw * dim)
        if x.ndim == 4:
            x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1) 
        else:
            x = (attn @ v).view(B, self.num_heads, N, -1).permute(0, 2, 1, 3).reshape(B, N, -1)

        x = self.proj(x) # 2.113929216 (B * hw * dim * dim) 14 * 1024 * 384 * 384

        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ResBottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        norm="LN",
        act_layer=nn.GELU,
        conv_kernels=3,
        conv_paddings=1,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = get_norm(norm, bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            conv_kernels,
            padding=conv_paddings,
            bias=False,
        )
        self.norm2 = get_norm(norm, bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = get_norm(norm, out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)

        out = x + out
        return out

class ECA(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=self.kernel_size(),
            padding=(self.kernel_size() - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs((math.log2(self.channels) / self.gamma) + self.b / self.gamma))
        out = k if k % 2 else k + 1
        return out

    def forward(self, x):

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)



class LTRM(nn.Module):
    """Efficient Block to replace the Original Transformer Block"""
    def __init__(self, dim, expand_ratio = 2, kernel_size = 5):
        super(LTRM, self).__init__()
        self.fc1 = nn.Linear(dim, dim*expand_ratio)
        self.act1 = nn.GELU()
        self.dwconv = nn.Conv2d(dim*expand_ratio, dim*expand_ratio, kernel_size=(kernel_size, kernel_size), groups=dim*expand_ratio, padding=(kernel_size//2, kernel_size//2))
        self.act2 = nn.GELU()
        self.fc2 = nn.Linear(dim*expand_ratio, dim)
        self.eca = ECA(dim)
    
    def forward(self, x, prev_msa=None):
        x = self.act1(self.fc1(x))
        x = self.act2(self.dwconv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        x =  self.fc2(x)
        y = x + self.eca(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return y





class EfficientBlock_ALR(nn.ModuleList): # from rething attention AAAI
    def __init__(self, model_dimension=128):
        super(EfficientBlock_ALR, self).__init__()
        # self.sentence_length=sentence_length
        self.model_dimension = model_dimension
        self.width = self.model_dimension
        self.layers=list()
        widths=[1,2,1]
        self.depth=len(widths)-1
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            self.layers.extend([nn.LayerNorm(self.width * widths[i]),nn.Linear(self.width * widths[i], self.width * widths[i+1])])
            if(i<self.depth-1):
                # self.layers.append(nn.LeakyReLU()) # 原论文用这个
                self.layers.append(nn.GELU())

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        use_cc_attn = False,
        use_residual_block=False,
        use_convnext_block=False,
        input_size=None,
        res_conv_kernel_size=3,
        res_conv_padding=1,
        use_efficient_block = False,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.window_size = window_size
        self.use_efficient_block = use_efficient_block
        if use_efficient_block:
            self.efficient_block = LTRM(dim = dim)

        self.use_residual_block = use_residual_block
        if use_residual_block:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm="LN",
                act_layer=act_layer,
                conv_kernels=res_conv_kernel_size,
                conv_paddings=res_conv_padding,
            )
        self.use_convnext_block = use_convnext_block
        if use_convnext_block:
            self.convnext = ConvNextBlock(dim = dim)

        if use_cc_attn:
            self.attn = CrissCrossAttention(dim)

    def msa_forward(self, x, policy, H, W):
        # shortcut = x
        x = self.norm1(x) # 0.27525
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

            x = self.attn(x, policy, H = H, W = W)
            
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        else:
            x = self.attn(x, policy, H = H, W = W)


        x =  self.drop_path(x)

        return x

    def mlp_forward(self, x): 
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, policy=None):
        B, H, W, C = x.shape
        N = H * W
        # B, N, C = x.shape
        shortcut = x
        
        if self.use_efficient_block:
            if self.training:
                fast_msa = self.efficient_block(x)
                slow_msa = self.msa_forward(x, policy, H, W).reshape(B, N, C)
                # slow_msa = slow_msa * (policy + (1. - policy).detach())
                # msa = torch.where(policy.bool(), slow_msa, fast_msa.reshape(B, N, C)).reshape(B, H, W, C)
                msa = (slow_msa * policy + fast_msa.reshape(B, -1, C) * (1. - policy)).reshape(B, H, W, C)
            else:
                msa = self.efficient_block(x)
                selected_indices = policy.squeeze(-1).bool()
                # if True in selected_indices:
                if torch.any(selected_indices == 1):
                    selected_x = x.reshape(B, -1, C)[selected_indices].unsqueeze(0)
                    slow_msa = self.msa_forward(selected_x, policy=None, H = H, W = W)
                    msa.masked_scatter_(selected_indices.reshape(B, H, W, 1), slow_msa)
        else:
            msa = self.msa_forward(x, policy, H, W).reshape(B, H, W, C)
        x = shortcut + msa
        x = x + self.mlp_forward(x)

        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if self.use_convnext_block:
            x = self.convnext(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x


class ViT(Backbone):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        window_block_indexes=(),
        residual_block_indexes=(),
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_feature="last_feat",
        res_conv_kernel_size=3, 
        res_conv_padding=1,
        topk = 1.,
        multi_score = False,
        router_module = "Ours",
        skip_all_block = False,
        max_number_token = 18500,
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            residual_block_indexes (list): Indexes for blocks using conv propagation.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            out_feature (str): name of the feature from the last block.
        """
        super().__init__()

        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.topk = topk
        self.multi_score = multi_score
        self.skip_all_block = skip_all_block
        self.window_block_indexes = window_block_indexes
        self.max_number_token = max_number_token
        self.router_module = router_module
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=True if (i in window_block_indexes and use_rel_pos == True) else False,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size // patch_size, img_size // patch_size),
                res_conv_kernel_size=res_conv_kernel_size,
                res_conv_padding=res_conv_padding,
                use_efficient_block=False if (i == 0 or i in window_block_indexes)  else True,
            )
            if use_act_checkpoint:
                block = checkpoint_wrapper(block)
            self.blocks.append(block)
        
        self.routers = nn.ModuleList()
        for i in range(depth - len(window_block_indexes)):
            router = Router(embed_dim = embed_dim)
            self.routers.append(router)


        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )

        B, H, W, C = x.shape
        N = H * W
        prev_decision = torch.ones(B, N, 1, dtype = x.dtype, device = x.device)
        out_pred_prob = []
        out_hard_keep_decision = []
        count = 0
        
        for i, blk in enumerate(self.blocks): # x.shape = torch.Size([16, 32, 32, 384])
            if i in self.window_block_indexes or i == 0:
                x = blk(x, policy = None)
                continue
            pred_score = self.routers[count](x.reshape(B, -1, C), prev_decision).reshape(B, -1, 2) # B, N, 2
            count = count + 1
            if self.training:
                hard_keep_decision = F.gumbel_softmax(pred_score, hard = True)[:, :, 0:1]
                out_pred_prob.append(hard_keep_decision.reshape(B, H*W))
                x = blk(x, policy = hard_keep_decision)
            else:
                """gumbel_softmax"""
                # hard_keep_decision = F.gumbel_softmax(pred_score, hard = True)[:, :, 0:1] # torch.Size([1, N, 1])
                
                """argmax"""
                hard_keep_decision = torch.zeros_like(pred_score[..., :1])
                hard_keep_decision[pred_score[..., 0] > pred_score[..., 1]] = 1.

                if hard_keep_decision.sum().item() > self.max_number_token:
                    print("=================================================================")
                    print(f"Decrease the number of tokens in Global attention: {self.max_number_token}")
                    print("=================================================================")
                    _, sort_index = torch.sort((pred_score[..., 0] - pred_score[..., 1]), descending=True)
                    sort_index = sort_index[:, :self.max_number_token]
                    hard_keep_decision = torch.zeros_like(pred_score[..., :1])
                    hard_keep_decision[:, sort_index.squeeze(0), :] = 1.

                # threshold = torch.quantile(pred_score[..., :1], 0.85)
                # hard_keep_decision = (pred_score[..., :1] >= threshold).float()
                
                # out_pred_prob.append(pred_score[..., 0].reshape(B, x.shape[1], x.shape[2]))

                out_pred_prob.append(pred_score.reshape(B, x.shape[1], x.shape[2], 2))
                out_hard_keep_decision.append(hard_keep_decision.reshape(B, x.shape[1], x.shape[2], 1))
                x = blk(x, policy = hard_keep_decision)

        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}

        if self.training:
            return outputs['last_feat'], out_pred_prob
        else:
            return outputs['last_feat'], out_pred_prob, out_hard_keep_decision


if __name__ == '__main__':
    model = ViT(
        in_chans=4,
        img_size=512,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        drop_path_rate=0,
        window_size=0,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            # 2, 5, 8 11 for global attention
            # 0,
            # 1,
            # 3,
            # 4,
            # 6,
            # 7,
            # 9,
            # 10,
        ],
        residual_block_indexes=[2, 5, 8, 11],
        use_rel_pos=True,
        out_feature="last_feat"
    )
    print(model)

    out, prob = model(torch.ones(2, 4, 512, 512)) # sum([p.sum() / p.numel() for p in prob]) / len(prob)
    print(out.shape)