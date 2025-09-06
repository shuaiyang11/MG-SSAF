# --------------------------------------------------------
# MG-SSAF
# Author: ShuaiYang
# Date: 20230322
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Mlp(nn.Module):
    def __init__(self, dim, mlp_hidden_dim, act_layer=nn.GELU, drop=0.):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, mlp_hidden_dim * dim, kernel_size=1, stride=1)
        self.act = act_layer()
        self.pwconv2 = nn.Conv2d(mlp_hidden_dim * dim, dim, kernel_size=1, stride=1)
        self.drop = nn.Dropout(drop)
        self.branch_dw = nn.Conv2d(mlp_hidden_dim * dim,
                                   mlp_hidden_dim * dim,
                                   kernel_size=7,
                                   stride=1,
                                   padding=3,
                                   groups=mlp_hidden_dim * dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.branch_dw(x)
        x = self.pwconv2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, input_resolution, ratio, width, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.ratio = ratio
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.act = nn.GELU()

        self.norm = LayerNorm(dim)
        self.Window = 7
        self.proj_qkv = nn.Conv2d(dim, dim * 2, 1, 1, bias=qkv_bias)

        local_heads = num_heads if self.ratio == 0 else int(self.num_heads * self.ratio)
        self.attn_local_win = nn.Sequential(nn.Conv2d(local_heads,
                                                      local_heads,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1,
                                                      groups=local_heads),
                                            nn.GELU(),
                                            nn.Conv2d(local_heads, local_heads, kernel_size=1, stride=1))

        if self.ratio != 0:
            global_heads = int(self.num_heads * self.ratio) // 3
            self.attn_global_win = nn.Sequential(nn.Conv2d(global_heads,
                                                           global_heads,
                                                           kernel_size=3,
                                                           stride=1,
                                                           padding=1,
                                                           groups=global_heads),
                                                 nn.GELU(),
                                                 nn.Conv2d(global_heads, global_heads, kernel_size=1, stride=1))

        self.proj_concat = nn.Conv2d(dim, dim, 1, 1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.WHV_use = True
        self.g_Window = [7, 7]
        self.g_Window_hor = [width, input_resolution[1]]
        self.g_Window_ver = [input_resolution[0], width]

        self.avgpool_win = nn.AvgPool2d(kernel_size=(7, 7), stride=7)
        self.avgpool_hor = nn.AvgPool2d(kernel_size=(width, input_resolution[1]), stride=width)
        self.avgpool_ver = nn.AvgPool2d(kernel_size=(input_resolution[0], width), stride=width)

    def forward(self, x):

        B, C, H, W = x.shape
        qkv = self.proj_qkv(x)

        if not self.ratio:
            output = self.Win_Attention(qkv, self.Window, self.num_heads)
        else:
            l_num_heads = int(self.num_heads * self.ratio)
            g_num_heads = int(self.num_heads * (1 - self.ratio))

            l_dim = int(2 * C * self.ratio)
            l_feats = qkv[:, 0:l_dim, :, :]
            g_feats = qkv[:, l_dim:, :, :]

            l_feats = self.Win_Attention(l_feats, self.Window, l_num_heads)
            g_feats = self.WHV_GlobalAttention(g_num_heads, g_feats)
            output = torch.cat((l_feats, g_feats), dim=1)

        output = self.act(output)
        output = self.proj_concat(output)
        output = self.proj_drop(output)

        return output

    def window_partition_H(self, x, window_size, num_heads):

        B, C, H, W = x.shape
        x = x.view(B, num_heads, C // num_heads, H // window_size, window_size, W // window_size, window_size)
        windows = x.permute(0, 3, 5, 1, 4, 6, 2).contiguous().view(-1, num_heads, window_size, window_size, C // num_heads)
        return windows

    def window_partition_W(self, x, window_size, num_heads):

        B, C, H, W = x.shape
        x = x.view(B, num_heads, C // num_heads, H // window_size, window_size, W // window_size, window_size)
        windows = x.permute(0, 3, 5, 1, 6, 4, 2).contiguous().view(-1, num_heads, window_size, window_size, C // num_heads)
        return windows

    def window_partition_v(self, x, window_size, num_heads):

        B, C, H, W = x.shape
        x = x.view(B, num_heads, C // num_heads, H // window_size, window_size, W // window_size, window_size)
        windows = x.permute(0, 3, 5, 1, 4, 6, 2).contiguous().view(-1, num_heads, window_size * window_size, C // num_heads)
        return windows

    def window_reverse(self, windows, window_size, H, W, num_heads):

        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, num_heads, window_size, window_size, -1)
        x = x.permute(0, 3, 6, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
        return x

    def WinBased_SSMSA(self, q, k, v, num_heads, window_size):
        B, C, H, W = q.shape

        q_H = self.window_partition_H(q, window_size, num_heads)
        k_H = self.window_partition_H(k, window_size, num_heads)
        q_H = q_H * self.scale
        atten_H = (q_H @ k_H.transpose(-2, -1))
        atten_H = self.softmax(atten_H)

        q_W = self.window_partition_W(q, window_size, num_heads)
        k_W = self.window_partition_W(k, window_size, num_heads)
        q_W = q_W * self.scale
        atten_W = (q_W @ k_W.transpose(-2, -1))
        atten_W = self.softmax(atten_W)

        atten_H = atten_H.unsqueeze(-1).expand(-1, num_heads, window_size, window_size, window_size, window_size)
        atten_W = atten_W.permute(0, 1, 3, 2, 4).unsqueeze(-3).expand(-1, num_heads, window_size, window_size,
                                                                      window_size, window_size)
        atten = (atten_H * atten_W)

        atten_or = atten.permute(0, 1, 2, 3, 5, 4).reshape(-1, num_heads, window_size * window_size,
                                                           window_size * window_size)

        atten_fusion = atten.permute(0, 2, 3, 1, 5, 4).reshape(-1, num_heads, window_size, window_size)
        atten_fusion = self.attn_local_win(atten_fusion)
        atten_fusion = atten_fusion.view(-1, window_size, window_size, num_heads, window_size,
                                         window_size).permute(0, 3, 1, 2, 4, 5)
        atten_fusion = atten_fusion.reshape(-1, num_heads, window_size * window_size, window_size * window_size)

        atten = self.attn_drop(atten_or + atten_fusion)
        v = self.window_partition_v(v, window_size, num_heads)

        atten_v = atten @ v
        atten_v = self.window_reverse(atten_v, window_size, H, W, num_heads)

        return atten_v

    def Win_Attention(self, qkv, window_size, num_heads):
        B, _, H, W = qkv.shape

        qkv = qkv.view(B, 2, -1, H, W).permute(1, 0, 2, 3, 4)
        q, k = qkv[0], qkv[1]

        Win_atten = self.WinBased_SSMSA(q, k, k, num_heads, window_size)

        return Win_atten

    def GlobalPool_MSA(self, qkv_G, num_heads, AvgPooling):

        B, C, H, W = qkv_G.shape
        C_ = C // 2

        q_G, k_G = qkv_G.chunk(2, dim=1)
        k_GA = AvgPooling(k_G)
        q_G = q_G.permute(0, 2, 3, 1).reshape(B, -1, num_heads, C_//num_heads).permute(0, 2, 1, 3)
        k_GA = k_GA.permute(0, 2, 3, 1).reshape(B, -1, num_heads, C_//num_heads).permute(0, 2, 1, 3)

        q_G = q_G * self.scale
        atten = (q_G @ k_GA.transpose(-2, -1))
        atten = self.softmax(atten)
        atten = self.attn_drop(atten)

        GloablFeats = atten @ k_GA
        GloablFeats = GloablFeats.permute(0, 1, 3, 2).view(B, num_heads, C_//num_heads, H, W,)
        GloablFeats = GloablFeats.reshape(B, C_, H, W)
        return GloablFeats

    def GlobalPool_SSMSA(self, qkv_G, num_heads, AvgPooling):

        B, C, H, W = qkv_G.shape
        C_ = C // 2

        q_G, k_G = qkv_G.chunk(2, dim=1)
        k_G = AvgPooling(k_G)
        _, _, k_height, _ = k_G.size()
        v_G = k_G.view(B, num_heads, C_ // num_heads, k_height, k_height).permute(0, 1, 3, 4, 2).reshape(B, num_heads, k_height * k_height, C_ //  num_heads)

        q_G = q_G.view(B, num_heads, C_ // num_heads, H * W).permute(0, 1, 3, 2)
        q_G = q_G.unsqueeze(-3).expand(B, num_heads, k_height, H * W, C_ // num_heads)

        k_H = k_G.view(B, num_heads, C_ // num_heads, k_height, k_height).permute(0, 1, 3, 4, 2)
        q_H = q_G * self.scale

        atten_H = (q_H @ k_H.transpose(-2, -1))
        atten_H = self.softmax(atten_H)

        k_W = k_G.view(B, num_heads, C_ // num_heads, k_height, k_height).permute(0, 1, 4, 3, 2)
        q_W = q_G * self.scale

        atten_W = (q_W @ k_W.transpose(-2, -1))
        atten_W = self.softmax(atten_W)

        atten_H = atten_H.permute(0, 1, 3, 4, 2)
        atten_W = atten_W.permute(0, 1, 3, 2, 4)

        atten = (atten_H * atten_W)

        atten_or = atten.permute(0, 1, 2, 4, 3).reshape(B, num_heads, H * W, k_height * k_height)

        atten_fusion = atten.permute(0, 2, 1, 4, 3).reshape(-1, num_heads, k_height, k_height)
        atten_fusion = self.attn_global_win(atten_fusion)
        atten_fusion = atten_fusion.view(-1, H * W, num_heads, k_height, k_height).permute(0, 2, 1, 3, 4)
        atten_fusion = atten_fusion.reshape(-1, num_heads, H * W, k_height * k_height)

        atten = self.attn_drop(atten_or + atten_fusion)

        GloablFeats = atten @ v_G
        GloablFeats = GloablFeats.permute(0, 1, 3, 2).view(B, num_heads, C_ // num_heads, H, W,)
        GloablFeats = GloablFeats.reshape(B, C_, H, W)

        return GloablFeats

    def WHV_GlobalAttention(self, num_heads, qkv):
        if self.WHV_use:
            l_win_heads = num_heads // 3
            l_hor_heads = num_heads // 3
            l_ver_heads = num_heads // 3

            win_feats, hor_feats, ver_feats = qkv.chunk(3, dim=1)

            win_feats = self.GlobalPool_SSMSA(win_feats, l_win_heads, self.avgpool_win)
            hor_feats = self.GlobalPool_MSA(hor_feats, l_hor_heads, self.avgpool_hor)
            ver_feats = self.GlobalPool_MSA(ver_feats, l_ver_heads, self.avgpool_ver)
            output = torch.cat((win_feats, hor_feats, ver_feats), dim=1)
        else:
            output = self.GlobalPool_SSMSA(qkv, num_heads, self.avgpool_win)

        return output



class MGSSAFBlock(nn.Module):
    def __init__(self, dim,
                 input_resolution,
                 num_heads,
                 ratio,
                 width,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.ratio = ratio
        self.width = width
        self.mlp_ratio = mlp_ratio

        self.norm1 = LayerNorm(dim)
        self.attn = Attention(self.dim,
                              input_resolution,
                              ratio,
                              width,
                              num_heads,
                              qkv_bias=qkv_bias,
                              attn_drop=attn_drop,
                              proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(dim=dim, mlp_hidden_dim=self.mlp_ratio, act_layer=act_layer, drop=drop)

    def forward(self, x):

        attn_shortcut = x
        x = self.norm1(x)
        attn_x = self.attn(x)
        x = attn_shortcut + self.drop_path(attn_x)

        x = x + self.drop_path(self.mlp(x))

        return x



class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.act = nn.GELU()
        self.norm = LayerNorm(dim, eps=1e-6)

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim)
        self.pwconv1 = nn.Conv2d(dim, 2 * dim, kernel_size=(1, 1), stride=(1, 1))
        self.pwconv2 = nn.Conv2d(2 * dim, dim * 2, kernel_size=(1, 1), stride=(1, 1))

        self.AveragePool = nn.AvgPool2d(2, stride=2)
        self.Fuse1x1 = nn.Conv2d(dim, dim * 2, kernel_size=(1, 1), stride=(1, 1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        downsample = x

        x = self.norm(x)
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        downsample = self.AveragePool(downsample)
        downsample = self.norm(downsample)
        downsample = self.Fuse1x1(downsample)

        out = x + downsample

        return out



class BasicLayer(nn.Module):
    def __init__(self, dim,
                 input_resolution,
                 depth,
                 num_heads,
                 ratio,
                 width,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            MGSSAFBlock(dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        ratio=ratio,
                        width=width,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x



class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.act = nn.GELU()
        self.interDim = int(embed_dim / 2)
        self.norm = LayerNorm(embed_dim, eps=1e-6)

        self.Conv2Down = nn.Conv2d(in_chans, self.interDim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.Conv4Down = nn.Conv2d(self.interDim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.ConvFuse = nn.Conv2d(embed_dim, embed_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [56, 56]
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = self.Conv2Down(x)
        out = self.act(out)

        out = self.Conv4Down(out)
        out = self.norm(out)
        out = self.act(out)

        out = self.ConvFuse(out)

        return out

class MGSSAF(nn.Module):
    def __init__(self, img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=[2, 2, 18, 2],
                 num_heads=[3, 6, 12, 24],
                 ratio=[0.0, 0.5, 0.5, 0.0],
                 width=[0, 2, 2, 0],
                 mlp_ratio=2,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 patch_norm=True,
                 use_checkpoint=False):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=4, in_chans=in_chans, embed_dim=embed_dim)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               ratio=ratio[i_layer],
                               width=width[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.last_conv = nn.Conv2d(self.num_features, 1280, kernel_size=1, stride=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(drop_rate)

        self.head = nn.Linear(1280, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x)
        x = self.last_conv(x)
        x = self.act(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


if __name__ == "__main__":

    model = MGSSAF()
    model.eval()
    input = torch.randn(2, 3, 224, 224)

    output = model(input)
    print(output.size())






