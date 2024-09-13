import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange


class ShiftConv2d0(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d0, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_div = 5
        g = inp_channels // self.n_div

        conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)
        mask = nn.Parameter(torch.zeros((self.out_channels, self.inp_channels, 3, 3)), requires_grad=False)
        mask[:, 0 * g:1 * g, 1, 2] = 1.0
        mask[:, 1 * g:2 * g, 1, 0] = 1.0
        mask[:, 2 * g:3 * g, 2, 1] = 1.0
        mask[:, 3 * g:4 * g, 0, 1] = 1.0
        mask[:, 4 * g:, 1, 1] = 1.0
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = mask

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1)
        return y


class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d1, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 5
        g = inp_channels // self.n_div
        self.weight[0 * g:1 * g, 0, 1, 2] = 1.0  ## left
        self.weight[1 * g:2 * g, 0, 1, 0] = 1.0  ## right
        self.weight[2 * g:3 * g, 0, 2, 1] = 1.0  ## up
        self.weight[3 * g:4 * g, 0, 0, 1] = 1.0  ## down
        self.weight[4 * g:, 0, 1, 1] = 1.0  ## identity

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        y = self.conv1x1(y)
        return y


class ShiftConv2d(nn.Module):
    def __init__(self, inp_channels, out_channels, conv_type='fast-training-speed'):
        super(ShiftConv2d, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        if conv_type == 'low-training-memory':
            self.shift_conv = ShiftConv2d0(inp_channels, out_channels)
        elif conv_type == 'fast-training-speed':
            self.shift_conv = ShiftConv2d1(inp_channels, out_channels)
        else:
            raise ValueError('invalid type of shift-conv2d')

    def forward(self, x):
        y = self.shift_conv(x)
        return y


class LFE(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type='relu'):
        super(LFE, self).__init__()
        self.exp_ratio = exp_ratio
        self.act_type = act_type

        self.conv0 = ShiftConv2d(inp_channels, out_channels * exp_ratio)
        self.conv1 = ShiftConv2d(out_channels * exp_ratio, out_channels)

        if self.act_type == 'linear':
            self.act = None
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError('unsupport type of activation')

    def forward(self, x):
        y = self.conv0(x)
        y = self.act(y)
        y = self.conv1(y)
        return y


class GMSA(nn.Module):
    def __init__(self, channels, shifts=4, window_sizes=None, calc_attn=True):
        super(GMSA, self).__init__()
        if window_sizes is None:
            window_sizes = [4, 8, 12]
        self.channels = channels
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.calc_attn = calc_attn

        if self.calc_attn:
            self.split_chns = [channels * 2 // 3, channels * 2 // 3, channels * 2 // 3]
            self.project_inp = nn.Sequential(
                nn.Conv2d(self.channels, self.channels * 2, kernel_size=1),
                nn.BatchNorm2d(self.channels * 2)
            )
            self.project_out = nn.Conv2d(channels, channels, kernel_size=1)
        else:
            self.split_chns = [channels // 3, channels // 3, channels // 3]
            self.project_inp = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=1),
                nn.BatchNorm2d(self.channels)
            )
            self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, prev_atns=None):
        b, c, h, w = x.shape
        x = self.project_inp(x)
        xs = torch.split(x, self.split_chns, dim=1)
        ys = []
        atns = []
        if prev_atns is None:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
                q, v = rearrange(
                    x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c',
                    qv=2, dh=wsize, dw=wsize
                )
                atn = (q @ q.transpose(-2, -1))
                atn = atn.softmax(dim=-1)
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
                    h=h // wsize, w=w // wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
                ys.append(y_)
                atns.append(atn)
            y = torch.cat(ys, dim=1)
            y = self.project_out(y)
            return y, atns
        else:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
                atn = prev_atns[idx]
                v = rearrange(
                    x_, 'b (c) (h dh) (w dw) -> (b h w) (dh dw) c',
                    dh=wsize, dw=wsize
                )
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
                    h=h // wsize, w=w // wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
                ys.append(y_)
            y = torch.cat(ys, dim=1)
            y = self.project_out(y)
            return y, prev_atns


class CGMSA(nn.Module):
    def __init__(self, channels, shifts=4, window_sizes=None, calc_attn=True):
        super(CGMSA, self).__init__()
        if window_sizes is None:
            window_sizes = [4, 8, 12]
        self.channels = channels
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.calc_attn = calc_attn

        if self.calc_attn:
            self.split_chns = [channels * 2 // 3, channels * 2 // 3, channels * 2 // 3]
            self.project_inp = nn.Sequential(
                nn.Conv2d(self.channels, self.channels * 2, kernel_size=1),
                nn.BatchNorm2d(self.channels * 2)
            )
            self.project_out = nn.Conv2d(channels, channels, kernel_size=1)
        else:
            self.split_chns = [channels // 3, channels // 3, channels // 3]
            self.project_inp = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=1),
                nn.BatchNorm2d(self.channels)
            )
            self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, y, prev_atns=None):
        b, c, h, w = x.shape
        x = self.project_inp(x)
        y = self.project_inp(y)
        xs = torch.split(x, self.split_chns, dim=1)
        if prev_atns is None:
            ys = torch.split(y, self.split_chns, dim=1)
        y1s = []
        atns = []
        if prev_atns is None:
            for idx, x_ in enumerate(xs):
                y_ = ys[idx]
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
                    y_ = torch.roll(y_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
                qx, vx = rearrange(
                    x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c',
                    qv=2, dh=wsize, dw=wsize
                )
                qy, vy = rearrange(
                    y_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c',
                    qv=2, dh=wsize, dw=wsize
                )
                atn = (qx @ qy.transpose(-2, -1))
                atn = atn.softmax(dim=-1)
                y1_ = (atn @ vx)
                y1_ = rearrange(
                    y1_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
                    h=h // wsize, w=w // wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y1_ = torch.roll(y1_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
                y1s.append(y1_)
                atns.append(atn)
            y1 = torch.cat(y1s, dim=1)
            y1 = self.project_out(y1)
            return y1, atns
        else:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
                atn = prev_atns[idx]
                vx = rearrange(
                    x_, 'b (c) (h dh) (w dw) -> (b h w) (dh dw) c',
                    dh=wsize, dw=wsize
                )
                y1_ = (atn @ vx)
                y1_ = rearrange(
                    y1_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
                    h=h // wsize, w=w // wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y1_ = torch.roll(y1_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
                y1s.append(y1_)
            y1 = torch.cat(y1s, dim=1)
            y1 = self.project_out(y1)
            return y1, atns


class ELAB(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, shifts=0, window_sizes=None, shared_depth=1):
        super(ELAB, self).__init__()
        if window_sizes is None:
            window_sizes = [4, 8, 16]
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.shared_depth = shared_depth

        modules_lfe = {}
        modules_gmsa = {}
        modules_lfe['lfe_0'] = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
        modules_gmsa['gmsa_0'] = GMSA(channels=inp_channels, shifts=shifts, window_sizes=window_sizes, calc_attn=True)
        for i in range(shared_depth):
            modules_lfe['lfe_{}'.format(i + 1)] = LFE(inp_channels=inp_channels, out_channels=out_channels,
                                                      exp_ratio=exp_ratio)
            modules_gmsa['gmsa_{}'.format(i + 1)] = GMSA(channels=inp_channels, shifts=shifts,
                                                         window_sizes=window_sizes, calc_attn=False)
        self.modules_lfe = nn.ModuleDict(modules_lfe)
        self.modules_gmsa = nn.ModuleDict(modules_gmsa)

    def forward(self, x):
        atn = None
        for i in range(1 + self.shared_depth):
            if i == 0:  # only calculate attention for the 1-st module
                x = self.modules_lfe['lfe_{}'.format(i)](x) + x
                y, atn = self.modules_gmsa['gmsa_{}'.format(i)](x, atn)
                x = y + x
            else:
                x = self.modules_lfe['lfe_{}'.format(i)](x) + x
                y, atn = self.modules_gmsa['gmsa_{}'.format(i)](x, atn)
                x = y + x
        return x


class CELAB(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, shifts=0, window_sizes=None, shared_depth=1):
        super(CELAB, self).__init__()
        if window_sizes is None:
            window_sizes = [4, 8, 16]
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.shared_depth = shared_depth

        modules_lfe_X = {}
        modules_gmsa_X = {}
        modules_lfe_X['lfe_0'] = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
        modules_gmsa_X['gmsa_0'] = GMSA(channels=inp_channels, shifts=shifts, window_sizes=window_sizes, calc_attn=True)
        for i in range(shared_depth):
            modules_lfe_X['lfe_{}'.format(i + 1)] = LFE(inp_channels=inp_channels, out_channels=out_channels,
                                                        exp_ratio=exp_ratio)
            modules_gmsa_X['gmsa_{}'.format(i + 1)] = GMSA(channels=inp_channels, shifts=shifts,
                                                           window_sizes=window_sizes, calc_attn=False)
        self.modules_lfe_X = nn.ModuleDict(modules_lfe_X)
        self.modules_gmsa_X = nn.ModuleDict(modules_gmsa_X)

        modules_lfe_Y = {}
        modules_gmsa_Y = {}
        modules_lfe_Y['lfe_0'] = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
        modules_gmsa_Y['gmsa_0'] = GMSA(channels=inp_channels, shifts=shifts, window_sizes=window_sizes, calc_attn=True)
        for i in range(shared_depth):
            modules_lfe_Y['lfe_{}'.format(i + 1)] = LFE(inp_channels=inp_channels, out_channels=out_channels,
                                                        exp_ratio=exp_ratio)
            modules_gmsa_Y['gmsa_{}'.format(i + 1)] = GMSA(channels=inp_channels, shifts=shifts,
                                                           window_sizes=window_sizes, calc_attn=False)
        self.modules_lfe_Y = nn.ModuleDict(modules_lfe_Y)
        self.modules_gmsa_Y = nn.ModuleDict(modules_gmsa_Y)

        modules_lfe_Z = {}
        modules_gmsa_Z = {}
        modules_lfe_Z['lfe_0'] = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
        modules_gmsa_Z['gmsa_0'] = GMSA(channels=inp_channels, shifts=shifts, window_sizes=window_sizes, calc_attn=True)
        for i in range(shared_depth):
            modules_lfe_Z['lfe_{}'.format(i + 1)] = LFE(inp_channels=inp_channels, out_channels=out_channels,
                                                        exp_ratio=exp_ratio)
            modules_gmsa_Z['gmsa_{}'.format(i + 1)] = GMSA(channels=inp_channels, shifts=shifts,
                                                           window_sizes=window_sizes, calc_attn=False)
        self.modules_lfe_Z = nn.ModuleDict(modules_lfe_Z)
        self.modules_gmsa_Z = nn.ModuleDict(modules_gmsa_Z)

        modules_lfe_XYZ_X = {}
        modules_lfe_XYZ_Y = {}
        modules_lfe_XYZ_Z = {}
        modules_gmsa_XY = {}
        modules_gmsa_YZ = {}
        modules_lfe_XYZ_X['lfe_0'] = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
        modules_lfe_XYZ_Y['lfe_0'] = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
        modules_lfe_XYZ_Z['lfe_0'] = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
        modules_gmsa_XY['gmsa_0'] = CGMSA(channels=inp_channels, shifts=shifts, window_sizes=window_sizes,
                                          calc_attn=True)
        modules_gmsa_YZ['gmsa_0'] = CGMSA(channels=inp_channels, shifts=shifts, window_sizes=window_sizes,
                                          calc_attn=True)
        for i in range(shared_depth):
            modules_lfe_XYZ_X['lfe_{}'.format(i + 1)] = LFE(inp_channels=inp_channels, out_channels=out_channels,
                                                           exp_ratio=exp_ratio)
            modules_lfe_XYZ_Y['lfe_{}'.format(i + 1)] = LFE(inp_channels=inp_channels, out_channels=out_channels,
                                                           exp_ratio=exp_ratio)
            modules_lfe_XYZ_Z['lfe_{}'.format(i + 1)] = LFE(inp_channels=inp_channels, out_channels=out_channels,
                                                            exp_ratio=exp_ratio)
            modules_gmsa_XY['gmsa_{}'.format(i + 1)] = CGMSA(channels=inp_channels, shifts=shifts,
                                                             window_sizes=window_sizes, calc_attn=False)
            modules_gmsa_YZ['gmsa_{}'.format(i + 1)] = CGMSA(channels=inp_channels, shifts=shifts,
                                                             window_sizes=window_sizes, calc_attn=False)
        self.modules_lfe_XYZ_X = nn.ModuleDict(modules_lfe_XYZ_X)
        self.modules_lfe_XYZ_Y = nn.ModuleDict(modules_lfe_XYZ_Y)
        self.modules_lfe_XYZ_Z = nn.ModuleDict(modules_lfe_XYZ_Z)
        self.modules_gmsa_XY = nn.ModuleDict(modules_gmsa_XY)
        self.modules_gmsa_YZ = nn.ModuleDict(modules_gmsa_YZ)

    def forward(self, x, y, z):
        atn = None
        for i in range(1 + self.shared_depth):
            if i == 0:  # only calculate attention for the 1-st module
                x = self.modules_lfe_X['lfe_{}'.format(i)](x) + x
                x1, atn = self.modules_gmsa_X['gmsa_{}'.format(i)](x, atn)
                x = x1 + x
            else:
                x = self.modules_lfe_X['lfe_{}'.format(i)](x) + x
                x1, atn = self.modules_gmsa_X['gmsa_{}'.format(i)](x, atn)
                x = x1 + x

        atn = None
        for i in range(1 + self.shared_depth):
            if i == 0:  # only calculate attention for the 1-st module
                y = self.modules_lfe_Y['lfe_{}'.format(i)](y) + y
                y1, atn = self.modules_gmsa_Y['gmsa_{}'.format(i)](y, atn)
                y = y1 + y
            else:
                y = self.modules_lfe_Y['lfe_{}'.format(i)](y) + y
                y1, atn = self.modules_gmsa_Y['gmsa_{}'.format(i)](y, atn)
                y = y1 + y

        atn = None
        for i in range(1 + self.shared_depth):
            if i == 0:  # only calculate attention for the 1-st module
                z = self.modules_lfe_Z['lfe_{}'.format(i)](z) + z
                z1, atn = self.modules_gmsa_Z['gmsa_{}'.format(i)](z, atn)
                z = z1 + z
            else:
                z = self.modules_lfe_Z['lfe_{}'.format(i)](z) + z
                z1, atn = self.modules_gmsa_Z['gmsa_{}'.format(i)](z, atn)
                z = z1 + z

        atn1 = None
        atn2 = None
        for i in range(1 + self.shared_depth):
            if i == 0:  # only calculate attention for the 1-st module
                x = self.modules_lfe_XYZ_X['lfe_{}'.format(i)](x) + x
                y = self.modules_lfe_XYZ_Y['lfe_{}'.format(i)](y) + y
                z = self.modules_lfe_XYZ_Z['lfe_{}'.format(i)](y) + z
                y1, atn1 = self.modules_gmsa_XY['gmsa_{}'.format(i)](x, y, atn1)
                y2, atn2 = self.modules_gmsa_YZ['gmsa_{}'.format(i)](z, y, atn2)
                y = y1 + y2 + y
            else:
                x = self.modules_lfe_XYZ_X['lfe_{}'.format(i)](x) + x
                y = self.modules_lfe_XYZ_Y['lfe_{}'.format(i)](y) + y
                z = self.modules_lfe_XYZ_Z['lfe_{}'.format(i)](y) + z
                y1, atn1 = self.modules_gmsa_XY['gmsa_{}'.format(i)](x, y, atn1)
                y2, atn2 = self.modules_gmsa_YZ['gmsa_{}'.format(i)](z, y, atn2)
                y = y1 + y2 + y

        return y


class MGT(nn.Module):
    def __init__(self, in_chans=1, embed_dim=60, Ex_depths=3, Fusion_depths=3, Re_depths=3,
                 norm_layer=nn.LayerNorm, img_range=1., window_sizes=None, **kwargs):
        super(MGT, self).__init__()
        num_out_ch = in_chans
        self.img_range = img_range
        embed_dim_temp = int(embed_dim / 2)
        self.mean = torch.zeros(1, 1, 1, 1)
        if window_sizes is None:
            self.window_sizes = [4, 8, 16]

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first1_A = nn.Conv2d(in_chans, embed_dim_temp, 3, 1, 1)
        self.conv_first2_A = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.Ex_num_layers = Ex_depths
        self.Fusion_num_layers = Fusion_depths
        self.Re_num_layers = Re_depths
        self.embed_dim = embed_dim
        self.num_features = embed_dim

        self.layers_Ex_A = nn.ModuleList()
        for i_layer in range(self.Ex_num_layers):
            layer = ELAB(inp_channels=embed_dim, out_channels=embed_dim, shifts=i_layer % 2, window_sizes=self.window_sizes)
            self.layers_Ex_A.append(layer)
        self.norm_Ex_A = norm_layer(self.num_features)

        self.layers_Ex_B = nn.ModuleList()
        for i_layer in range(self.Ex_num_layers):
            layer = ELAB(inp_channels=embed_dim, out_channels=embed_dim, shifts=i_layer % 2, window_sizes=self.window_sizes)
            self.layers_Ex_B.append(layer)
        self.norm_Ex_B = norm_layer(self.num_features)

        #####################################################################################################
        ###################################### 3, deep feature fusion #######################################
        # obtain fused features
        self.conv_before_cross_attention1 = nn.Conv2d(2 * embed_dim, 2 * embed_dim, 3, 1, 1)
        self.conv_before_cross_attention2 = nn.Conv2d(2 * embed_dim, embed_dim, 3, 1, 1)

        # calculate cross attention
        self.layers_Fusion = nn.ModuleList()
        for i_layer in range(self.Fusion_num_layers):
            layer = CELAB(inp_channels=embed_dim, out_channels=embed_dim, shifts=i_layer % 2, window_sizes=self.window_sizes)
            self.layers_Fusion.append(layer)
        self.norm_Fusion = norm_layer(self.num_features)

        self.layers_Re = nn.ModuleList()
        for i_layer in range(self.Re_num_layers):
            layer = ELAB(inp_channels=embed_dim, out_channels=embed_dim, shifts=i_layer % 2, window_sizes=self.window_sizes)
            self.layers_Re.append(layer)
        self.norm_Re = norm_layer(self.num_features)

        #####################################################################################################
        ################################ 4, high quality image reconstruction ###############################
        self.conv_last1 = nn.Conv2d(embed_dim, embed_dim_temp, 3, 1, 1)
        self.conv_last2 = nn.Conv2d(embed_dim_temp, int(embed_dim_temp / 2), 3, 1, 1)
        self.conv_last3 = nn.Conv2d(int(embed_dim_temp / 2), num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize * self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features_Ex_A(self, x):
        x = self.lrelu(self.conv_first1_A(x))
        x = self.lrelu(self.conv_first2_A(x))
        xs = []
        for layer in self.layers_Ex_A:
            x = layer(x)
            xs.append(x)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm_Ex_A(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x, xs

    def forward_features_Ex_B(self, x):
        x = self.lrelu(self.conv_first1_A(x))
        x = self.lrelu(self.conv_first2_A(x))
        xs = []
        for layer in self.layers_Ex_B:
            x = layer(x)
            xs.append(x)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm_Ex_B(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x, xs

    def forward_features_Fusion(self, x, y, xs, ys):
        fe = torch.cat([x, y], 1)
        fe = self.lrelu(self.conv_before_cross_attention1(fe))
        fe = self.lrelu(self.conv_before_cross_attention2(fe))

        count = 0
        for layer in self.layers_Fusion:
            fe1 = layer(xs[count], fe, ys[count])
            fe = fe + fe1
            count += 1

        B, C, H, W = fe.shape
        fe = fe.flatten(2).transpose(1, 2)
        fe = self.norm_Fusion(fe)
        fe = fe.transpose(1, 2).view(B, C, H, W)

        return fe

    def forward_features_Re(self, x):
        for layer in self.layers_Re:
            x = layer(x)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm_Re(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.lrelu(self.conv_last1(x))
        x = self.lrelu(self.conv_last2(x))
        x = self.conv_last3(x)
        return x

    def forward(self, A, B):
        x = A
        y = B
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        y = self.check_image_size(y)

        self.mean_A = self.mean.type_as(x)
        self.mean_B = self.mean.type_as(y)
        self.mean = (self.mean_A + self.mean_B) / 2

        x = (x - self.mean_A) * self.img_range
        y = (y - self.mean_B) * self.img_range

        # Feedforward
        x, xs = self.forward_features_Ex_A(x)
        y, ys = self.forward_features_Ex_B(y)
        x = self.forward_features_Fusion(x, y, xs, ys)
        # fe_F = x
        x = self.forward_features_Re(x)
        x = x / self.img_range + self.mean
        return x[:, :, :H, :W]  # , fe_A[:, :, :H, :W], fe_B[:, :, :H, :W], fe_F[:, :, :H, :W]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers_Ex_A):
            flops += layer.flops()
        for i, layer in enumerate(self.layers_Ex_B):
            flops += layer.flops()
        for i, layer in enumerate(self.layers_Fusion):
            flops += layer.flops()
        for i, layer in enumerate(self.layers_Re):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        return flops


if __name__ == '__main__':
    height = 480
    width = 640
    model = MGT(Ex_depths=3, Fusion_depths=3, Re_depths=3).cuda()
    x = torch.rand(1, 1, height, width).cuda()
    # y = model(x, x)
    # print(y)
    tensor = (x, x)
    import thop
    flops, params = thop.profile(model, inputs=tensor)  # input
    print(flops / 1e9)
    print(params)
    par = 0
    for p in model.parameters():
        par += p.view(-1).size()[0]
    print('parameter', par)
