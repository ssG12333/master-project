import torch
import torch.nn as nn
import torch.nn.functional as F

class Focus(nn.Module):
    def __init__(self, c):
        super(Focus, self).__init__()

    def forward(self, x):
        x = x.unsqueeze(0)
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 0)

class WADD(nn.Module):
    def __init__(self, c1, c2):
        super(WADD, self).__init()

        self.inter_dim = c1
        compress_c = 8

        self.weight_level_0 = nn.Conv2d(c1, compress_c, kernel_size=1, stride=1)
        self.weight_level_1 = nn.Conv2d(c1, compress_c, kernel_size=1, stride=1)
        self.weights_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1)
        self.conv = nn.Conv2d(c1, c1, kernel_size=3, padding=1)

    def forward(self, x0, x1):
        x_level_0, x_level_1 = x0, x1
        level_0_weight_v = self.weight_level_0(x_level_0)
        level_1_weight_v = self.weight_level_1(x_level_1)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), dim=1)
        levels_weight = F.softmax(levels_weight_v, dim=1)
        fused_out_reduced = x_level_0 * levels_weight[:, 0:1] + x_level_1 * levels_weight[:, 1:2]
        return fused_out_reduced

class CLA(nn.Module):
    def __init__(self, c, topk=3, nheads=8):
        super(CLA, self).__init()
        self.c_ = c
        self.num_heads = nheads
        assert self.c_ % nheads == 0, 'dim must be divisible by num_heads!'
        self.q = nn.Linear(self.c_, self.c_)
        self.k = nn.Linear(self.c_, self.c_)
        self.v = nn.Linear(self.c_, self.c_)
        self.attend = nn.Softmax(dim=-1)
        self.output_linear = nn.Conv2d(self.c_, self.c_, kernel_size=1)
        self.focus = Focus(self.c_)
        self.topk = topk 
        self.scale = self.c_ ** -0.5
        # self.add = WADD(self.c_, self.c_)

    def forward(self, x):
        x1 = x[0]  
        x2 = x[1] 
        b1, c1, w1, h1 = x1.shape
        b2, c2, w2, h2 = x2.shape
        head_dim = c2 // self.num_heads 
        x1_ = self.focus(x1).permute(0, 1, 3, 4, 2) # (k n c h w )-->(k n h w c)
        x2_ = x2.unsqueeze(0).permute(0, 1, 3, 4, 2) # (n c h w)-->(1 n h w c)
        q1, k1, v1, q2 = self.q(x1_), self.k(x1_), self.v(x1_), self.q(x2_)

        _, idx_r = torch.topk(q2.flatten(2, 3) * k1.flatten(2, 3), k=self.topk, dim=0, largest=True)  # n (hw) k c long tensor
        # idx_r = idx_r[1:]
        idx_r = idx_r.unsqueeze(1).view(b1, self.num_heads, h2 * w2, -1, head_dim) #(n nh hw k headdim)
        qq2 = q2.view(b1, self.num_heads, h2 * w2, -1, head_dim)

        qq = q1.view(b1, self.num_heads, h2 * w2, -1, head_dim)
        kk = k1.view(b1, self.num_heads, h2 * w2, -1, head_dim)
        vv = v1.view(b1, self.num_heads, h2 * w2, -1, head_dim) # (n nh hw -1 headdim)

        q_g = torch.gather(qq, -2, idx_r)
        k_g = torch.gather(kk, -2, idx_r)  # (n nh hw k headdim)
        v_g = torch.gather(vv, -2, idx_r) # (n nh hw -1 headdim)

        att = self.attend(self.scale * (qq2 - q_g) * k_g)
        out = att * v_g # (n nh hw k hdim)
        out = torch.mean(out, dim=-2).view(b2, c2, h2, w2) # (n nh hw hdim)--> nchw
        out = self.output_linear(out)
        # return self.add(out, x2)
        return (out + x2) / 2






# class Focus(nn.Layer):
#     # focus wh information into c-space
#     def __init__(self, c):
#         super(Focus, self).__init__()

#     def forward(self, x):
#         x = x.unsqueeze(0)
#         return paddle.concat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 0)  # k n c h w


# class WADD(nn.Layer):
#     def __init__(self, c1, c2):
#         super(WADD, self).__init__()

#         self.inter_dim = c1  # Assuming both c1 and c2 have the same dimension
#         compress_c = 8

#         self.weight_level_0 = paddle.nn.Conv2D(self.inter_dim, compress_c, kernel_size=1, stride=1)
#         self.weight_level_1 = paddle.nn.Conv2D(self.inter_dim, compress_c, kernel_size=1, stride=1)
#         self.weights_levels = paddle.nn.Conv2D(compress_c * 2, 2, kernel_size=1, stride=1)
#         self.conv = paddle.nn.Conv2D(self.inter_dim, self.inter_dim, kernel_size=3, padding=1)

#     def forward(self, x0, x1):
#         x_level_0, x_level_1 = x0, x1
#         level_0_weight_v = self.weight_level_0(x_level_0)
#         level_1_weight_v = self.weight_level_1(x_level_1)
#         levels_weight_v = paddle.concat((level_0_weight_v, level_1_weight_v), axis=1)
#         levels_weight = self.weights_levels(levels_weight_v)
#         levels_weight = F.softmax(levels_weight, axis=1)
#         fused_out_reduced = x_level_0 * levels_weight[:, 0:1] + x_level_1 * levels_weight[:, 1:2]
#         return fused_out_reduced

# class CLA(nn.Layer):
#     def __init__(self, c, topk=3, nheads=8):
#         super(CLA, self).__init__()
#         self.c_ = c
#         self.num_heads = nheads
#         assert self.c_ % nheads == 0, 'dim must be divisible by num_heads!'
#         self.q = nn.Linear(self.c_, self.c_)
#         self.k = nn.Linear(self.c_, self.c_)
#         self.v = nn.Linear(self.c_, self.c_)
#         self.attend = nn.Softmax(axis=-1)
#         self.output_linear = nn.Conv2D(self.c_, self.c_, kernel_size=1)
#         self.focus = Focus(self.c_)
#         self.topk = topk 
#         self.scale = self.c_ ** -0.5
#         # self.add = WADD(self.c_, self.c_)

#     def forward(self, x):
#         x1 = x[0]  
#         x2 = x[1] 
#         b1, c1, w1, h1 = x1.shape
#         b2, c2, w2, h2 = x2.shape
#         head_dim = c2//self.num_heads 
#         x1_ = self.focus(x1).transpose([0, 1, 3, 4, 2]) # (k n c h w )-->(k n h w c)
#         # x1_ = x1.unsqueeze(0).reshape([-1, b2, c2, w2, h2]).transpose([0, 1, 3, 4, 2])
#         x2_ = x2.unsqueeze(0).transpose([0, 1, 3, 4, 2]) # (n c h w)-->(1 n h w c)
#         q1, k1, v1, q2= self.q(x1_), self.k(x1_), self.v(x1_), self.q(x2_)

#         _, idx_r = paddle.topk(q2.flatten(2, 3)*k1.flatten(2, 3), 
#                                 k=self.topk, 
#                                 axis=0, 
#                                 largest=True)  # n (hw) k c long tensor
#         # idx_r = idx_r[1:]
#         idx_r = idx_r.unsqueeze(1).reshape([b1, self.num_heads, h2*w2, -1, head_dim]) #(n nh hw k headdim)
#         qq2 = q2.reshape([b1, self.num_heads, h2*w2, -1, head_dim])

#         qq = q1.reshape([b1, self.num_heads, h2*w2, -1, head_dim])
#         kk = k1.reshape([b1, self.num_heads, h2*w2, -1, head_dim])
#         vv = v1.reshape([b1, self.num_heads, h2*w2, -1, head_dim]) # (n nh hw -1 headdim)

#         q_g = paddle.take_along_axis(qq,
#                 idx_r,
#                 axis=-2)
#         k_g = paddle.take_along_axis(kk, 
#                 idx_r, 
#                 axis=-2)  # (n nh hw k headdim)
#         v_g = paddle.take_along_axis(vv,
#                 idx_r, 
#                 axis= -2) # (n nh hw -1 headdim)

#         att = self.attend(self.scale*(qq2-q_g)*k_g)
#         out = att*v_g # (n nh hw k hdim)
#         out = paddle.mean(out, axis=-2).reshape([b2, c2, h2, w2]) # (n nh hw hdim)--> nchw
#         out = self.output_linear(out)
#         # return self.add(out, x2)
#         return (out + x2) / 2