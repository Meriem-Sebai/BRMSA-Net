import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

class BasicConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x    
    
def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B*W, 1, 1)

class CrissCrossAttention(nn.Module):
    
    def __init__(self, dim):

        super(CrissCrossAttention,self).__init__()

        self.project_in = nn.Conv2d(dim, dim//2, kernel_size=1) 
        self.project_out = nn.Conv2d(dim//2, dim, kernel_size=1)  

        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_high, x_low):

        B, C, H, W = x_high.size() 

        q = self.project_in(x_low)     
        k = self.project_in(x_high)     
        v = self.project_in(x_high)     

        q_h = rearrange(q, 'b c h w -> (b w) h c') # [BxW, head, H, c]
        q_w = rearrange(q, 'b c h w -> (b h) w c') # [BxH, head, W, c]
        k_h = rearrange(k, 'b c h w -> (b w) c h') # [BxW, head, c, H]
        k_w = rearrange(k, 'b c h w -> (b h) c w') # [BxH, head, c, W]
        v_h = rearrange(v, 'b c h w -> (b w) c h') # [BxW, head, c, H]
        v_w = rearrange(v, 'b c h w -> (b h) c w') # [BxH, head, c, W]

        energy_h = q_h @ k_h + self.INF(B, H, W) # [BxW, head, H, H] 
        energy_h = rearrange(energy_h, '(b w) h1 h2 -> b h1 w h2', b=B, w=W) # [B, head, H, W, H]
        energy_w = q_w @ k_w # [BxH, head, W, W]
        energy_w = rearrange(energy_w, '(b h) w1 w2 -> b h w1 w2', b=B, h=H) # [B, head, H, W, H]
        concate = self.softmax(torch.cat([energy_h, energy_w], 3)) # [B, head, H, W, H+W]
        att_h = rearrange(concate[:, :, :, 0:H], 'b h1 w h2 -> (b w) h2 h1') # [BxW, head, H, H]
        att_w = rearrange(concate[:, :, :, H:H+W], 'b h w1 w2 -> (b h) w2 w1') # [BxH, head, W, W]
        out_h = v_h @ att_h # [BxW, head, c, H]
        out_h = rearrange(out_h, '(b w) c h -> b c h w', b=B, w=W) # [B, C, H, W]
        out_w = v_w @ att_w # [BxH, head, c, W]
        out_w = rearrange(out_w, '(b h) c w -> b c h w', b=B, h=H) # [B, C, H, W]

        out =  self.project_out(out_h + out_w) + x_high
        
        return out
    
class ChannelAttention(nn.Module):
    def __init__(self, dim, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(dim, dim//ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(dim//ratio, dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel=7):
        super(SpatialAttention, self).__init__()

        assert kernel in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class SpatialChannelAttention(nn.Module):
    def __init__(self, dim):
        super(SpatialChannelAttention, self).__init__()
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention()

    def forward(self, x):
        ca = self.ca(x) * x
        sa = self.sa(x) * x
        return ca + sa      
 
class CFF(nn.Module):
    def __init__(self, channel):
        super(CFF, self).__init__()
        self.ca_sa = SpatialChannelAttention(channel)
        self.cross_axis = CrissCrossAttention(channel)  

    def forward(self, T, X):
        L = self.ca_sa(X)   
        T = self.cross_axis(T, L)
        T = self.cross_axis(T, L) 
        T = torch.cat([L, T], dim=1)   
        return T      
 
class MFA(nn.Module):
    def __init__(self, channel=32):
        super(MFA, self).__init__()
        
        in_channels=[64, 128, 320, 512]

        self.Translayer2_1 = BasicConv2d(in_channels[1], channel, 1)
        self.Translayer3_1 = BasicConv2d(in_channels[2], channel, 1)
        self.Translayer4_1 = BasicConv2d(in_channels[3], channel, 1)
              
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)       
        self.CFF3 = CFF(channel)      

        self.conv3 = BasicConv2d(2 * channel, channel, 3, padding=1) 
        self.conv_upsample5 = BasicConv2d(channel, channel, 3, padding=1)       
        self.CFF2 = CFF(channel) 

        self.conv2 = BasicConv2d(2 * channel, channel, 3, padding=1)  
        self.conv_out = nn.Conv2d(channel, 1, 1)    
    
    def forward(self, inputs):
                
        x4 = inputs[3]
        x3 = inputs[2]
        x2 = inputs[1]

        x2 = self.Translayer2_1(x2)  
        x3 = self.Translayer3_1(x3)  
        x4 = self.Translayer4_1(x4)                     
                
        x2 = self.conv_upsample2(self.upsample(self.upsample(x4))) * self.conv_upsample3(self.upsample(x3)) * x2
        x3 = self.conv_upsample1(self.upsample(x4)) * x3 

        T4 = self.conv_upsample4(self.upsample(x4))       
        T3 = self.CFF3(T4, x3)   

        T3 = self.conv3(T3)      
        T3 = self.conv_upsample5(self.upsample(T3))   
        T2 = self.CFF2(T3, x2) 

        T2 = self.conv2(T2)    
        output = self.conv_out(T2)        
        
        return T2, output   
    
class BNPReLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim, eps=1e-3)
        self.acti = nn.PReLU(dim)

    def forward(self, input):
        x = self.bn(input)
        x = self.acti(x)        
        return x

class Conv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride=1, padding=0, dilation=1, groups=1, bn_act=False, bias=False):
        super().__init__()               
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)   
        self.bn_act = bn_act       
        if self.bn_act:
            self.bn_relu = BNPReLU(out_dim)
            
    def forward(self, input):
        x = self.conv(input)
        if self.bn_act:
            x = self.bn_relu(x)
        return x 
  
class Pooling(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = Conv(in_dim, out_dim, 1, 1, 0, bn_act=True)

    def forward(self, input):
        size = input.shape[-2:]
        x = self.pool(input)
        x = self.conv(x)
        x = nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)   
        return x 
   
class self_attn(nn.Module):
    def __init__(self, dim, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = Conv(dim, dim// 8, 1, 1, 0)
        self.key_conv = Conv(dim, dim// 8, 1, 1, 0)
        self.value_conv = Conv(dim, dim, 1, 1, 0)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).reshape(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).reshape(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.sigmoid(attention_map)
        projected_value = self.value_conv(x).reshape(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out
    
class AA(nn.Module):
    def __init__(self, dim):
        super(AA, self).__init__()
        self.conv0 = Conv(dim, dim, 1, 1, 0)
        self.conv1 = Conv(dim, dim, 3, 1, 1)
        self.Hattn = self_attn(dim, mode='h')
        self.Wattn = self_attn(dim, mode='w')

    def forward(self, input):
        x = self.conv0(input)
        x = self.conv1(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(Hx)

        return Wx    
   
class ISF(nn.Module):
    def __init__(self, dim, kernel=3, dilation=1):
        super(ISF, self).__init__()
        
        self.bn_relu_1 = BNPReLU(dim)
        self.conv1x1_1 = Conv(dim, dim//4, kernel, 1, 1, bn_act=True)            
                
        self.dconv_1_1 = Conv(dim//4, dim//16, kernel, 1, 1, 1, dim//16, bn_act=True)        
        self.dconv_1_2 = Conv(dim//16, dim//16, kernel, 1, 1, 1, dim//16, bn_act=True)        
        self.dconv_1_3 = Conv(dim//16, dim//8, kernel, 1, 1, 1, dim//16, bn_act=True)       
                
        self.dconv_2_1 = Conv(dim//4, dim//16, kernel, 1, int(dilation/4+1), int(dilation/4+1), dim//16, bn_act=True)        
        self.dconv_2_2 = Conv(dim//16, dim//16, kernel, 1, int(dilation/4+1), int(dilation/4+1), dim//16, bn_act=True)        
        self.dconv_2_3 = Conv(dim//16, dim//8, kernel, 1, int(dilation/4+1), int(dilation/4+1), dim//16, bn_act=True) 
                
        self.dconv_3_1 = Conv(dim//4, dim//16, kernel, 1, int(dilation/2+1), int(dilation/2+1), dim//16, bn_act=True)        
        self.dconv_3_2 = Conv(dim//16, dim//16, kernel, 1, int(dilation/2+1), int(dilation/2+1), dim//16, bn_act=True)        
        self.dconv_3_3 = Conv(dim//16, dim//8, kernel, 1, int(dilation/2+1), int(dilation/2+1), dim//16, bn_act=True)   

        self.dconv_4_1 = Conv(dim//4, dim//16, kernel, 1, 1*dilation+1, dilation+1, dim//16, bn_act=True)        
        self.dconv_4_2 = Conv(dim//16, dim//16, kernel, 1, 1*dilation+1, dilation+1, dim//16, bn_act=True)        
        self.dconv_4_3 = Conv(dim//16, dim//8, kernel, 1, 1*dilation+1, dilation+1, dim//16, bn_act=True)  

        self.conv_2_1 = nn.Conv2d(dim, dim//2, 1, 1)
        self.conv_2_2 = nn.Conv2d(dim, dim//2, 1, 1)  
        
        self.pool1 = Pooling(dim, dim//2) 
        self.pool2 = Pooling(dim, dim//2) 
        self.pool3 = Pooling(dim, dim)             

        self.conv1x1 = nn.Sequential(
                            Conv(2*dim, dim, 1, 1, 0, bn_act=True),  
                            nn.Dropout(0.1)
                        )    

        self.AA = AA(dim)           
        
    def forward(self, input):

        x = self.bn_relu_1(input)
        y = self.conv1x1_1(x)
        
        y1_1 = self.dconv_1_1(y)
        y1_2 = self.dconv_1_2(y1_1)
        y1_3 = self.dconv_1_3(y1_2)
        output_1 = torch.cat([y1_1, y1_2, y1_3], dim=1)
                
        y2_1 = self.dconv_2_1(y)
        y2_2 = self.dconv_2_2(y2_1)
        y2_3 = self.dconv_2_3(y2_2)
        output_2 = torch.cat([y2_1, y2_2, y2_3], dim=1) 

        output_1_2 = torch.cat([output_1, output_2], dim=1) 
        pool1 = self.pool1(x)
        output_1_2 = torch.cat([output_1_2, pool1], dim=1)
                
        y3_1 = self.dconv_3_1(y)
        y3_2 = self.dconv_3_2(y3_1)
        y3_3 = self.dconv_3_3(y3_2)
        output_3 = torch.cat([y3_1, y3_2, y3_3], dim=1)
                
        y4_1 = self.dconv_4_1(y)
        y4_2 = self.dconv_4_2(y4_1)
        y4_3 = self.dconv_4_3(y4_2)
        output_4 = torch.cat([y4_1, y4_2, y4_3], dim=1)    
                
        output_3_4 = torch.cat([output_3, output_4], dim=1)
        pool2 = self.pool2(x)  
        output_3_4 = torch.cat([output_3_4, pool2], dim=1)          
        
        output = torch.cat([self.conv_2_1(output_1_2), self.conv_2_2(output_3_4)], dim=1)
        pool3 = self.pool3(x)
        output = torch.cat([output, pool3], dim=1)       
        output = self.conv1x1(output) 

        out = output + input
        out = self.AA(out) + out  
                        
        return out  
 
def run_sobel(conv_x, conv_y, input):

    g_x = conv_x(input)
    g_y = conv_y(input)
    return torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))

def get_sobel(in_chan, out_chan):
    
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)    
    
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x).cuda()
    filter_y = torch.from_numpy(filter_y).cuda()

    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)

    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y    
   
    return conv_x, conv_y

def get_contour(input):
    
    sobel_x, sobel_y = get_sobel(1, 1)
    edge = run_sobel(sobel_x, sobel_y, input)     
    
    return edge    
   
class BRC(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(BRC, self).__init__()

        self.num_heads = num_heads           
        self.norm_weight = nn.Parameter(torch.ones((dim,)))
        self.norm_bias = nn.Parameter(torch.zeros((dim,)))               

    def forward(self, F, P):        

        P_map = torch.sigmoid(P)
        fg_map = torch.max(P_map - 0.5, torch.zeros_like(P_map))
        b_map = torch.sigmoid(get_contour(P_map))
        b_map = torch.max(b_map - 0.5, torch.zeros_like(b_map)) 
        bg_map = torch.max(0.5 - P_map, torch.zeros_like(P_map))  
        b_mask = torch.where(b_map > 0, 1.0, 0.0)
        fg_mask = torch.where(fg_map > 0, 1.0, 0.0)        
        bg_mask = torch.where(bg_map > 0, 1.0, 0.0)   

        N, C, H, W = F.shape         
        F = F.permute(0, 2, 3, 1).contiguous().view(F.shape[0], -1, F.shape[1])
        mu = F.mean(-1, keepdim=True)
        sigma = F.var(-1, keepdim=True, unbiased=False)
        F = (F - mu) / torch.sqrt(sigma+1e-5) * self.norm_weight + self.norm_bias
        F = F.view(F.shape[0], H, W, F.shape[2]).permute(0, 3, 1, 2).contiguous()       
        
        F_spatial = F.clone()        
        with torch.no_grad():
            for i in range(N):
                b_feat = []
                fg_feat = []
                b_loc = torch.argwhere(b_mask[i][0] == 1)
                fg_loc = torch.argwhere(fg_mask[i][0] == 1)          
                for j in range(C):
                    b_feat.append(F[i, j, b_loc[:, 0], b_loc[:, 1]])
                    fg_feat.append(F[i, j, fg_loc[:, 0], fg_loc[:, 1]])                
                b_feat = torch.stack(b_feat).cuda()
                fg_feat = torch.stack(fg_feat).cuda()

                k_spatial = fg_feat.view(self.num_heads, fg_feat.shape[0] // self.num_heads, fg_feat.shape[1]).permute(0, 2, 1)
                v_spatial = fg_feat.view(self.num_heads, fg_feat.shape[0] // self.num_heads, fg_feat.shape[1]).permute(0, 2, 1)
                q_spatial = b_feat.view(self.num_heads, b_feat.shape[0] // self.num_heads, b_feat.shape[1]).permute(0, 2, 1)
                q_spatial = nn.functional.normalize(q_spatial, dim=-1)
                k_spatial = nn.functional.normalize(k_spatial, dim=-1)   
                
                attn_spatial = (q_spatial @ k_spatial.transpose(-2, -1))
                attn_spatial = attn_spatial.softmax(dim=-1) 
                out_spatial = (attn_spatial @ v_spatial) + q_spatial                 
                out_spatial = out_spatial.permute(0, 2, 1).reshape(self.num_heads * out_spatial.shape[2], out_spatial.shape[1])
                for j in range(C):
                    F_spatial[i, j, b_loc[:, 0], b_loc[:, 1]] = out_spatial[j]    

        b_bg_mask = b_mask + bg_mask
        b_bg_mask = torch.where(b_bg_mask > 0, 1.0, 0.0)                             
        fg_feat = F * fg_mask
        b_bg_feat = F * b_bg_mask        
        k_channel = fg_feat.view(N, self.num_heads, -1, H * W)
        v_channel = fg_feat.view(N, self.num_heads, -1, H * W)
        q_channel = b_bg_feat.view(N, self.num_heads, -1, H * W)        
        q_channel = nn.functional.normalize(q_channel, dim=-1)
        k_channel = nn.functional.normalize(k_channel, dim=-1)    

        attn_channel = (q_channel @ k_channel.transpose(-2, -1))
        attn_channel = attn_channel.softmax(dim=-1)   
        out_channel = (attn_channel @ v_channel) + q_channel              
        F_channel = out_channel.view(N, C, H, W)
        F_out = F + F_spatial + F_channel
        
        return F_out
        
class IBR(nn.Module):
    def __init__(self, cdim, pdim, dim=32):
        super(IBR, self).__init__()

        self.ISF = ISF(cdim, dilation=8)        
        self.BRC = BRC(cdim)
        self.cat = Conv(pdim+cdim, cdim, 3, 1, 1, bn_act=True)     
        self.out_conv1 = Conv(cdim, dim, 3, 1, 1, bn_act=True)
        self.out_conv2 = Conv(dim, dim, 3, 1, 1, bn_act=True)
        self.out_conv3 = Conv(dim, 1, 3, 1, 1, bn_act=True)                   

    def forward(self, X, prev_F, prev_P):   

        isf = self.ISF(X)                
                             
        prev_F = nn.functional.interpolate(prev_F, size=X.size()[-2:], mode='bilinear', align_corners=True)
        prev_F = self.cat(torch.cat([isf, prev_F], dim=1))
        prev_P = nn.functional.interpolate(prev_P, size=X.size()[-2:], mode='bilinear')
        F = self.BRC(prev_F, prev_P)
        
        P = self.out_conv1(F) 
        P = self.out_conv2(P)          
        P = self.out_conv3(P) 
        
        return F, P, prev_P 

    

    

    
