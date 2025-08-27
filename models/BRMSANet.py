import torch
import torch.nn as nn
import torch.nn.functional as F

from .lib.BRMSANet_modules import MFA, IBR
from .lib.mix_transformer import mit_b3
 
class BRMSANet(nn.Module):
    
    def __init__(self, pretrained=None):
        super(BRMSANet, self).__init__()        
        self.backbone = mit_b3()
        self.backbone.init_weights(pretrained=pretrained)   

        self.MFA = MFA(32) 

        self.IBR3 = IBR(512, 32, 32) 
        self.IBR2 = IBR(320, 512, 32)
        self.IBR1 = IBR(128, 320, 32)           
                
    def forward(self, x):

        segout = self.backbone(x)
        x1 = segout[0]  
        x2 = segout[1]  
        x3 = segout[2]  
        x4 = segout[3]  

        F5, P5 = self.MFA([x1, x2, x3, x4]) 
        P5_up = nn.functional.interpolate(P5, scale_factor=8, mode='bilinear')         

        F4, P4, P5 = self.IBR3(x4, F5, P5)
        P4 = P4 + P5     
        P4_up = nn.functional.interpolate(P4, scale_factor=32, mode='bilinear')                

        F3, P3, P4 = self.IBR2(x3, F4, P4)
        P3 = P3 + P4  
        P3_up = nn.functional.interpolate(P3, scale_factor=16, mode='bilinear')            

        _, P, P3 = self.IBR1(x2, F3, P3)
        P = P + P3   
        P_up = nn.functional.interpolate(P, scale_factor=8, mode='bilinear')   

        return P_up, P3_up, P4_up, P5_up
        
        
        
        