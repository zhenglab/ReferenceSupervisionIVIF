
from matplotlib import image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
from utils.utils_color import RGB_HSV, RGB_YCbCr
from models.loss_ssim import ssim
import torchvision.transforms.functional as TF

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_C,  image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_C = self.sobelconv(image_C)

        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(torch.max(image_A, image_B), image_C)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient

class L_Grad2(nn.Module):
    def __init__(self):
        super(L_Grad2, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B,  image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        # gradient_C = self.sobelconv(image_C)

        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(image_A, image_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient
    
class L_Grad5(nn.Module):
    def __init__(self):
        super(L_Grad5, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_C, image_D,image_E, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_C = self.sobelconv(image_C)
        gradient_D = self.sobelconv(image_D)
        gradient_E = self.sobelconv(image_E)

        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(torch.max(torch.max(torch.max(image_A, image_B), image_C),image_D),image_E)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient

class L_Grad5_avg(nn.Module):
    def __init__(self):
        super(L_Grad5_avg, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_C, image_D,image_E, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_C = self.sobelconv(image_C)
        gradient_D = self.sobelconv(image_D)
        gradient_E = self.sobelconv(image_E)

        # gradient_fused = self.sobelconv(image_fused)
        # gradient_joint = torch.max(torch.max(torch.max(torch.max(image_A, image_B), image_C),image_D),image_E)
        Loss_gradient = F.l1_loss(image_A, image_fused)+F.l1_loss(image_B, image_fused)+F.l1_loss(image_C, image_fused)+F.l1_loss(image_D, image_fused)+F.l1_loss(image_E, image_fused)
        return Loss_gradient    
class L_Grad7(nn.Module):
    def __init__(self):
        super(L_Grad7, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_C, image_D,image_E, image_F,image_G,image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_C = self.sobelconv(image_C)
        gradient_D = self.sobelconv(image_D)
        gradient_E = self.sobelconv(image_E)
        gradient_F = self.sobelconv(image_F)
        gradient_G = self.sobelconv(image_G)

        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(image_A, image_B), image_C),image_D),image_E),image_F),image_G)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient
    
class L_Grad9(nn.Module):
    def __init__(self):
        super(L_Grad9, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_C, image_D,image_E, image_F,image_G,image_H,image_I,image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_C = self.sobelconv(image_C)
        gradient_D = self.sobelconv(image_D)
        gradient_E = self.sobelconv(image_E)
        gradient_F = self.sobelconv(image_F)
        gradient_G = self.sobelconv(image_G)
        gradient_H = self.sobelconv(image_H)
        gradient_I = self.sobelconv(image_I)

        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(image_A, image_B), image_C),image_D),image_E),image_F),image_G),image_H),image_I)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient



class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_C, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_C = self.sobelconv(image_C)
        
        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C)
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C)
        weight_C = torch.mean(gradient_C) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C)
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused) + weight_C * ssim(image_C, image_fused)
        return Loss_SSIM
    
class L_SSIM2(nn.Module):
    def __init__(self):
        super(L_SSIM2, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        # gradient_C = self.sobelconv(image_C)
        
        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B)) 
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B)) 
        # weight_C = torch.mean(gradient_C) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C)
        # print()
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return Loss_SSIM
        
class L_SSIM5(nn.Module):
    def __init__(self):
        super(L_SSIM5, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_C, image_D,image_E,image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_C = self.sobelconv(image_C)
        gradient_D = self.sobelconv(image_D)
        gradient_E = self.sobelconv(image_E)

        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) 
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) 
        weight_C = torch.mean(gradient_C) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) 
        weight_D = torch.mean(gradient_D) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) 
        weight_E = torch.mean(gradient_E) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) 

        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused) + weight_C * ssim(image_C, image_fused)+ weight_D * ssim(image_D, image_fused)+weight_E * ssim(image_E, image_fused)
        return Loss_SSIM
    
class L_SSIM7(nn.Module):
    def __init__(self):
        super(L_SSIM7, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_C, image_D,image_E,image_H,image_I,image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_C = self.sobelconv(image_C)
        gradient_D = self.sobelconv(image_D)
        gradient_E = self.sobelconv(image_E)
        gradient_H = self.sobelconv(image_H)
        gradient_I = self.sobelconv(image_I)

        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) +torch.mean(gradient_H)+torch.mean(gradient_I)
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) +torch.mean(gradient_H)+torch.mean(gradient_I)
        weight_C = torch.mean(gradient_C) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) +torch.mean(gradient_H)+torch.mean(gradient_I)
        weight_D = torch.mean(gradient_D) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) +torch.mean(gradient_H)+torch.mean(gradient_I)
        weight_E = torch.mean(gradient_E) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) +torch.mean(gradient_H)+torch.mean(gradient_I)

        weight_H = torch.mean(gradient_H) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) +torch.mean(gradient_H)+torch.mean(gradient_I)
        weight_I = torch.mean(gradient_I) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) +torch.mean(gradient_H)+torch.mean(gradient_I)
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused) + weight_C * ssim(image_C, image_fused)+ weight_D * ssim(image_D, image_fused)+weight_E * ssim(image_E, image_fused)+\
                    weight_H * ssim(image_H, image_fused) + weight_I * ssim(image_I, image_fused)
        return Loss_SSIM
    

    
class L_SSIM9(nn.Module):
    def __init__(self):
        super(L_SSIM9, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_C, image_D,image_E,image_H,image_I,image_F,image_G,image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_C = self.sobelconv(image_C)
        gradient_D = self.sobelconv(image_D)
        gradient_E = self.sobelconv(image_E)
        gradient_H = self.sobelconv(image_H)
        gradient_I = self.sobelconv(image_I)

        gradient_F = self.sobelconv(image_F)
        gradient_G = self.sobelconv(image_G)

        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) +torch.mean(gradient_H)+torch.mean(gradient_I)+torch.mean(gradient_F)+torch.mean(gradient_G)
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) +torch.mean(gradient_H)+torch.mean(gradient_I)+torch.mean(gradient_F)+torch.mean(gradient_G)
        weight_C = torch.mean(gradient_C) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) +torch.mean(gradient_H)+torch.mean(gradient_I)+torch.mean(gradient_F)+torch.mean(gradient_G)
        weight_D = torch.mean(gradient_D) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) +torch.mean(gradient_H)+torch.mean(gradient_I)+torch.mean(gradient_F)+torch.mean(gradient_G)
        weight_E = torch.mean(gradient_E) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) +torch.mean(gradient_H)+torch.mean(gradient_I)+torch.mean(gradient_F)+torch.mean(gradient_G)

        weight_H = torch.mean(gradient_H) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) +torch.mean(gradient_H)+torch.mean(gradient_I)+torch.mean(gradient_F)+torch.mean(gradient_G)
        weight_I = torch.mean(gradient_I) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) +torch.mean(gradient_H)+torch.mean(gradient_I)+torch.mean(gradient_F)+torch.mean(gradient_G)
       
        weight_F = torch.mean(gradient_F) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) +torch.mean(gradient_H)+torch.mean(gradient_I)+torch.mean(gradient_F)+torch.mean(gradient_G)
        weight_G = torch.mean(gradient_G) / (torch.mean(gradient_A) + torch.mean(gradient_B)) + torch.mean(gradient_C) +  torch.mean(gradient_D) +  torch.mean(gradient_E) +torch.mean(gradient_H)+torch.mean(gradient_I)+torch.mean(gradient_F)+torch.mean(gradient_G)      
       
       
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused) + weight_C * ssim(image_C, image_fused)+ weight_D * ssim(image_D, image_fused)+weight_E * ssim(image_E, image_fused)+\
                    weight_H * ssim(image_H, image_fused) + weight_I * ssim(image_I, image_fused)+weight_F * ssim(image_F, image_fused) + weight_G * ssim(image_G, image_fused)
        return Loss_SSIM
    

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_C, image_D,image_E, image_fused):        
        intensity_joint =  torch.max(torch.max(torch.max(torch.max(image_A, image_B), image_C),image_D),image_E)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity

class L_Intensity2(nn.Module):
    def __init__(self):
        super(L_Intensity2, self).__init__()

    def forward(self, image_A, image_B, image_fused): 
        # print(image_A.size()) 
        # print(image_B.size())     
        intensity_joint =  torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity
    
class L_Intensity5(nn.Module):
    def __init__(self):
        super(L_Intensity5, self).__init__()

    def forward(self, image_A, image_B, image_C, image_D,image_E,image_fused):        
        intensity_joint = torch.max(torch.max(torch.max(torch.max(image_A, image_B), image_C),image_D),image_E)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity


class L_Intensity5_avg(nn.Module):
    def __init__(self):
        super(L_Intensity5_avg, self).__init__()

    def forward(self, image_A, image_B, image_C, image_D,image_E,image_fused):        
        # intensity_joint = torch.max(torch.max(torch.max(torch.max(image_A, image_B), image_C),image_D),image_E)
        Loss_intensity = F.l1_loss(image_fused, image_A)+F.l1_loss(image_fused, image_B)+F.l1_loss(image_fused, image_C)+\
                        +F.l1_loss(image_fused, image_D)+F.l1_loss(image_fused, image_E)
        return Loss_intensity
    
class L_Intensity7(nn.Module):
    def __init__(self):
        super(L_Intensity7, self).__init__()

    def forward(self, image_A, image_B, image_C, image_D,image_E,image_F,image_G,image_fused):        
        intensity_joint = torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(image_A, image_B), image_C),image_D),image_E),image_F),image_G)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity
    
class L_Intensity9(nn.Module):
    def __init__(self):
        super(L_Intensity9, self).__init__()

    def forward(self, image_A, image_B, image_C, image_D,image_E,image_F,image_G,image_H,image_I,image_fused):        
        intensity_joint = torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(image_A, image_B), image_C),image_D),image_E),image_F),image_G),image_H),image_I)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity


class fusion_loss_vif(nn.Module):
    def __init__(self):
        super(fusion_loss_vif, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()

        # print(1)
    def forward(self, image_A, image_B, image_fused):
        loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 20 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM


class fusion_loss_vif_3(nn.Module):
    def __init__(self):
        super(fusion_loss_vif_3, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()

        # print(1)
    def forward(self, image_A, image_B, image_fused,GT,GT_2,GT_3):
        loss_l1 = 20 * self.L_Inten(GT,GT_2,GT_3, image_fused)
        loss_gradient = 20 * self.L_Grad(GT,GT_2,GT_3, image_fused)
        loss_SSIM = 10 * (1 - self.L_SSIM(GT,GT_2,GT_3, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM

class fusion_loss_vif_2(nn.Module):
    def __init__(self):
        super(fusion_loss_vif_2, self).__init__()
        self.L_Grad = L_Grad2()
        self.L_Inten = L_Intensity2()
        self.L_SSIM = L_SSIM2()

        # print(1)
    def forward(self, image_A, image_B, image_fused,GT,GT_2):
        loss_l1 = 20 * self.L_Inten(GT,GT_2, image_fused)
        loss_gradient = 20 * self.L_Grad(GT,GT_2, image_fused)
        loss_SSIM = 10 * (1 - self.L_SSIM(GT,GT_2, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM
    
class fusion_loss_vif_5(nn.Module):
    def __init__(self):
        super(fusion_loss_vif_5, self).__init__()
        self.L_Grad = L_Grad5()
        self.L_Inten = L_Intensity5()
        self.L_SSIM = L_SSIM5()

        # print(1)
    def forward(self, image_A, image_B, image_fused,GT,GT_2,GT_3,GT_4,GT_5):
        loss_l1 = 20 * self.L_Inten(GT,GT_2,GT_3, GT_4,GT_5, image_fused)
        loss_gradient = 20 * self.L_Grad(GT,GT_2,GT_3, GT_4,GT_5,image_fused)
        loss_SSIM = 10 * (1 - self.L_SSIM(GT,GT_2,GT_3, GT_4,GT_5, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM

class fusion_loss_vif_5_avg(nn.Module):
    def __init__(self):
        super(fusion_loss_vif_5_avg, self).__init__()
        self.L_Grad = L_Grad5_avg()
        self.L_Inten = L_Intensity5_avg()
        self.L_SSIM = L_SSIM5()

        # print(1)
    def forward(self, image_A, image_B, image_fused,GT,GT_2,GT_3,GT_4,GT_5):
        loss_l1 = 20 * self.L_Inten(GT,GT_2,GT_3, GT_4,GT_5, image_fused)
        loss_gradient = 20 * self.L_Grad(GT,GT_2,GT_3, GT_4,GT_5,image_fused)
        loss_SSIM = 10 * (1 - self.L_SSIM(GT,GT_2,GT_3, GT_4,GT_5, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM  

class fusion_loss_vif_7(nn.Module):
    def __init__(self):
        super(fusion_loss_vif_7, self).__init__()
        self.L_Grad = L_Grad7()
        self.L_Inten = L_Intensity7()
        self.L_SSIM = L_SSIM7()

        # print(1)
    def forward(self, image_A, image_B, image_fused,GT,GT_2,GT_3,GT_4,GT_5,GT_6,GT_7):
        loss_l1 = 20 * self.L_Inten(GT,GT_2,GT_3, GT_4,GT_5, GT_6,GT_7,image_fused)
        loss_gradient = 20 * self.L_Grad(GT,GT_2,GT_3, GT_4,GT_5,GT_6,GT_7,image_fused)
        loss_SSIM = 10 * (1 - self.L_SSIM(GT,GT_2,GT_3, GT_4,GT_5, GT_6,GT_7,image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM
    
class fusion_loss_vif_9(nn.Module):
    def __init__(self):
        super(fusion_loss_vif_9, self).__init__()
        self.L_Grad = L_Grad9()
        self.L_Inten = L_Intensity9()
        self.L_SSIM = L_SSIM9()

        # print(1)
    def forward(self, image_A, image_B, image_fused,GT,GT_2,GT_3,GT_4,GT_5,GT_6,GT_7,GT_8,GT_9):
        loss_l1 = 20 * self.L_Inten(GT,GT_2,GT_3, GT_4,GT_5, GT_6,GT_7,GT_8,GT_9,image_fused)
        loss_gradient = 20 * self.L_Grad(GT,GT_2,GT_3, GT_4,GT_5,GT_6,GT_7,GT_8,GT_9,image_fused)
        loss_SSIM = 10 * (1 - self.L_SSIM(GT,GT_2,GT_3, GT_4,GT_5, GT_6,GT_7,GT_8,GT_9,image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM