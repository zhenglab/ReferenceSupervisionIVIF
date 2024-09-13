# -*- coding: utf-8 -*-
"""
@author: Zixiang Zhao (zixiangzhao@stu.xjtu.edu.cn)

Pytorch implement for "DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion" (IJCAI 2020)

https://www.ijcai.org/Proceedings/2020/135
"""
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import scipy.io as scio
import kornia
import argparse
from skimage.io import imsave
from loss_tensor import compute_loss,structure_loss
from DIDFuse import AE_Encoder,AE_Decoder
from utils_didfuse import *

# =============================================================================
# Hyperparameters Setting 
# =============================================================================
Train_data_choose = 'IVSI'#'FLIR' & 'NIR'
if Train_data_choose == 'IVSI':
    train_data_path = './dataset'

#fusion
addition_mode = 'Sum'

#data path
root_VIS = train_data_path + 'vis/'
root_IR = train_data_path + 'ir/'
image_gt1 = train_data_path + '1/'
image_gt2 = train_data_path + '2/'
image_gt3 = train_data_path + '3/'
image_gt4 = train_data_path + '4/'
image_gt5 = train_data_path + '5/'

device = "cuda"
batch_size = 24
channel = 64
epochs = 120
lr = 1

parser = argparse.ArgumentParser(description='Train with pytorch')
parser.add_argument('-a', type=float, default=2)
parser.add_argument('-b', type=float, default=0)
parser.add_argument( '-c', type=float, default=0)
parser.add_argument( '-model_path', type=str, default='./model')
args = parser.parse_args()

if os.path.exists(args.train_path) is False:
    os.mkdir(args.train_path)

Train_Image_Number=len(os.listdir(train_data_path+'./vis'))
print(Train_Image_Number)
Iter_per_epoch=(Train_Image_Number % batch_size!=0)+Train_Image_Number//batch_size

# =============================================================================
#                Preprocessing and dataset establishment 
# =============================================================================
transforms = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        ])
                                  
Data_VIS = torchvision.datasets.ImageFolder(root_VIS,transform=transforms)
dataloader_VIS = torch.utils.data.DataLoader(Data_VIS, batch_size,shuffle=False)
Data_IR = torchvision.datasets.ImageFolder(root_IR,transform=transforms)
dataloader_IR = torch.utils.data.DataLoader(Data_IR, batch_size,shuffle=False)
Data_1 = torchvision.datasets.ImageFolder(image_gt1,transform=transforms)
dataloader_1 = torch.utils.data.DataLoader(Data_1, batch_size,shuffle=False)
Data_2 = torchvision.datasets.ImageFolder(image_gt2,transform=transforms)
dataloader_2 = torch.utils.data.DataLoader(Data_2, batch_size,shuffle=False)
Data_3 = torchvision.datasets.ImageFolder(image_gt3,transform=transforms)
dataloader_3 = torch.utils.data.DataLoader(Data_3, batch_size,shuffle=False)
Data_4 = torchvision.datasets.ImageFolder(image_gt4,transform=transforms)
dataloader_4 = torch.utils.data.DataLoader(Data_4, batch_size,shuffle=False)
Data_5 = torchvision.datasets.ImageFolder(image_gt5,transform=transforms)
dataloader_5 = torch.utils.data.DataLoader(Data_5, batch_size,shuffle=False)

# =============================================================================
#                       Models
# =============================================================================
AE_Encoder=AE_Encoder()
AE_Decoder=AE_Decoder()
is_cuda = True
if is_cuda:
    AE_Encoder=AE_Encoder.cuda()
    AE_Decoder=AE_Decoder.cuda()
 
optimizer1 = optim.Adam(AE_Encoder.parameters(), lr = lr)
optimizer2 = optim.Adam(AE_Decoder.parameters(), lr = lr)
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [epochs//3,epochs//3*2], gamma=0.1)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, [epochs//3,epochs//3*2], gamma=0.1)

MSELoss = nn.MSELoss()
SmoothL1Loss=nn.SmoothL1Loss()
L1Loss=nn.L1Loss()
ssim = kornia.losses.SSIM(11, reduction='mean')
# =============================================================================
#                          Training
# =============================================================================
print('============ Training Begins ===============')
loss_train=[]
mse_loss_B_train=[]
mse_loss_D_train=[]
mse_loss_VF_train=[]
mse_loss_IF_train=[]
Gradient_loss_train=[]
lr_list1=[]
lr_list2=[]
alpha_list=[]
sigmoid=torch.nn.Sigmoid()
for iteration in range(epochs):
    AE_Encoder.train()
    AE_Decoder.train()
    
    data_iter_VIS = iter(dataloader_VIS)
    data_iter_IR = iter(dataloader_IR)
    data_iter_1 = iter(dataloader_1)
    data_iter_2 = iter(dataloader_2)
    data_iter_3 = iter(dataloader_3)
    data_iter_4 = iter(dataloader_4)
    data_iter_5 = iter(dataloader_5)
    
    
    for step in range(Iter_per_epoch):
        data_VIS,_ = next(data_iter_VIS)
        data_IR,_  = next(data_iter_IR)
        data_1,_  = next(data_iter_1)
        data_2,_  = next(data_iter_2)
        data_3,_  = next(data_iter_3)
        data_4,_  = next(data_iter_4)
        data_5,_  = next(data_iter_5)   
          
        if is_cuda:
            data_VIS = data_VIS.cuda()
            data_IR = data_IR.cuda()
            data_1 = data_1.cuda()
            data_2 = data_2.cuda()
            data_3 = data_3.cuda()
            data_4 = data_4.cuda()
            data_5 = data_5.cuda()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        # =====================================================================
        #                   Calculate loss 
        # =====================================================================
        feature_V_1, feature_V_2, feature_V_B, feature_V_D = AE_Encoder(data_VIS)
        feature_I_1, feature_I_2, feature_I_B, feature_I_D = AE_Encoder(data_IR)

        if addition_mode == 'Sum':      
            F_b=(feature_I_B+feature_V_B)
            F_d=(feature_I_D+feature_V_D)
            F_1=(feature_I_1+feature_V_1)
            F_2=(feature_I_2+feature_V_2)
        elif addition_mode == 'Average':
            F_b=(feature_I_B+feature_V_B)/2         
            F_d=(feature_I_D+feature_V_D)/2
            F_1=(feature_I_1+feature_V_1)/2
            F_2=(feature_I_2+feature_V_2)/2
        elif addition_mode == 'l1_norm':
            F_b=l1_addition(feature_I_B,feature_V_B)
            F_d=l1_addition(feature_I_D,feature_V_D)
            F_1=l1_addition(feature_I_1,feature_V_1)
            F_2=l1_addition(feature_I_2,feature_V_2)
	
        img_recon_V = AE_Decoder(F_1,F_2,F_b,F_d)

        mse_loss_B  = L1Loss(feature_I_B, feature_V_B)
        mse_loss_D  = L1Loss(feature_I_D, feature_V_D)
        mse_loss_VF =  0.2 * (ssim(data_1, img_recon_V) + ssim(data_2, img_recon_V) + ssim(data_3, img_recon_V) + ssim(data_4, img_recon_V) + ssim(data_5, img_recon_V))+\
           0.2 * (MSELoss(data_1, img_recon_V) + MSELoss(data_2, img_recon_V) + MSELoss(data_3, img_recon_V) + MSELoss(data_4, img_recon_V) + MSELoss(data_5, img_recon_V))
                                        
        Gradient_loss = 0.2 * (L1Loss(kornia.filters.SpatialGradient()(data_1), kornia.filters.SpatialGradient()(img_recon_V)) +\
            L1Loss(kornia.filters.SpatialGradient()(data_2), kornia.filters.SpatialGradient()(img_recon_V)) +\
            L1Loss(kornia.filters.SpatialGradient()(data_3), kornia.filters.SpatialGradient()(img_recon_V)) +\
            L1Loss(kornia.filters.SpatialGradient()(data_4), kornia.filters.SpatialGradient()(img_recon_V)) +\
            L1Loss(kornia.filters.SpatialGradient()(data_5), kornia.filters.SpatialGradient()(img_recon_V)))
                                                             
        #Total loss
        loss = args.a*mse_loss_VF + args.b*torch.tanh(mse_loss_B) - args.b*torch.tanh(mse_loss_D) + args.c*Gradient_loss
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        
        t_loss = loss.item()
        los_B = mse_loss_B.item()
        los_D = mse_loss_D.item()
        los_VF = mse_loss_VF.item()
        los_G = Gradient_loss.item()
        
        print('Epoch/step: %d/%d, loss: %.7f, lr: %f' %(iteration+1, step+1, t_loss, optimizer1.state_dict()['param_groups'][0]['lr']))

        #Save Loss
        loss_train.append(loss.item())
        mse_loss_B_train.append(mse_loss_B.item())
        mse_loss_D_train.append(mse_loss_D.item())
        mse_loss_VF_train.append(mse_loss_VF.item())
        Gradient_loss_train.append(Gradient_loss.item())
    scheduler1.step()
    scheduler2.step()
    lr_list1.append(optimizer1.state_dict()['param_groups'][0]['lr'])
    lr_list2.append(optimizer2.state_dict()['param_groups'][0]['lr'])


# Save Weights and result
torch.save( {'weight': AE_Encoder.state_dict(), 'epoch':epochs}, 
   os.path.join(args.train_path,'Encoder_weight.pkl'))
torch.save( {'weight': AE_Decoder.state_dict(), 'epoch':epochs}, 
   os.path.join(args.train_path,'Decoder_weight.pkl'))

scio.savemat(os.path.join(args.train_path, 'TrainData.mat'), 
                         {'Loss': np.array(loss_train),
                          'Base_layer_loss'  : np.array(mse_loss_B_train),
                          'Detail_layer_loss': np.array(mse_loss_D_train),
                          'V_recon_loss': np.array(mse_loss_VF_train),
                          'Gradient_loss': np.array(Gradient_loss_train),
                          })
scio.savemat(os.path.join(args.train_path, 'TrainData_plot_loss.mat'), 
                         {'loss_train': np.array(loss_train),
                          'mse_loss_B_train'  : np.array(mse_loss_B_train),
                          'mse_loss_D_train': np.array(mse_loss_D_train),
                          'mse_loss_VF_train': np.array(mse_loss_VF_train),
                          'Gradient_loss_train': np.array(Gradient_loss_train),
                          })
# plot
def Average_loss(loss):
    return [sum(loss[i*Iter_per_epoch:(i+1)*Iter_per_epoch])/Iter_per_epoch for i in range(int(len(loss)/Iter_per_epoch))]

plt.figure(figsize=[12,8])
plt.subplot(2,3,1), plt.plot(Average_loss(loss_train)), plt.title('Loss')
plt.subplot(2,3,2), plt.plot(Average_loss(mse_loss_B_train)), plt.title('Base_layer_loss')
plt.subplot(2,3,3), plt.plot(Average_loss(mse_loss_D_train)), plt.title('Detail_layer_loss')
plt.subplot(2,3,4), plt.plot(Average_loss(mse_loss_VF_train)), plt.title('V_recon_loss')
plt.subplot(2,3,5), plt.plot(Average_loss(Gradient_loss_train)), plt.title('Gradient_loss')
plt.tight_layout() 


