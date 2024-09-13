# -*- coding: utf-8 -*-
"""
@author: Zixiang Zhao (zixiangzhao@stu.xjtu.edu.cn)

Pytorch implement for "DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion" (IJCAI 2020)

https://www.ijcai.org/Proceedings/2020/135
"""

import numpy as np
import torch
import os
from PIL import Image
from skimage.io import imsave
from utils_didfuse import Test_fusion
import re
import argparse
# =============================================================================
# Test Details 
# =============================================================================
device = 'cuda'
addition_mode = 'l1_norm' #'Sum'&'Average'&'l1_norm'
Test_data_choose = 'IVSI' 

parser = argparse.ArgumentParser(description='Test with pytorch')
parser.add_argument( '-path', type=str, default='')
parser.add_argument( '-outpath', type=str, default='')
args = parser.parse_args()

if Test_data_choose == 'TNO':
    test_data_path = './dataset/tno/vis'
elif  Test_data_choose == 'IVSI':
    test_data_path_ir = './dataset/IVSI/vis' 
    test_data_path_vis = './dataset/IVSI/ir' 

# Determine the number of files
names = os.listdir(test_data_path_ir)
Test_Image_Number = len(os.listdir(test_data_path_ir))
# =============================================================================
# Test
# =============================================================================
for i in range(int(Test_Image_Number)):
    if Test_data_choose =='Test_data_TNO':
        Test_IR = Image.open(test_data_path+'/ir'+str(i+1)+'.bmp') # infrared image
        Test_Vis = Image.open(test_data_path+'/vis'+str(i+1)+'.bmp') # visible image
    elif Test_data_choose =='IVSI':
        Test_IR = Image.open(test_data_path_ir+names[i]) # infrared image
        Test_Vis = Image.open(test_data_path_vis+names[i]) # visible image
        name_ir = re.split(r'[.]', names[i])
        name_ir = name_ir[0]

    Fusion_image = Test_fusion(Test_IR,Test_Vis,args.path)
    if os.path.exists(args.outpath) is False:
        os.mkdir(args.outpath)
    imsave(args.outpath + name_ir + '.jpg', Fusion_image)
