import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


class DatasetMEF(data.Dataset): 
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """
    def __init__(self, opt):
        super(DatasetMEF, self).__init__()
        print('Dataset: MEF for Multi-exposure Image Fusion.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_A = util.get_image_paths(opt['dataroot_A'])
        self.paths_B = util.get_image_paths(opt['dataroot_B'])
        self.paths_GT = util.get_image_paths(opt['dataroot_GT'])

        self.paths_GT_2 = util.get_image_paths(opt['dataroot_GT'].replace("1_enhance", "2_enhance"))
        self.paths_GT_3 = util.get_image_paths(opt['dataroot_GT'].replace("1_enhance", "3_enhance"))
        self.paths_GT_4 = util.get_image_paths(opt['dataroot_GT'].replace("1_enhance", "4_enhance"))
        self.paths_GT_5 = util.get_image_paths(opt['dataroot_GT'].replace("1_enhance", "5_enhance"))

    def __getitem__(self, index):

        # ------------------------------------
        # get under-exposure image, over-exposure image
        # and norm-exposure image
        # ------------------------------------
        A_path = self.paths_A[index]
        B_path = self.paths_B[index]
        GT_path = self.paths_GT[index]
        GT_path_2 = self.paths_GT_2[index]
        GT_path_3 = self.paths_GT_3[index]
        GT_path_4 = self.paths_GT_4[index]
        GT_path_5 = self.paths_GT_5[index]  
        img_A = util.imread_uint(A_path, self.n_channels)
        img_B = util.imread_uint(B_path, self.n_channels)
        img_GT = util.imread_uint(GT_path, self.n_channels)

        img_GT_2 = util.imread_uint(GT_path_2, self.n_channels)
        img_GT_3 = util.imread_uint(GT_path_3, self.n_channels)

        img_GT_4 = util.imread_uint(GT_path_4, self.n_channels)
        img_GT_5 = util.imread_uint(GT_path_5, self.n_channels) 

        if self.opt['phase'] == 'train': 
            """
            # --------------------------------
            # get under/over/norm patch pairs
            # --------------------------------
            """
            H, W, _ = img_A.shape
            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_A = img_A[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,:]
            patch_B = img_B[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,:]
            patch_GT = img_GT[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,:]
            patch_GT_2 = img_GT_2[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,:]
            patch_GT_3 = img_GT_3[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,:]
            patch_GT_4 = img_GT_4[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,:]
            patch_GT_5 = img_GT_5[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,:]
            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0,7)
            
            patch_A, patch_B, patch_GT = util.augment_img(patch_A, mode=mode), util.augment_img(patch_B, mode=mode), util.augment_img(patch_GT, mode=mode)
            patch_GT_2 = util.augment_img(patch_GT_2, mode=mode)
            patch_GT_3 = util.augment_img(patch_GT_3, mode=mode)
            patch_GT_4 = util.augment_img(patch_GT_4, mode=mode)
            patch_GT_5 = util.augment_img(patch_GT_5, mode=mode)
            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_A = util.uint2tensor3(patch_A)
            img_B = util.uint2tensor3(patch_B)
            img_GT = util.uint2tensor3(patch_GT)
            img_GT_2 = util.uint2tensor3(patch_GT_2)
            img_GT_3 = util.uint2tensor3(patch_GT_3)
            img_GT_4 = util.uint2tensor3(patch_GT_4)
            img_GT_5 = util.uint2tensor3(patch_GT_5)
        else: 
            """
            # --------------------------------
            # get under/over/norm image pairs
            # --------------------------------
            """
            img_A = util.uint2single(img_A)
            img_B = util.uint2single(img_B)
            img_GT = util.uint2single(img_GT)
            img_GT_2 = util.uint2single(img_GT_2)
            img_GT_3 = util.uint2single(img_GT_3)

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_A = util.single2tensor3(img_A)
            img_B = util.single2tensor3(img_B)
            img_GT = util.single2tensor3(img_GT)
            img_GT_2 = util.single2tensor3(img_GT_2)
            img_GT_3 = util.single2tensor3(img_GT_3)

        return {'A': img_A, 'B': img_B, 'GT': img_GT, 'GT_2': img_GT_2, 'GT_3': img_GT_3, 'GT_4': img_GT_4, 'GT_5': img_GT_5, 'A_path': A_path, 'B_path': B_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)
