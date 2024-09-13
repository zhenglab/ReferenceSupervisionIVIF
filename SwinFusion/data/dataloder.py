import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


class Dataset(data.Dataset): 
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """
    def __init__(self, root_A, root_B,root_A1, root_B1,root_A2, root_B2, root_A3, root_B3, in_channels):
        super(Dataset, self).__init__()
        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_A = util.get_image_paths(root_A)
        self.paths_B = util.get_image_paths(root_B)
        self.paths_A1 = util.get_image_paths(root_A1)
        self.paths_B1 = util.get_image_paths(root_B1)
        self.paths_A2 = util.get_image_paths(root_A2)
        self.paths_B2 = util.get_image_paths(root_B2)
        self.paths_A3 = util.get_image_paths(root_A3)
        self.paths_B3 = util.get_image_paths(root_B3)


        self.inchannels = in_channels

    def __getitem__(self, index):

        # ------------------------------------
        # get under-exposure image, over-exposure image
        # and norm-exposure image
        # ------------------------------------
        # print('input channels:', self.n_channels)
        A_path = self.paths_A[index]
        B_path = self.paths_B[index]
        A_path1 = self.paths_A1[index]
        B_path1 = self.paths_B1[index]
        A_path2 = self.paths_A2[index]
        B_path2 = self.paths_B2[index]
        A_path3 = self.paths_A3[index]
        B_path3 = self.paths_B3[index]


        img_A = util.imread_uint(A_path, self.inchannels)
        img_B = util.imread_uint(B_path, self.inchannels)
        img_A1 = util.imread_uint(A_path1, self.inchannels)
        img_B1 = util.imread_uint(B_path1, self.inchannels)
 

        img_A2 = util.imread_uint(A_path2, self.inchannels)
        img_B2 = util.imread_uint(B_path2, self.inchannels)
        img_A3 = util.imread_uint(A_path3, self.inchannels)
        img_B3 = util.imread_uint(B_path3, self.inchannels)

        """
        # --------------------------------
        # get testing image pairs
        # --------------------------------
        """
        img_A =  util.single2tensor3(util.uint2single(img_A))
        img_B =  util.single2tensor3(util.uint2single(img_B))
        img_A1 =  util.single2tensor3(util.uint2single(img_A1))
        img_B1 =  util.single2tensor3(util.uint2single(img_B1))

        img_A2 =  util.single2tensor3(util.uint2single(img_A2))
        img_B2 =  util.single2tensor3(util.uint2single(img_B2))
        img_A3 =  util.single2tensor3(util.uint2single(img_A3))
        img_B3 =  util.single2tensor3(util.uint2single(img_B3))


        # --------------------------------
        # HWC to CHW, numpy to tensor
        # --------------------------------
        # img_A = util.single2tensor3(img_A)
        # img_B = util.single2tensor3(img_B)

        return {'A': img_A, 'B': img_B, 'C': img_A1, 'D': img_B2, 'E': img_A2, 'F': img_B2, 'G': img_A3, 'H': img_B3, 'A_path': A_path, 'B_path': B_path}

    def __len__(self):
        return len(self.paths_A)
