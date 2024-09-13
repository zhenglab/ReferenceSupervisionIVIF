import argparse
from cgi import test
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
import time
import sys
from models.network_swinfusion1 import SwinFusion as net
from utils import utils_image as util
from data.dataloder import Dataset as D
from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='fusion', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='test_dataset/')
    parser.add_argument('--iter_number', type=str,
                        default='10000')
    parser.add_argument('--root_path', type=str, default='dataset_100/',
                        help='input test image root folder')
    parser.add_argument('--dataset', type=str, default='motivation_fig',
                        help='input test image name')
    parser.add_argument('--model_name', type=str, default='motivation_fig',
                        help='input test image name')
    parser.add_argument('--A_dir', type=str, default='IR',
                        help='input test image name')
    parser.add_argument('--B_dir', type=str, default='VI_Y',
                        help='input test image name')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--in_channel', type=int, default=1, help='3 means color image and 1 means gray image')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model_path = os.path.join(args.model_path, args.iter_number + '_G.pth')
    if os.path.exists(model_path):
        print(f'loading model from {args.model_path}')
    else:
        print('Traget model path: {} not existing!!!'.format(model_path))
        sys.exit()
    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    a_dir = os.path.join(args.root_path, args.dataset, args.A_dir)
    b_dir = os.path.join(args.root_path, args.dataset, args.B_dir)

    a1_dir = os.path.join(args.root_path, 'ME', args.A_dir)
    b1_dir = os.path.join(args.root_path, 'ME', args.B_dir)

    a2_dir = os.path.join(args.root_path, 'MF', args.A_dir)
    b2_dir = os.path.join(args.root_path, 'MF', args.B_dir)


    a3_dir = os.path.join(args.root_path, 'Me', args.A_dir)
    b3_dir = os.path.join(args.root_path, 'Me', args.B_dir)

    print(a_dir)
    os.makedirs(save_dir, exist_ok=True)
    test_set = D(a_dir, b_dir,a1_dir, b1_dir,a2_dir, b2_dir,a3_dir, b3_dir, args.in_channel)
    test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
    cout = 0
    total_time = 0.0  #
    for i, test_data in enumerate(test_loader):
        imgname = test_data['A_path'][0]
        img_a = test_data['A'].to(device)
        img_b = test_data['B'].to(device)
        img_c = test_data['C'].to(device)
        img_d = test_data['D'].to(device)
        img_e = test_data['E'].to(device)
        img_f = test_data['F'].to(device)
        img_g = test_data['G'].to(device)
        img_h = test_data['H'].to(device)


        start = time.time()
        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_a.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_a = torch.cat([img_a, torch.flip(img_a, [2])], 2)[:, :, :h_old + h_pad, :]
            img_a = torch.cat([img_a, torch.flip(img_a, [3])], 3)[:, :, :, :w_old + w_pad]
            img_b = torch.cat([img_b, torch.flip(img_b, [2])], 2)[:, :, :h_old + h_pad, :]
            img_b = torch.cat([img_b, torch.flip(img_b, [3])], 3)[:, :, :, :w_old + w_pad]

            img_c = torch.cat([img_c, torch.flip(img_c, [2])], 2)[:, :, :h_old + h_pad, :]
            img_c = torch.cat([img_c, torch.flip(img_c, [3])], 3)[:, :, :, :w_old + w_pad]
            img_d = torch.cat([img_d, torch.flip(img_d, [2])], 2)[:, :, :h_old + h_pad, :]
            img_d = torch.cat([img_d, torch.flip(img_d, [3])], 3)[:, :, :, :w_old + w_pad]


            img_e = torch.cat([img_e, torch.flip(img_e, [2])], 2)[:, :, :h_old + h_pad, :]
            img_e = torch.cat([img_e, torch.flip(img_e, [3])], 3)[:, :, :, :w_old + w_pad]
            img_f = torch.cat([img_f, torch.flip(img_f, [2])], 2)[:, :, :h_old + h_pad, :]
            img_f = torch.cat([img_f, torch.flip(img_f, [3])], 3)[:, :, :, :w_old + w_pad]


            img_g = torch.cat([img_g, torch.flip(img_g, [2])], 2)[:, :, :h_old + h_pad, :]
            img_g = torch.cat([img_g, torch.flip(img_g, [3])], 3)[:, :, :, :w_old + w_pad]
            img_h = torch.cat([img_h, torch.flip(img_h, [2])], 2)[:, :, :h_old + h_pad, :]
            img_h = torch.cat([img_h, torch.flip(img_h, [3])], 3)[:, :, :, :w_old + w_pad]

            output,_,_,_ = test(img_a, img_b, img_c, img_d, img_e, img_f, img_g, img_h, model, args, window_size)
    

def define_model(args):
    model = net(upscale=args.scale, in_chans=args.in_channel, img_size=128, window_size=8,
                img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                mlp_ratio=2, upsampler=None, resi_connection='1conv')
    param_key_g = 'params'
    model_path = os.path.join(args.model_path, args.iter_number + '_E.pth')
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    
        
    return model


def setup(args):   
    save_dir = f'SwinFusion_{args.dataset}_{args.model_name}'
    folder = os.path.join(args.root_path, args.dataset, 'A_Y')
    print('folder:', folder)
    border = 0
    window_size = 8

    return folder, save_dir, border, window_size


def get_image_pair(args, path, a_dir=None, b_dir=None):
    a_path = os.path.join(a_dir, os.path.basename(path))
    b_path = os.path.join(b_dir, os.path.basename(path))
    print("A image path:", a_path)
    assert not args.in_channel == 3 or not args.in_channel == 1, "Error in input parameters "
    img_a = util.imread_uint(a_path, args.in_channel)
    img_b = util.imread_uint(b_path, args.in_channel)
    img_a = util.uint2single(img_a)
    img_b = util.uint2single(img_b)
    return os.path.basename(path), img_a, img_b


def test(img_a, img_b, img_c, img_d, img_e, img_f, img_g, img_h, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_a, img_b)
        output1 = model(img_c, img_d)
        output2 = model(img_e, img_f)
        output3 = model(img_g, img_h)



    else:
        # test the image tile by tile
        b, c, h, w = img_a.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_a)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_a[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output,output1,output2,output3

if __name__ == '__main__':
    main()
