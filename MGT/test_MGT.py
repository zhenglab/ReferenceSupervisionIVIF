import torch
import warnings
import os
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from MGT import MGT
from data import TestDataset, YCrCb2RGB
import argparse

warnings.filterwarnings("ignore")
EPSILON = 1e-5
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='', help='model name: (default: arch+timestamp)')
    parser.add_argument('--dataset_name', default='')
    parser.add_argument('--isRGB', default='false')
    args = parser.parse_args()
    return args


def main():
    # GPU/CPU
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model path
    model_path = 'MGT/' + args.model_name + '/MGT/MGT_epoch_x.pth'
    # Dataset
    Vis_RGB = False
    testset_type = args.dataset_name  # MSRS/TNO


    # load model
    model = MGT(Ex_depths=3, Fusion_depths=3, Re_depths=3)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # load dataset
    test_path = os.path.join('test_dataset', testset_type)
    print('Loading test dataset from {}.'.format(test_path))
    testset = TestDataset(test_path, Vis_RGB)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    test_tqdm = tqdm(testloader, total=len(testloader))

    # create result directory
    fused_image_save_path = os.path.join('result/', args.dataset_name, args.model_name)
    if os.path.exists(fused_image_save_path) is not True:
        os.makedirs(fused_image_save_path)

    # testing
    print('Staring testing on {}'.format(device))
    if Vis_RGB:
        for vis_y_image, vis_cb_image, vis_cr_image, inf_image, name in test_tqdm:
            _, _, H, W = vis_y_image.shape
            vis_y_image = vis_y_image.to(device)
            inf_image = inf_image.to(device)
            with torch.no_grad():
                img_fusion = model(inf_image, vis_y_image)
                img_fusion = img_fusion.cpu()
                img_fusion = YCrCb2RGB(img_fusion[0], vis_cb_image[0], vis_cr_image[0])
                img_fusion = img_fusion * 255
                out_path = fused_image_save_path + '/' + name[0]
                cv2.imwrite(out_path, img_fusion.numpy())
    else:
        for vis_image, inf_image, name in test_tqdm:
            _, _, H, W = vis_image.shape
            vis_image = vis_image.to(device)
            inf_image = inf_image.to(device)
            with torch.no_grad():
                img_fusion = model(inf_image, vis_image)
                img_fusion = img_fusion[0].cpu().numpy().transpose(1, 2, 0)
                img_fusion = img_fusion * 255
                out_path = fused_image_save_path + '/' + name[0]
                cv2.imwrite(out_path, img_fusion)

    print('Done.')


if __name__ == '__main__':
    main()
