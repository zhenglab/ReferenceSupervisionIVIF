from __future__ import print_function

import time

# from utils import list_images
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
# from train_gt import train
from generate import generate
import scipy.ndimage
import re
import argparse

BATCH_SIZE = 24
EPOCHES = 10
LOGGING = 40
root_path='./'

IS_TRAINING =False

def main():

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
	parser.add_argument('-checkpoint_dir', type=str, default='')
	args = parser.parse_args()
	path ='./dataset/'
	savepath = 'output/'
	if not os.path.exists(savepath):
			os.makedirs(savepath)
	MODEL_SAVE_PATH = root_path + args.checkpoint_dir
	Time=[]
	path1 = path + 'ir/'
	path2 = path + 'vis/'
	filename=os.listdir(path1)
	for i in range(len(filename)):
		index = i + 1
		ir_path = path1 + filename[i]
		vis_path = path2 + filename[i]
		name_ir = re.split(r'[.]', filename[i])
		name_ir = name_ir[0]
		model_path = MODEL_SAVE_PATH +'.ckpt'
		generate(ir_path, vis_path, model_path, name_ir,index, output_path = savepath)
		print("pic_num:%s" % index)


if __name__ == '__main__':
	main()
