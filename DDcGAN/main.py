from __future__ import print_function

import time
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from generate import generate
import scipy.ndimage
import re
import argparse
import sys

BATCH_SIZE = 24
EPOCHES = 20
LOGGING = 40

MODEL_SAVE_PATH_root = './model/'
if not os.path.exists(MODEL_SAVE_PATH_root):
	os.makedirs(MODEL_SAVE_PATH_root)
IS_TRAINING =True


def main():
	if IS_TRAINING:
		print(('\nBegin to train the network ...\n'))
		parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
		parser.add_argument('-data_path', type=str, default='',
								help='number of characters to sample')
		parser.add_argument('-checkpoint_dir', type=str, default='',
								help='number of characters to sample')
		parser.add_argument('-image_size', type=int, default='',
								help='number of characters to sample')
		args = parser.parse_args()
		while len(sys.argv) > 1:
			sys.argv.pop()
   
		MODEL_SAVE_PATH = MODEL_SAVE_PATH_root + args.checkpoint_dir + '/'
		if not os.path.exists(MODEL_SAVE_PATH):
			os.makedirs(MODEL_SAVE_PATH)	

		checkpoint_module = args.checkpoint_dir
		cgan_class_name = 'train'
		try:
			imported_module = __import__(checkpoint_module)
			train = getattr(imported_module, cgan_class_name)
		except ImportError as e:
			raise ImportError(f"Failed to import module '{checkpoint_module}': {e}")
		except AttributeError as e:
			raise AttributeError(f"Module '{checkpoint_module}' does not have class '{cgan_class_name}': {e}")
  
		train(args, MODEL_SAVE_PATH, EPOCHES, BATCH_SIZE, logging_period = LOGGING)
	
	else:
		print('\nBegin to generate pictures ...\n')


if __name__ == '__main__':
	main()
