# -*- coding: utf-8 -*-
"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image 
# import scipy.misc
import imageio
import scipy.misc
import scipy.ndimage
import numpy as np
import tensorflow.compat.v1 as tf
import cv2

FLAGS = tf.app.flags.FLAGS

def read_data(path):
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    weight = np.array(hf.get('weight'))
    return data, weight

def preprocess(path, scale=3):
  image = imread(path, is_grayscale=True)
  image = (image-127.5 )/ 127.5 
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

  return input_

def read_txt(txt_path):
    data = []
    weight = []
    lines = []
    for line in open(txt_path, "r"): 
        lines.append(line)
    for line in lines:  
        line = line.strip('\n')
        line1,line2 = line.split(',')
        data.append(line1)
        lin2_new = np.reshape(float(line2), [1, 1, 1])
        weight.append(lin2_new)
    return data, weight

def prepare_data(sess, dataset):
  if FLAGS.is_train:
        data,weight = read_txt(dataset) 
  return data, weight

def make_data(sess, data,weight,data_dir,check_path):
  if FLAGS.is_train:
    savepath = os.path.join('.', os.path.join(check_path,data_dir,'train.h5'))
    if not os.path.exists(os.path.join('.',os.path.join(check_path,data_dir))):
        os.makedirs(os.path.join('.',os.path.join(check_path,data_dir)))
  else:
    savepath = os.path.join('.', os.path.join(check_path,data_dir,'test.h5'))
    if not os.path.exists(os.path.join('.',os.path.join(check_path,data_dir))):
        os.makedirs(os.path.join('.',os.path.join(check_path,data_dir)))
  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('weight', data=weight)

def imread(path, is_grayscale=True):
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='L').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def input_setup(sess,config,data_root,data_file,check_path,index=0):
  if config.is_train:
    data, weight= prepare_data(sess, dataset=data_root + data_file)
  else:
    data, weight= prepare_data(sess, dataset=data_root + data_file)

  sub_input_sequence = []
  weight_patch = []
  padding = 0

  if config.is_train:
    data_file = data_file.split('.')
    path = data_root + (data_file[0])+'/'
    for i in range(len(data)):
      input_ = (imread(path + data[i])-127.5)/127.5
      weight1 = weight[i]
      if len(input_.shape) == 3:
        h, w, _ = input_.shape
      else:
        h, w = input_.shape
      for x in range(0, h-config.image_size+1, config.stride):
        for y in range(0, w-config.image_size+1, config.stride):
          sub_input = input_[x:x+config.image_size, y:y+config.image_size]   
          if data_file == "Train":
            sub_input=cv2.resize(sub_input, (config.image_size/4,config.image_size/4),interpolation=cv2.INTER_CUBIC)
            sub_input = sub_input.reshape([config.image_size/4, config.image_size/4, 1])
            print('error')
          else:
            sub_input = sub_input.reshape([config.image_size, config.image_size, 1])            
          sub_input_sequence.append(sub_input)
      weight_tmp = weight1*len(sub_input)
      weight_patch.append(weight_tmp)
  arrdata = np.asarray(sub_input_sequence) 
  arrweight = np.asarray(weight_patch)
  make_data(sess, arrdata, arrweight,data_file[0],check_path)


def imsave(image, path):
  return  imageio.imsave(path,image)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h*size[0], w*size[1], 1))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image
  return (img*127.5+127.5)
  
def gradient(input):
    filter=tf.reshape(tf.constant([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]),[3,3,1,1])
    d=tf.nn.conv2d(input,filter,strides=[1,1,1,1], padding='SAME')
    return d

def low_pass(input):
    filter=tf.reshape(tf.constant([[0.0947,0.1183,0.0947],[0.1183,0.1478,0.1183],[0.0947,0.1183,0.0947]]),[3,3,1,1])
    d=tf.nn.conv2d(input,filter,strides=[1,1,1,1], padding='SAME')
    return d

def blur_2th(input):
    filter=tf.reshape(tf.constant([[0.0947,0.1183,0.0947],[0.1183,0.1478,0.1183],[0.0947,0.1183,0.0947]]),[3,3,1,1])
    blur=tf.nn.conv2d(input,filter,strides=[1,1,1,1], padding='SAME')
    diff=tf.abs(input-blur)
    return diff

def _tf_fspecial_gauss(size, sigma):
		x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
		x_data = np.expand_dims(x_data, axis=-1)
		x_data = np.expand_dims(x_data, axis=-1)
		y_data = np.expand_dims(y_data, axis=-1)
		y_data = np.expand_dims(y_data, axis=-1)
		x = tf.constant(x_data, dtype=tf.float32)
		y = tf.constant(y_data, dtype=tf.float32)
		g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
		return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=8, sigma=1.5):
	window = _tf_fspecial_gauss(size, sigma) 
	K1 = 0.01
	K2 = 0.03
	L = 1  
	C1 = (K1*L)**2
	C2 = (K2*L)**2
	mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
	mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
	mu1_sq = mu1*mu1
	mu2_sq = mu2*mu2
	mu1_mu2 = mu1*mu2
	sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
	sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
	sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
	if cs_map:
		value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
					(sigma1_sq + sigma2_sq + C2)),
				(2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
	else:
		value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
					(sigma1_sq + sigma2_sq + C2))

	if mean_metric:
		value = tf.reduce_mean(value)
	return value
   
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)
    
def l2_norm(input_x, epsilon=1e-12):
    input_x_norm = input_x/(tf.reduce_sum(input_x**2)**0.5 + epsilon)
    return input_x_norm
