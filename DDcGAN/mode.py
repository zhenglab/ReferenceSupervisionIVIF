from __future__ import print_function
import os
import scipy.io as scio
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import time
import scipy.ndimage

from Generator import Generator
from Discriminator import Discriminator1, Discriminator2
from LOSS import SSIM_LOSS, L1_LOSS, Fro_LOSS, _tf_fspecial_gauss
from generate import generate

from utils_weight_500 import (
  read_data, 
  input_setup
)


patch_size = 84
LEARNING_RATE = 0.0002
EPSILON = 1e-4
DECAY_RATE = 0.9
eps = 1e-8
def gaussian_blur(img, kernel_size=11, sigma=5):
    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],
                                  padding='SAME', data_format='NHWC')
def norm(label):
    label=label*255
    return label
def train(args, save_path, EPOCHES_set, BATCH_SIZE, logging_period = 1):
	from datetime import datetime
	start_time = datetime.now()
	EPOCHS = EPOCHES_set
	print('Epoches: %d, Batch_size: %d' % (EPOCHS, BATCH_SIZE))

	# create the graph
	with tf.Graph().as_default(), tf.Session() as sess:
		images_vi = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'images_vi')
		images_ir = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'images_ir')
		images_gt1 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'images_gt1')
		images_gt2 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'images_gt2')
		images_gt3 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'images_gt3')
		images_gt4 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'images_gt4')
		images_gt5 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'images_gt5')
		weight1 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, 1, 1, 1), name = 'weight1')
		weight2 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, 1, 1, 1), name = 'weight2')
		weight3 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, 1, 1, 1), name = 'weight3')
		weight4 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, 1, 1, 1), name = 'weight4')
		weight5 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, 1, 1, 1), name = 'weight5')


		G = Generator('Generator')
		generated_img = G.transform(vis = images_vi, ir = images_ir)
		print('generate:', generated_img.shape)
		D1 = Discriminator1('Discriminator1')
		grad_of_gt1 = grad(images_gt1)
		grad_of_gt2 = grad(images_gt2)
		grad_of_gt3 = grad(images_gt3)
		grad_of_gt4 = grad(images_gt4)
		grad_of_gt5 = grad(images_gt5)
		D1_real1 = D1.discrim(images_gt1, reuse = False)
		D1_real2 = D1.discrim(images_gt2, reuse = False)
		D1_real3 = D1.discrim(images_gt3, reuse = False)
		D1_real4 = D1.discrim(images_gt4, reuse = False)
		D1_real5 = D1.discrim(images_gt5, reuse = False)
		D1_fake = D1.discrim(generated_img, reuse = True)

		#######  LOSS FUNCTION
		# Loss for Generator
		G_loss_GAN_D1 = -tf.reduce_mean(tf.log(D1_fake + eps))
		G_loss_GAN = G_loss_GAN_D1
		weight11 = weight1/(weight1+weight2+weight3+weight4+weight5)
		weight21 = weight2/(weight1+weight2+weight3+weight4+weight5)
		weight31 = weight3/(weight1+weight2+weight3+weight4+weight5)
		weight41 = weight4/(weight1+weight2+weight3+weight4+weight5)
		weight51 = weight5/(weight1+weight2+weight3+weight4+weight5)
		LOSS_IR = Fro_LOSS(generated_img - images_gt1,weight11)+Fro_LOSS(generated_img - images_gt2,weight21)+\
			Fro_LOSS(generated_img - images_gt3,weight31)+Fro_LOSS(generated_img - images_gt4,weight41)+\
		Fro_LOSS(generated_img - images_gt5,weight51)
  
		LOSS_VIS = L1_LOSS(grad(generated_img) - grad_of_gt1,weight1)+\
					L1_LOSS(grad(generated_img) - grad_of_gt1,weight2)+\
					L1_LOSS(grad(generated_img) - grad_of_gt2,weight3)+\
					L1_LOSS(grad(generated_img) - grad_of_gt3,weight3)+\
					L1_LOSS(grad(generated_img) - grad_of_gt4,weight4)+\
					L1_LOSS(grad(generated_img) - grad_of_gt5,weight5)
		G_loss_norm = LOSS_IR /(16) + 1.2 * LOSS_VIS	
		G_loss = G_loss_GAN + 0.6 * G_loss_norm
		
		# Loss for Discriminator1
		D1_loss_real = (-tf.reduce_mean(tf.log(D1_real1 + eps))+\
			-tf.reduce_mean(tf.log(D1_real2 + eps))+\
			-tf.reduce_mean(tf.log(D1_real3 + eps))+\
			-tf.reduce_mean(tf.log(D1_real4 + eps))+\
			-tf.reduce_mean(tf.log(D1_real5 + eps)))/5

		D1_loss_fake = -tf.reduce_mean(tf.log(1. - D1_fake + eps))
		D1_loss = D1_loss_fake + D1_loss_real

		current_iter = tf.Variable(0)
		n_batches1=50
		learning_rate = tf.train.exponential_decay(learning_rate = LEARNING_RATE, global_step = current_iter,
		                                           decay_steps = int(n_batches1), decay_rate = DECAY_RATE,
		                                           staircase = False)
		theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Generator')
		theta_D1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator1')

		G_GAN_solver = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss_GAN, global_step = current_iter,
		                                                                 var_list = theta_G)
		G_solver = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss, global_step = current_iter,
		                                                             var_list = theta_G)
		D1_solver = tf.train.GradientDescentOptimizer(learning_rate).minimize(D1_loss, global_step = current_iter,
		                                                                      var_list = theta_D1)
		clip_G = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_G]
		clip_D1 = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_D1]
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(max_to_keep = 500)

		tf.summary.scalar('Learning rate', learning_rate)
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("logs/", sess.graph)

		# ** Start Training **
		step = 0
		count_loss = 0

		# num_imgs = source_imgs.shape[0]
		input_setup(args,args.data_path,"ir.txt",args.checkpoint_dir)
		input_setup(args,args.data_path,"vis.txt",args.checkpoint_dir)
		input_setup(args,args.data_path,"1.txt",args.checkpoint_dir)
		input_setup(args,args.data_path,"2.txt",args.checkpoint_dir)
		input_setup(args,args.data_path,"3.txt",args.checkpoint_dir)
		input_setup(args,args.data_path,"4.txt",args.checkpoint_dir)
		input_setup(args,args.data_path,"5.txt",args.checkpoint_dir)   
		data_dir_ir = os.path.join('{}'.format(args.checkpoint_dir), "ir","train.h5")
		data_dir_vi = os.path.join('{}'.format(args.checkpoint_dir), "vis","train.h5")
		data_dir_gt1 = os.path.join('{}'.format(args.checkpoint_dir), "1","train.h5")
		data_dir_gt2 = os.path.join('{}'.format(args.checkpoint_dir), "2","train.h5")
		data_dir_gt3 = os.path.join('{}'.format(args.checkpoint_dir), "3","train.h5")
		data_dir_gt4 = os.path.join('{}'.format(args.checkpoint_dir), "4","train.h5")
		data_dir_gt5 = os.path.join('{}'.format(args.checkpoint_dir), "5","train.h5")

		train_data_ir,_ = read_data(data_dir_ir)
		train_data_vi,_ = read_data(data_dir_vi)
		train_data_gt1, weight11 = read_data(data_dir_gt1)
		train_data_gt2, weight21 = read_data(data_dir_gt2)
		train_data_gt3, weight31 = read_data(data_dir_gt3)
		train_data_gt4, weight41 = read_data(data_dir_gt4)
		train_data_gt5, weight51 = read_data(data_dir_gt5)

		counter = 0
		num_imgs = len(train_data_gt4)
		print(num_imgs)
		mod = num_imgs % BATCH_SIZE
		n_batches = int(num_imgs // BATCH_SIZE)
		print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))
		for epoch in range(EPOCHS):
			for batch in range(0, n_batches):
				step += 1
				current_iter = step
				batch_images_ir = train_data_ir[batch*BATCH_SIZE : (batch+1)*BATCH_SIZE]
				batch_images_vi = train_data_vi[batch*BATCH_SIZE : (batch+1)*BATCH_SIZE]
				batch_images_gt1 = train_data_gt1[batch*BATCH_SIZE : (batch+1)*BATCH_SIZE]
				batch_images_gt2 = train_data_gt2[batch*BATCH_SIZE : (batch+1)*BATCH_SIZE]
				batch_images_gt3 = train_data_gt3[batch*BATCH_SIZE : (batch+1)*BATCH_SIZE]
				batch_images_gt4 = train_data_gt4[batch*BATCH_SIZE : (batch+1)*BATCH_SIZE]
				batch_images_gt5 = train_data_gt5[batch*BATCH_SIZE : (batch+1)*BATCH_SIZE]
				batch_weight1 = weight11[batch*BATCH_SIZE : (batch+1)*BATCH_SIZE]
				batch_weight2 = weight21[batch*BATCH_SIZE : (batch+1)*BATCH_SIZE]
				batch_weight3 = weight31[batch*BATCH_SIZE : (batch+1)*BATCH_SIZE]
				batch_weight4 = weight41[batch*BATCH_SIZE : (batch+1)*BATCH_SIZE]
				batch_weight5 = weight51[batch*BATCH_SIZE : (batch+1)*BATCH_SIZE]
				counter += 1

				FEED_DICT = {images_ir: batch_images_ir, images_vi: batch_images_vi, images_gt1: batch_images_gt1,images_gt2: batch_images_gt2,images_gt3: batch_images_gt3,  \
						images_gt4: batch_images_gt4,images_gt5: batch_images_gt5,weight1:batch_weight1,weight2:batch_weight2,weight3:batch_weight3,weight4:batch_weight4,weight5:batch_weight5}

				it_g = 0
				it_d1 = 0
				it_d2 = 0
				# run the training step
				if batch % 2==0:
					sess.run([D1_solver, clip_D1], feed_dict = FEED_DICT)
					it_d1 += 1
				else:
					sess.run([G_solver, clip_G], feed_dict = FEED_DICT)
					it_g += 1
				g_loss, d1_loss = sess.run([G_loss, D1_loss], feed_dict = FEED_DICT)

				if batch%2==0:
					while d1_loss > 1.7 and it_d1 < 20:
						sess.run([D1_solver, clip_D1], feed_dict = FEED_DICT)
						d1_loss = sess.run(D1_loss, feed_dict = FEED_DICT)
						it_d1 += 1
				else:
					while (d1_loss < 1.4 ) and it_g < 20:
						sess.run([G_GAN_solver, clip_G], feed_dict = FEED_DICT)
						g_loss, d1_loss = sess.run([G_loss, D1_loss], feed_dict = FEED_DICT)
						it_g += 1
					while (g_loss > 200) and it_g < 20:
						sess.run([G_solver, clip_G], feed_dict = FEED_DICT)
						g_loss = sess.run(G_loss, feed_dict = FEED_DICT)
						it_g += 1
				print("epoch: %d/%d, batch: %d\n" % (epoch + 1, EPOCHS, batch))

				if batch % 10 == 0:
					elapsed_time = datetime.now() - start_time
					lr = sess.run(learning_rate)
					print('G_loss: %s, D1_loss: %s' % (
						g_loss, d1_loss))
					print("lr: %s, elapsed_time: %s\n" % (lr, elapsed_time))

				result = sess.run(merged, feed_dict=FEED_DICT)
				writer.add_summary(result, step)
				if step % logging_period == 0:
					saver.save(sess, save_path + str(step) + '/' + str(step) + '.ckpt')

				is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)
				if is_last_step or step % logging_period == 0:
					elapsed_time = datetime.now() - start_time
					lr = sess.run(learning_rate)
					print('epoch:%d/%d, step:%d, lr:%s, elapsed_time:%s' % (
						epoch + 1, EPOCHS, step, lr, elapsed_time))
		writer.close()
		saver.save(sess, save_path + str(epoch) + '/' + str(epoch) + '.ckpt')


def grad(img):
	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	g = tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
	return g
