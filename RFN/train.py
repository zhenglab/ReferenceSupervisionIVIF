# Training a NestFuse network
# auto-encoder

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from net import NestFuse_light2_nodense, Fusion_network
from args import args
from scipy.misc import imread, imsave, imresize
import pytorch_msssim
from loss_nir import L_TV,L_exp,USMSharp
from matplotlib import pyplot as plt
import kornia
from  loss_tensor_sum import compute_loss
import torch.nn.functional as F
from enh import enhance

EPSILON = 1e-5

def main():
	path=args.dataset_ir + 'ir'
	original_imgs_path, _ = utils.list_images_1(path)
	print(path.replace('ir', 'vis'))
	original_imgs_path_vis, _= utils.list_images_1(path.replace('ir', 'vis'))

	original_imgs_path_1, weight_1 = utils.list_images_1(path.replace('ir', '1'))
	original_imgs_path_2, weight_2 = utils.list_images_1(path.replace('ir', '2'))
	original_imgs_path_3, weight_3 = utils.list_images_1(path.replace('ir', '3'))
	original_imgs_path_4, weight_4 = utils.list_images_1(path.replace('ir', '4'))
	original_imgs_path_5, weight_5 = utils.list_images_1(path.replace('ir', '5'))


	train_num = 80000
	original_imgs_path = original_imgs_path[:train_num]
	# True - RGB , False - gray
	img_flag = False
	train(original_imgs_path,weight_1,weight_2,weight_3,weight_4,weight_5,img_flag)


def train(original_imgs_path, weight_1,weight_2,weight_3,weight_4,weight_5,img_flag):

	batch_size = args.batch_size
	# load network model
	nc = 1
	input_nc = nc
	input_nc_1=1
	output_nc = 1
	nb_filter = [64, 112, 160, 208, 256]
	f_type = 'res'
	
	deepsupervision = False
	nest_model = NestFuse_light2_nodense(nb_filter, input_nc, output_nc, deepsupervision)
	model_path = args.resume_nestfuse
	optimizer = Adam(nest_model .parameters(), args.lr)
	mse_loss = torch.nn.MSELoss()
	ssim_loss = pytorch_msssim.msssim
	denoise_loss=L_TV()
	contast_loss=L_exp()
	edge_loss=USMSharp(radius=50)

	if args.cuda:
		nest_model.cuda()
		nest_model.cuda()

	tbar = trange(args.epochs)
	print('Start training.....')

	# creating save path
	temp_path_model = os.path.join(args.save_fusion_model)
	temp_path_loss  = os.path.join(args.save_loss_dir)
	if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)

	if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)

	temp_path_model_w = args.save_fusion_model
	temp_path_loss_w  = args.save_loss_dir
	if os.path.exists(temp_path_model_w) is False:
		os.mkdir(temp_path_model_w)

	if os.path.exists(temp_path_loss_w) is False:
		os.mkdir(temp_path_loss_w)

	Loss_feature = []
	Loss_ssim = []
	Loss_all = []
	count_loss = 0
	all_ssim_loss = 0.
	all_fea_loss = 0.
	all_tidu_loss = 0.
	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
		nest_model.train()
		nest_model.cuda()
		count = 0
		gt=[]
		for batch in range(batches):
			weight1=weight_1[batch * batch_size:(batch * batch_size + batch_size)]
			weight2=weight_2[batch * batch_size:(batch * batch_size + batch_size)]
			weight3=weight_3[batch * batch_size:(batch * batch_size + batch_size)]
			weight4=weight_4[batch * batch_size:(batch * batch_size + batch_size)]
			weight5=weight_5[batch * batch_size:(batch * batch_size + batch_size)]

			image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			img_ir, _ = utils.get_train_images(image_paths_ir,weight1,height=args.HEIGHT, width=args.WIDTH, flag=img_flag)

			image_paths_vi = [x.replace('ir', 'vis') for x in image_paths_ir]
			image_paths_ir_enhance = [x.replace('ir_enhance', 'ir') for x in image_paths_ir]

			img_vi, _ = utils.get_train_images(image_paths_vi, weight1,height=args.HEIGHT, width=args.WIDTH, flag=img_flag)
			img_ir_enhance, _ = utils.get_train_images(image_paths_ir_enhance, weight1,height=args.HEIGHT, width=args.WIDTH, flag=img_flag)
			
			target_paths_first = [x.replace('ir', '1') for x in image_paths_ir]
			target_first, weight11 = utils.get_train_images(target_paths_first, weight1,height=args.HEIGHT, width=args.WIDTH, flag=img_flag)
			target_paths_second = [x.replace('ir', '2') for x in image_paths_ir]
			target_second, weight12= utils.get_train_images(target_paths_second,weight2, height=args.HEIGHT, width=args.WIDTH, flag=img_flag)

			target_paths_third = [x.replace('ir', '3') for x in image_paths_ir]
			target_third, weight13 = utils.get_train_images(target_paths_third,weight3, height=args.HEIGHT, width=args.WIDTH, flag=img_flag)

			target_paths_four = [x.replace('ir', '4') for x in image_paths_ir]
			target_four, weight14 = utils.get_train_images(target_paths_four, weight4,height=args.HEIGHT, width=args.WIDTH, flag=img_flag)

			target_paths_five = [x.replace('ir', '5') for x in image_paths_ir]
			target_five, weight15 = utils.get_train_images(target_paths_five, weight5,height=args.HEIGHT, width=args.WIDTH, flag=img_flag)
			
			count += 1
			optimizer.zero_grad()
			img_ir = Variable(img_ir)
			img_vi = Variable(img_vi)
			target_first = Variable(target_first)
			target_second = Variable(target_second)
			target_third = Variable(target_third)
			target_four = Variable(target_four)
			target_five = Variable(target_five)
   

			if args.cuda:
				img_ir = img_ir.cuda()
				img_vi = img_vi.cuda()
				target_first = target_first.cuda()
				target_second = target_second.cuda()
				target_third = target_third.cuda()
				target_four = target_four.cuda()
				target_five = target_five.cuda()
					
				weight11 = weight11/(weight11+weight12+weight13+weight14+weight15)
				weight12 = weight12/(weight11+weight12+weight13+weight14+weight15)
				weight13 = weight13/(weight11+weight12+weight13+weight14+weight15)
				weight14 = weight14/(weight11+weight12+weight13+weight14+weight15)
				weight15 = weight15/(weight11+weight12+weight13+weight14+weight15)
    
				weight11 = weight11.cuda()
				weight12 = weight12.cuda()
				weight13 = weight13.cuda()
				weight14 = weight14.cuda()
				weight15 = weight15.cuda()
			
			#encoder
			en_ir= nest_model.encoder(img_ir)
			en_vi = nest_model.encoder(img_vi)
			target_1 = nest_model.encoder(target_first)
			target_2 = nest_model.encoder(target_second)
			target_3 = nest_model.encoder(target_third)
			target_4 = nest_model.encoder(target_four)
			target_5 = nest_model.encoder(target_five)

			# fusion
			f_en = nest_model.fusionnet(en_ir, en_vi)
			# decoder
			outputs = nest_model.decoder_train(f_en)


			######################### LOSS FUNCTION #########################
			loss1_value = 0.
			loss2_value = 0.
			loss3_value = 0.
		
			for output in outputs:
				output = (output - torch.min(output)) / (torch.max(output) - torch.min(output) + EPSILON)
				output = output * 255
				
				# ---------------------- LOSS IMAGES ------------------------------------

				#ssim		
				ssim_loss_temp2 = weight11*(1-ssim_loss(output, target_first, normalize=True))+\
					weight12*(1-ssim_loss(output, target_second, normalize=True))+\
					weight13*(1-ssim_loss(output, target_third, normalize=True))+\
					weight14*(1-ssim_loss(output, target_four, normalize=True))+\
					weight15*(1-ssim_loss(output, target_five, normalize=True))										
				loss1_value +=  ssim_loss_temp2
				
				# feature loss
				g2_1_fea = target_1
				g2_2_fea = target_2
				g2_3_fea = target_3
				g2_4_fea = target_4
				g2_5_fea = target_5

				g2_fuse_fea = f_en
				w1 = weight11
				w2 = weight12
				w3 = weight13
				w4 = weight14
				w5 = weight15
				w_1 = [w1, w1, w1, w1]
				w_2 = [w2, w2, w2, w2]
				w_3 = [w3, w3, w3, w3]
				w_4 = [w4, w4, w4, w4]
				w_5 = [w5, w5, w5, w5]
				w_fea = [1, 10, 100, 1000]

				for ii in range(4):
					g2_1_temp = g2_1_fea[ii]
					g2_2_temp = g2_2_fea[ii]
					g2_3_temp = g2_3_fea[ii]
					g2_4_temp = g2_4_fea[ii]
					g2_5_temp = g2_5_fea[ii]
					
					g2_fuse_temp = g2_fuse_fea[ii]
				loss2_value += w_fea[ii]*mse_loss(g2_fuse_temp, w_1[ii]*g2_1_temp + w_2[ii]*g2_2_temp+w_3[ii]*g2_3_temp+w_4[ii]*g2_4_temp+w_5[ii]*g2_5_temp)

			loss1_value /= len(outputs)
			loss2_value /= len(outputs)
			total_loss = loss1_value + loss2_value
			total_loss.backward()
			optimizer.step()

			all_fea_loss += loss2_value.item() # 
			all_ssim_loss += loss1_value.item() # 

			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\t  W-IR: {}\tEpoch {}:\t[{}/{}]\t ssim loss: {:.6f}\t fea loss: {:.6f}\t  total: {:.6f}".format(
					time.ctime(), w1, e + 1, count, batches,
								  all_ssim_loss / args.log_interval,
								  all_fea_loss / args.log_interval,
								  (all_ssim_loss+all_fea_loss) / args.log_interval
				)
				tbar.set_description(mesg)
				Loss_ssim.append( all_ssim_loss / args.log_interval)
				Loss_feature.append(all_fea_loss / args.log_interval)
				Loss_all.append(( all_ssim_loss+all_fea_loss) / args.log_interval)
				all_ssim_loss = 0.
				all_fea_loss = 0.

			if (batch + 1) % (batches) == 0:
				# save model
				nest_model.eval()
				nest_model.cpu()

				save_model_filename = "Epoch_" + str(e) + "_iters"  + ".model"
				save_model_path = os.path.join(args.save_fusion_model, save_model_filename)
				torch.save(nest_model.state_dict(), save_model_path)
				# save loss data
				# pixel loss
				loss_data_ssim = np.array(Loss_ssim)
				loss_filename_path = temp_path_loss_w + "/loss_ssim_epoch_" + str(args.epochs) + "_iters_" + str(count)+ ".mat"
				scio.savemat(loss_filename_path, {'loss_ssim': loss_data_ssim})
				# SSIM loss
				loss_data_fea = np.array(Loss_feature)
				loss_filename_path = temp_path_loss_w + "/loss_fea_epoch_" + str(args.epochs) + "_iters_" + str(count)  + ".mat"
				scio.savemat(loss_filename_path, {'loss_fea': loss_data_fea})
				# all loss
				loss_data = np.array(Loss_all)
				loss_filename_path = temp_path_loss_w + "/loss_all_epoch_" + str(args.epochs) + "_iters_" + str(count)  + ".mat"
				scio.savemat(loss_filename_path, {'loss_all': loss_data})

				nest_model.train()
				nest_model.cuda()
				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

		plt.figure(figsize=[12,8])
		plt.subplot(1,3,1), plt.semilogy(loss_data_fea), plt.title('loss_pixel')
		plt.subplot(1,3,2), plt.semilogy(loss_data_ssim), plt.title('loss_ssim')
		plt.subplot(1,3,3), plt.semilogy(loss_data), plt.title('loss_total')
        
		curve_path = args.save_loss_dir
		plt.savefig(curve_path+'loss',dpi=90)
		print("\nDone, trained model saved at", save_model_path)


if __name__ == "__main__":
	main()
