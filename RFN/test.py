# test phase
import os
import torch
from torch.autograd import Variable
from net import NestFuse_light2_nodense, Fusion_network, Fusion_strategy
import utils
from args import args
import numpy as np
import kornia

def load_model(path_auto, fs_type, flag_img):
	if flag_img is True:
		nc = 3
	else:
		nc =1
	input_nc = 1
	output_nc = 1
	nb_filter = [64, 112, 160, 208, 256]
	nest_model = NestFuse_light2_nodense(nb_filter, input_nc, output_nc, deepsupervision=False)
	nest_model.load_state_dict(torch.load(path_auto))
	fusion_strategy = Fusion_strategy(fs_type)
	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))
	nest_model.eval()
	nest_model.cuda()
	return nest_model, fusion_strategy


def run_demo(nest_model, fusion_strategy, infrared_path, visible_path, output_path_root, name_ir, fs_type, use_strategy, flag_img):
	img_ir, h, w, c = utils.get_test_image(infrared_path, flag=flag_img)  # True for rgb
	img_vi, h, w, c = utils.get_test_image(visible_path, flag=flag_img)

	if c is 1: #c:channel
		if args.cuda:
			img_ir = img_ir.cuda()
			img_vi = img_vi.cuda()
		img_ir = Variable(img_ir, requires_grad=False)
		img_vi = Variable(img_vi, requires_grad=False)
		# encoder
		en_r = nest_model.encoder(img_ir)
		en_v = nest_model.encoder(img_vi)
		# fusion net
		if use_strategy:
			f = fusion_strategy(en_r, en_v)
		else:
			f = nest_model.fusionnet(en_r, en_v)
		# decoder
		img_fusion_list = nest_model.decoder_eval(f)
	else:
		# fusion each block
		img_fusion_blocks = []
		for i in range(c):
			# encoder
			img_vi_temp = img_vi[i]
			img_ir_temp = img_ir[i]
			if args.cuda:
				img_vi_temp = img_vi_temp.cuda()
				img_ir_temp = img_ir_temp.cuda()
			img_vi_temp = Variable(img_vi_temp, requires_grad=False)
			img_ir_temp = Variable(img_ir_temp, requires_grad=False)

			en_r = nest_model.encoder(img_ir_temp)
			en_v = nest_model.encoder(img_vi_temp)
			# fusion net
			if use_strategy:
				f = fusion_strategy(en_r, en_v)
			else:
				f = nest_model.fusionnet(en_r, en_v)
			# decoder
			img_fusion_temp = nest_model.decoder_eval(f)
			img_fusion_blocks.append(img_fusion_temp)
		img_fusion_list = utils.recons_fusion_images(img_fusion_blocks, h, w)

	output_count = 0
	for img_fusion in img_fusion_list:
		file_name = name_ir
		output_path = output_path_root + file_name
		output_count += 1
		# save images
		utils.save_image_test(img_fusion, output_path)


def main():
	# False - gray
	flag_img = False
	test_path = "IVSI/"
	model_path = args.save_fusion_model+'/'+'Epoch_'+str(args.epoch) + '.model' #model path
	output = "./output" + args.name
	if os.path.exists(output) is False:
		os.mkdir(output)
	output_path_root = output+'/' + args.name
	if os.path.exists(output_path_root) is False:
		os.mkdir(output_path_root)
	fs_type = 'res'  # res (RFN), add, avg, max, spa, nuclear
	use_strategy = False  # True - static strategy; False - RFN

	with torch.no_grad():
		if os.path.exists(output_path_root) is False:
			os.mkdir(output_path_root)	
		model, fusion_strategy = load_model(model_path, fs_type, flag_img)
		imgs_paths_ir, names = utils.list_images_test(test_path)
		num = len(imgs_paths_ir)
		for i in range(num):
			name_ir = names[i]
			infrared_path = imgs_paths_ir[i]
			visible_path = infrared_path.replace('./ir/', './vis/')
			run_demo(model, fusion_strategy, infrared_path, visible_path, output_path_root, name_ir, fs_type, use_strategy, flag_img)
		print('Done......')


if __name__ == '__main__':
	main()
