
class args():

	# training args
	epochs = 15 
	batch_size = 4 
	dataset_ir = "./input/"
	dataset_vi = "./input/"

	HEIGHT = 256
	WIDTH = 256
	name = 'model_name'
	save_fusion_model = "./model/" + name 
	save_loss_dir = "./model/" + name 

	image_size = 256 #"size of training images, default is 256 X 256"
	cuda =1 #"set it to 1 for running on GPU, 0 for CPU"
	seed = 42 #"random seed for training"

	lr = 1e-4 #"learning rate, default is 0.001"
	log_interval = 10 #"number of images after which the training loss is logged, default is 500"
	resume_fusion_model = None
	# nest net model
	resume_nestfuse = None


