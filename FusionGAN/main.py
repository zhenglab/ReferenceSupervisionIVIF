# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.keras import backend as K
import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Number of epoch [10]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 132, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 120, "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 256, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_7_36", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("data_path", '/home/lgh/lgh/project/project/11_crocet/compute_score/dataset_20_new/dataset_5/', "dataset")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_string("summary_dir", "log", "Name of log directory [log]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
  os.environ['CUDA_VISIBLE_DEVICES']='0'
  log_device_placement=True
  allow_soft_placement=True
  tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.99
  config.gpu_options.allow_growth = True
  K.set_session(tf.Session(graph=tf.get_default_graph(),config=config))
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  
    checkpoint_module = FLAGS.checkpoint_dir
    cgan_class_name = 'CGAN'
    try:
        imported_module = __import__(checkpoint_module)
        CGAN = getattr(imported_module, cgan_class_name)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{checkpoint_module}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Module '{checkpoint_module}' does not have class '{cgan_class_name}': {e}")


  with tf.Session() as sess:
    srcnn = CGAN(sess, 
                  image_size=FLAGS.image_size, 
                  label_size=FLAGS.label_size, 
                  batch_size=FLAGS.batch_size,
                  c_dim=FLAGS.c_dim, 
                  data_path=FLAGS.data_path,
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir)

    srcnn.train(FLAGS)
    
if __name__ == '__main__':
  tf.app.run()
