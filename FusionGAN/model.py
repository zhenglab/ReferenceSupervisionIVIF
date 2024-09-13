# -*- coding: utf-8 -*-
from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  gradient,
  lrelu,
  weights_spectral_norm,
  l2_norm
)

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow.compat.v1 as tf
class CGAN(object):

  def __init__(self, 
               sess, 
               image_size=132,
               label_size=120,
               batch_size=32,
               c_dim=1, 
               data_path='',
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    with tf.name_scope('IR_input'):
        self.images_ir = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_ir')
        self.labels_ir = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_ir')
    with tf.name_scope('VI_input'):
        self.images_vi = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_vi')
        self.labels_vi = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_vi')
    with tf.name_scope('gt1_input'):
        self.images_gt1 = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_gt1')
        self.labels_gt1 = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_gt1')
    with tf.name_scope('gt2_input'):
        self.images_gt2 = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_gt2')
        self.labels_gt2 = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_gt2')
      
    with tf.name_scope('gt3_input'):
        self.images_gt3 = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_gt3')
        self.labels_gt3 = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_gt3')

    with tf.name_scope('gt4_input'):
        self.images_gt4 = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_gt4')
        self.labels_gt4 = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_gt4')

    with tf.name_scope('gt5_input'):
        self.images_gt5 = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_gt5')
        self.labels_gt5 = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_gt5')
    with tf.name_scope('weight1'):
        self.weight1 = tf.placeholder(tf.float32, [None,1,1,1], name='weight1')
    with tf.name_scope('weight2'):
        self.weight2 = tf.placeholder(tf.float32, [None,1,1,1], name='weight2')
    with tf.name_scope('weight3'):
        self.weight3 = tf.placeholder(tf.float32, [None,1,1,1], name='weight3')   
    with tf.name_scope('weight4'):
        self.weight4 = tf.placeholder(tf.float32, [None,1,1,1], name='weight4')
    with tf.name_scope('weight5'):
        self.weight5 = tf.placeholder(tf.float32, [None,1,1,1], name='weight5')
      
    with tf.name_scope('input'):
        self.input_image=tf.concat([self.images_ir,self.images_vi],axis=-1)

    with tf.name_scope('fusion'): 
        self.fusion_image = self.fusion_model(self.input_image)
    with tf.name_scope('d_loss'):
        pos1 = self.discriminator(self.labels_gt1,reuse=False)
        pos2 = self.discriminator(self.labels_gt2,reuse=False)
        pos3 = self.discriminator(self.labels_gt3,reuse=False)
        pos4 = self.discriminator(self.labels_gt4,reuse=False)
        pos5 = self.discriminator(self.labels_gt5,reuse=False)


        neg=self.discriminator(self.fusion_image,reuse=True,update_collection='NO_OPS')
        pos_loss=tf.reduce_mean(tf.square(pos1-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2)))+\
                tf.reduce_mean(tf.square(pos2-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2)))+\
                tf.reduce_mean(tf.square(pos3-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2)))+\
                tf.reduce_mean(tf.square(pos4-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2)))+\
                tf.reduce_mean(tf.square(pos5-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2)))
        neg_loss=tf.reduce_mean(tf.square(neg-tf.random_uniform(shape=[self.batch_size,1],minval=0,maxval=0.3,dtype=tf.float32)))

        self.d_loss=neg_loss+pos_loss
        tf.summary.scalar('loss_d',self.d_loss)
    with tf.name_scope('g_loss'):
        weight11 = self.weight1/(self.weight1+self.weight2+self.weight3+self.weight4+self.weight5)
        weight21 = self.weight2/(self.weight1+self.weight2+self.weight3+self.weight4+self.weight5)
        weight31 = self.weight3/(self.weight1+self.weight2+self.weight3+self.weight4+self.weight5)
        weight41 = self.weight4/(self.weight1+self.weight2+self.weight3+self.weight4+self.weight5)
        weight51 = self.weight5/(self.weight1+self.weight2+self.weight3+self.weight4+self.weight5)

        self.g_loss_2 = tf.reduce_mean(tf.square((self.fusion_image)-(self.labels_gt1)))\
            +tf.reduce_mean(tf.square((self.fusion_image)-(self.labels_gt2)))\
            +tf.reduce_mean(tf.square((self.fusion_image)-(self.labels_gt3)))\
            +tf.reduce_mean(tf.square((self.fusion_image)-(self.labels_gt4)))\
            +tf.reduce_mean(tf.square((self.fusion_image)-(self.labels_gt5)))\
            +5*tf.reduce_mean(tf.square(gradient((self.fusion_image)) - gradient ((self.labels_gt1))))\
            +5*tf.reduce_mean(tf.square(gradient((self.fusion_image)) - gradient ((self.labels_gt2))))\
            +5*tf.reduce_mean(tf.square(gradient((self.fusion_image)) - gradient ((self.labels_gt3))))\
            +5*tf.reduce_mean(tf.square(gradient((self.fusion_image))- gradient ((self.labels_gt4))))\
            +5*tf.reduce_mean(tf.square(gradient((self.fusion_image)) - gradient ((self.labels_gt5))))
        self.g_loss_1=tf.reduce_mean(tf.square(neg-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2,dtype=tf.float32)))
        tf.summary.scalar('g_loss_1',self.g_loss_1)
          
        tf.summary.scalar('g_loss_2',self.g_loss_2)
        self.g_loss_total=self.g_loss_1+100*self.g_loss_2
        tf.summary.scalar('loss_g',self.g_loss_total)
    self.saver = tf.train.Saver(max_to_keep=5*0)
    
  def train(self, config):
    if config.is_train:
      input_setup(self.sess, config,config.data_path,"ir.txt",config.checkpoint_dir)
      input_setup(self.sess,config,config.data_path,"vis.txt",config.checkpoint_dir)
      input_setup(self.sess,config,config.data_path,"1.txt",config.checkpoint_dir)
      input_setup(self.sess,config,config.data_path,"2.txt",config.checkpoint_dir)
      input_setup(self.sess,config,config.data_path,"3.txt",config.checkpoint_dir)
      input_setup(self.sess,config,config.data_path,"4.txt",config.checkpoint_dir)
      input_setup(self.sess,config,config.data_path,"5.txt",config.checkpoint_dir)
    else:
      nx_ir, ny_ir = input_setup(self.sess, config,"Test_ir")
      nx_vi, ny_vi = input_setup(self.sess, config,"Test_vi")

    if config.is_train:     
      data_dir_ir = os.path.join('{}'.format(config.checkpoint_dir), "ir","train.h5")
      data_dir_vi = os.path.join('{}'.format(config.checkpoint_dir), "vis","train.h5")
      data_dir_gt1 = os.path.join('{}'.format(config.checkpoint_dir), "1","train.h5")
      data_dir_gt2 = os.path.join('{}'.format(config.checkpoint_dir), "2","train.h5")
      data_dir_gt3 = os.path.join('{}'.format(config.checkpoint_dir), "3","train.h5")
      data_dir_gt4 = os.path.join('{}'.format(config.checkpoint_dir), "4","train.h5")
      data_dir_gt5 = os.path.join('{}'.format(config.checkpoint_dir), "5","train.h5")      
      
    else:
      data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir),"Test_ir", "test.h5")
      data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir),"Test_vi", "test.h5")
    train_data_ir, train_label_ir,_ = read_data(data_dir_ir)
    train_data_vi, train_label_vi,_ = read_data(data_dir_vi)
    train_data_gt1, train_label_gt1, weight1 = read_data(data_dir_gt1)
    train_data_gt2, train_label_gt2, weight2 = read_data(data_dir_gt2)
    train_data_gt3, train_label_gt3, weight3 = read_data(data_dir_gt3)
    train_data_gt4, train_label_gt4, weight4 = read_data(data_dir_gt4)
    train_data_gt5, train_label_gt5, weight5 = read_data(data_dir_gt5)

    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
    self.g_vars = [var for var in t_vars if 'fusion_model' in var.name]

    # Stochastic gradient descent with the standard backpropagation
    with tf.name_scope('train_step'):
        self.train_fusion_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total,var_list=self.g_vars)
        self.train_discriminator_op=tf.train.AdamOptimizer(config.learning_rate).minimize(self.d_loss,var_list=self.d_vars)
    #
    self.summary_op = tf.summary.merge_all()
    #
    self.train_writer = tf.summary.FileWriter(config.summary_dir + '/train',self.sess.graph,flush_secs=60)
    
    tf.initialize_all_variables().run()
    
    counter = 0
    start_time = time.time()

    if config.is_train:
      print("Training...")

      for ep in range(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data_ir) // config.batch_size
        for idx in range(0, batch_idxs):
          batch_images_ir = train_data_ir[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_ir = train_label_ir[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_vi = train_data_vi[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_vi = train_label_vi[idx*config.batch_size : (idx+1)*config.batch_size]

          batch_images_gt1 = train_data_gt1[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_gt2 = train_data_gt2[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_gt3 = train_data_gt3[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_gt4 = train_data_gt4[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_gt5 = train_data_gt5[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_gt1 = train_label_gt1[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_gt2 = train_label_gt2[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_gt3 = train_label_gt3[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_gt4 = train_label_gt4[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_gt5 = train_label_gt5[idx*config.batch_size : (idx+1)*config.batch_size]
          
          batch_weight1 = weight1[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_weight2 = weight2[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_weight3 = weight3[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_weight4 = weight4[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_weight5 = weight5[idx*config.batch_size : (idx+1)*config.batch_size]

          counter += 1
          for i in range(2):
            _, err_d = self.sess.run([self.train_discriminator_op, self.d_loss], feed_dict={self.images_ir: batch_images_ir, self.images_vi: batch_images_vi, self.labels_vi: batch_labels_vi,self.labels_ir:batch_labels_ir,self.images_gt1: batch_images_gt1,self.images_gt2: batch_images_gt2,self.labels_gt1: batch_labels_gt1,self.labels_gt2: batch_labels_gt2,\
            self.images_gt3: batch_images_gt3,  self.images_gt4: batch_images_gt4,self.images_gt5: batch_images_gt5,self.labels_gt1: batch_labels_gt1,self.labels_gt2: batch_labels_gt2,self.labels_gt3: batch_labels_gt3, self.labels_gt4: batch_labels_gt4,self.labels_gt5: batch_labels_gt5,\
            self.weight1:batch_weight1,self.weight2:batch_weight2,self.weight3:batch_weight3,self.weight4:batch_weight4,self.weight5:batch_weight5
            })
            # self.sess.run(self.clip_disc_weights)
          _, err_g,summary_str= self.sess.run([self.train_fusion_op, self.g_loss_total,self.summary_op], feed_dict={self.images_ir: batch_images_ir, self.images_vi: batch_images_vi, \
          self.labels_ir: batch_labels_ir,self.labels_vi:batch_labels_vi,self.images_gt1: batch_images_gt1,self.images_gt2: batch_images_gt2,self.images_gt3: batch_images_gt3,  \
          self.images_gt4: batch_images_gt4,self.images_gt5: batch_images_gt5,self.labels_gt1: batch_labels_gt1,self.labels_gt2: batch_labels_gt2,self.labels_gt3: batch_labels_gt3,
          self.labels_gt4: batch_labels_gt4,self.labels_gt5: batch_labels_gt5,self.weight1:batch_weight1,self.weight2:batch_weight2,self.weight3:batch_weight3,self.weight4:batch_weight4,self.weight5:batch_weight5
          })
          self.train_writer.add_summary(summary_str,counter)
        
          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_d: [%.8f],loss_g:[%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err_d,err_g))
        self.save(config.checkpoint_dir, ep)

    else:
      print("Testing...")

      result = self.fusion_image.eval(feed_dict={self.images_ir: train_data_ir, self.labels_ir: train_label_ir,self.images_vi: train_data_vi, self.labels_vi: train_label_vi})
      result=result*127.5+127.5
      result = merge(result, [nx_ir, ny_ir])
      result = result.squeeze()
      image_path = os.path.join(os.getcwd(), config.sample_dir)
      image_path = os.path.join(image_path, "test_image.png")
      imsave(result, image_path)

  def fusion_model(self,img):
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer1',reuse=tf.AUTO_REUSE):
                weights=tf.get_variable("w1",[5,5,2,256],initializer=tf.truncated_normal_initializer(stddev=1e-3))
                weights=weights_spectral_norm(weights)
                bias=tf.get_variable("b1",[256],initializer=tf.constant_initializer(0.0))
          
                conv1_ir= tf.layers.batch_normalization(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='VALID')+ bias, momentum=0.9,  epsilon=1e-5, training=True)

                conv1_ir = lrelu(conv1_ir)
        with tf.variable_scope('layer2'):
            weights=tf.get_variable("w2",[5,5,256,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b2",[128],initializer=tf.constant_initializer(0.0))
            conv2_ir= tf.layers.batch_normalization(tf.nn.conv2d(conv1_ir, weights, strides=[1,1,1,1], padding='VALID') + bias, momentum=0.9,  epsilon=1e-5, training=True)
            conv2_ir = lrelu(conv2_ir)
        with tf.variable_scope('layer3'):
            weights=tf.get_variable("w3",[3,3,128,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b3",[64],initializer=tf.constant_initializer(0.0))
            conv3_ir= tf.layers.batch_normalization(tf.nn.conv2d(conv2_ir, weights, strides=[1,1,1,1], padding='VALID') + bias, momentum=0.9, epsilon=1e-5, training=True)
            conv3_ir = lrelu(conv3_ir)
        with tf.variable_scope('layer4'):
            weights=tf.get_variable("w4",[3,3,64,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b4",[32],initializer=tf.constant_initializer(0.0))
            conv4_ir= tf.layers.batch_normalization(tf.nn.conv2d(conv3_ir, weights, strides=[1,1,1,1], padding='VALID') + bias, momentum=0.9, epsilon=1e-5, training=True)
            conv4_ir = lrelu(conv4_ir)
        with tf.variable_scope('layer5'):
            weights=tf.get_variable("w5",[1,1,32,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b5",[1],initializer=tf.constant_initializer(0.0))
            conv5_ir= tf.nn.conv2d(conv4_ir, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv5_ir=tf.nn.tanh(conv5_ir)
    return conv5_ir
    
  def discriminator(self,img,reuse,update_collection=None):
    with tf.variable_scope('discriminator',reuse=reuse):
        _,_,_,c = img.shape
        # tf.reset_default_graph()
        if c==1:
            with tf.variable_scope('layer_1'):
              weights=tf.get_variable("w_1",[3,3,1,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
              weights_1=tf.get_variable("w_11",[3,3,5,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
              weights=weights_spectral_norm(weights,update_collection=update_collection)
              bias=tf.get_variable("b_1",[32],initializer=tf.constant_initializer(0.0))
              conv1_vi=tf.nn.conv2d(img, weights, strides=[1,2,2,1], padding='VALID') + bias
              conv1_vi = lrelu(conv1_vi)
        elif c==5:
              with tf.variable_scope('layer_1',reuse=tf.AUTO_REUSE):
                weights=tf.get_variable("w_1",[3,3,1,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
                weights_1=tf.get_variable("w_11",[3,3,5,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
                weights=weights_spectral_norm(weights_1,update_collection=update_collection)
                bias=tf.get_variable("b_1",[32],initializer=tf.constant_initializer(0.0))
                conv1_vi=tf.nn.conv2d(img, weights, strides=[1,2,2,1], padding='VALID') + bias
                conv1_vi = lrelu(conv1_vi)     

        with tf.variable_scope('layer_2',reuse=tf.AUTO_REUSE):
            weights=tf.get_variable("w_2",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_2",[64],initializer=tf.constant_initializer(0.0))
            conv2_vi= tf.layers.batch_normalization(tf.nn.conv2d(conv1_vi, weights, strides=[1,2,2,1], padding='VALID') + bias, momentum=0.9, epsilon=1e-5, training=True)
            conv2_vi = lrelu(conv2_vi)
            #print(conv2_vi.shape)
        with tf.variable_scope('layer_3',reuse=tf.AUTO_REUSE):
            weights=tf.get_variable("w_3",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_3",[128],initializer=tf.constant_initializer(0.0))
            conv3_vi= tf.layers.batch_normalization(tf.nn.conv2d(conv2_vi, weights, strides=[1,2,2,1], padding='VALID') + bias, momentum=0.9, epsilon=1e-5, training=True)
            conv3_vi=lrelu(conv3_vi)
            #print(conv3_vi.shape)
        with tf.variable_scope('layer_4',reuse=tf.AUTO_REUSE):
            weights=tf.get_variable("w_4",[3,3,128,256],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_4",[256],initializer=tf.constant_initializer(0.0))
            conv4_vi= tf.layers.batch_normalization(tf.nn.conv2d(conv3_vi, weights, strides=[1,2,2,1], padding='VALID') + bias, momentum=0.9, epsilon=1e-5, training=True)
            conv4_vi=lrelu(conv4_vi)
            conv4_vi = tf.reshape(conv4_vi,[self.batch_size,6*6*256])
        with tf.variable_scope('line_5',reuse=tf.AUTO_REUSE):
            weights=tf.get_variable("w_5",[6*6*256,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("b_5",[1],initializer=tf.constant_initializer(0.0))
            line_5=tf.matmul(conv4_vi, weights) + bias
    return line_5

  def save(self, checkpoint_dir, step):
    model_name = "CGAN.model"
    model_dir = "%s_%s" % ("CGAN", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("CGAN", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
        return True
    else:
        return False
