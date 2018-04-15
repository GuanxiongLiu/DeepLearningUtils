######## libs
import tensorflow as tf
from utils import *
import numpy as np
from keras.datasets import cifar10
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='Train.log',
                    filemode='w+')










######## class
class CIFAR_GAN:
    def __init__(self):
        # network parameters
        self.IMG_ROW = 32
        self.IMG_COL = 32
        self.IMG_CHA = 3
        self.g_in_dim = 100
        self.g_in_shape = [None,self.g_in_dim]
        self.d_in_shape = [None,self.IMG_ROW,self.IMG_COL,self.IMG_CHA]
        self.label_shape = [None,1]
        # training parameters
        self.D_lr = 0.0002
        self.D_beta = 0.5
        self.G_lr = 0.0002
        self.G_beta = 0.5
        self.train_steps = 100000
        self.g_train_steps = 2
        self.half_batch = 64
        
        
    def load_cifar(self):
        if not os.path.exists('./cifar_train.npy'):
            # loading data
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()
            # enlarge
            large_X = []
            for img in X_train:
                large_X.append(resize(img, (self.IMG_ROW,self.IMG_COL)))
            X_train = np.array(large_X)
            assert X_train.shape[1:] == (32,32,3)
            # 255 degree to [0,1]
            X_train = (X_train.astype('float32') * 2) - 1.
            assert np.min(X_train) < 0
            assert np.min(X_train) >= -1
            assert np.max(X_train) > 0
            assert np.max(X_train) <= 1
            # input reshape 
            X_train = X_train.reshape(-1, self.IMG_ROW,self.IMG_COL,self.IMG_CHA)
            # assign to variable
            self.X_train = X_train
            # shuffle and save
            np.random.shuffle(self.X_train)
            np.save('./cifar_train.npy', self.X_train)
        else:
            self.X_train = np.load('./cifar_train.npy')
        
        
    def generator(self, g_in):
        # define generator
        with tf.variable_scope('Generator'):
            # block1
            G_dense1 = dense_layer(tensor_in=g_in, dim=2*2*512, 
                                   init=tf.truncated_normal_initializer(stddev=0.02), 
                                   b_init=tf.constant_initializer(0.0))
            G_reshape1 = reshape_layer(tensor_in=G_dense1, out_shape=[-1,2,2,512])
            G_norm1 = bnorm_layer(tensor_in=G_reshape1, axis=-1, is_training=True)
            G_act1  = relu_layer(tensor_in=G_norm1) 
            # block2
            G_deconv2 = deconv2D_layer(tensor_in=G_act1, out_dim=256, 
                                       window=[5,5], stride=[2,2], padding='same',
                                       init=tf.truncated_normal_initializer(stddev=0.02), 
                                       b_init=tf.constant_initializer(0.0))
            G_norm2 = bnorm_layer(tensor_in=G_deconv2, axis=-1, is_training=True)
            G_act2 = relu_layer(tensor_in=G_norm2)
            # block3
            G_deconv3 = deconv2D_layer(tensor_in=G_act2, out_dim=128, 
                                       window=[5,5], stride=[2,2], padding='same',
                                       init=tf.truncated_normal_initializer(stddev=0.02), 
                                       b_init=tf.constant_initializer(0.0))
            G_norm3 = bnorm_layer(tensor_in=G_deconv3, axis=-1, is_training=True)
            G_act3 = relu_layer(tensor_in=G_norm3)
            # block4
            G_deconv4 = deconv2D_layer(tensor_in=G_act3, out_dim=64, 
                                       window=[5,5], stride=[2,2], padding='same',
                                       init=tf.truncated_normal_initializer(stddev=0.02), 
                                       b_init=tf.constant_initializer(0.0))
            G_norm4 = bnorm_layer(tensor_in=G_deconv4, axis=-1, is_training=True)
            G_act4 = relu_layer(tensor_in=G_norm4)
            # block5
            G_deconv5 = deconv2D_layer(tensor_in=G_act4, out_dim=3, 
                                       window=[5,5], stride=[2,2], padding='same',
                                       init=tf.truncated_normal_initializer(stddev=0.02), 
                                       b_init=tf.constant_initializer(0.0))
            G_out = tanh_layer(tensor_in=G_deconv5)
        return G_out
    
    
    def discriminator(self, img, reuse):
        # define discriminator
        with tf.variable_scope('Discriminator'):
            # block1
            D_conv1 = conv2D_layer(tensor_in=img, out_dim=64, window=[5,5], padding='same', 
                                   stride=[2,2], reuse=reuse, name='dis_conv1', 
                                   init=tf.truncated_normal_initializer(stddev=0.02), 
                                   b_init=tf.constant_initializer(0.0))
            D_act1 = lrelu_layer(tensor_in=D_conv1, name='dis_act1')
            # block2
            D_conv2 = conv2D_layer(tensor_in=D_act1, out_dim=128, window=[5,5], padding='same', 
                                   stride=[2,2], reuse=reuse, name='dis_conv2', 
                                   init=tf.truncated_normal_initializer(stddev=0.02), 
                                   b_init=tf.constant_initializer(0.0))
            D_norm2 = bnorm_layer(tensor_in=D_conv2, axis=-1, is_training=True, reuse=reuse, name='dis_norm2')
            D_act2 = lrelu_layer(tensor_in=D_norm2, name='dis_act2')
            # block3
            D_conv3 = conv2D_layer(tensor_in=D_act2, out_dim=256, window=[5,5], padding='same', 
                                   stride=[2,2], reuse=reuse, name='dis_conv3', 
                                   init=tf.truncated_normal_initializer(stddev=0.02), 
                                   b_init=tf.constant_initializer(0.0))
            D_norm3 = bnorm_layer(tensor_in=D_conv3, axis=-1, is_training=True, reuse=reuse, name='dis_norm3')
            D_act3 = lrelu_layer(tensor_in=D_norm3, name='dis_act3')
            # block4
            D_conv4 = conv2D_layer(tensor_in=D_act3, out_dim=512, window=[5,5], padding='same', 
                                   stride=[2,2], reuse=reuse, name='dis_conv4', 
                                   init=tf.truncated_normal_initializer(stddev=0.02), 
                                   b_init=tf.constant_initializer(0.0))
            D_norm4 = bnorm_layer(tensor_in=D_conv4, axis=-1, is_training=True, reuse=reuse, name='dis_norm4')
            D_act4 = lrelu_layer(tensor_in=D_norm4, name='dis_act4')
            # block5
            D_flat5 = flat_layer(tensor_in=D_act4, name='dis_flat5')
            D_logit = dense_layer(tensor_in=D_flat5, dim=1, reuse=reuse, 
                                  name='dis_logit', init=tf.truncated_normal_initializer(stddev=0.02), 
                                  b_init=tf.constant_initializer(0.0))
        return D_logit
        
    def build(self):
        # define inputs
        self.Noise = tf.placeholder(tf.float32, shape=self.g_in_shape)
        self.Image = tf.placeholder(tf.float32, shape=self.d_in_shape)
        self.N_gen_image = self.generator(self.Noise)
        self.N_dis_logit = self.discriminator(self.N_gen_image, False)
        self.I_dis_logit = self.discriminator(self.Image, True)
        self.N_dis_sigmoid = tf.nn.sigmoid(x=self.N_dis_logit)
        self.I_dis_sigmoid = tf.nn.sigmoid(x=self.I_dis_logit)
            
        # define loss functions
        D_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.I_dis_logit), 
                                                                         logits=self.I_dis_logit))
        D_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.N_dis_logit), 
                                                                         logits=self.N_dis_logit))
        G_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.N_dis_logit), 
                                                                         logits=self.N_dis_logit))
        self.D_loss = D_loss1 + D_loss2
        self.G_loss = G_loss1
        
        # define optimizer
        self.D_optimizier = Adam(lr=self.D_lr, beta=self.D_beta)
        self.D_train = self.D_optimizier.minimize(self.D_loss, 
                                                  var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                                                             scope='Discriminator'))
        self.G_optimizier = Adam(lr=self.G_lr, beta=self.G_beta)
        self.G_train = self.G_optimizier.minimize(self.G_loss, 
                                                  var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                                                             scope='Generator'))
        
        
    def train(self):
        # init parameters
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        # training loop
        init = 0
        for step in range(self.train_steps):
            # get train data 1
            batch_index = init * self.half_batch % self.X_train.shape[0]
            batch_index = min(batch_index, self.X_train.shape[0] - self.half_batch)
            real = self.X_train[batch_index:(batch_index + self.half_batch)]
            fake = np.random.normal(size=(self.half_batch, self.g_in_dim))
            init += 1
            
            # train discriminator
            gen_img, real_pred, fake_pred, _ = sess.run(fetches=[self.N_gen_image, 
                                                                 self.I_dis_sigmoid, 
                                                                 self.N_dis_sigmoid, 
                                                                 self.D_train], 
                                                        feed_dict={self.Noise: fake, 
                                                                   self.Image: real})
            
            # get train data 2
            for i in range(self.g_train_steps):
                batch_index = init * self.half_batch % self.X_train.shape[0]
                batch_index = min(batch_index, self.X_train.shape[0] - self.half_batch)
                real = self.X_train[batch_index:(batch_index + self.half_batch)]
                fake = np.random.normal(size=(self.half_batch, self.g_in_dim))
                init += 1

                # train generator
                gen_img, real_pred, fake_pred, _ = sess.run(fetches=[self.N_gen_image, 
                                                                     self.I_dis_sigmoid, 
                                                                     self.N_dis_sigmoid, 
                                                                     self.G_train], 
                                                            feed_dict={self.Noise: fake, 
                                                                       self.Image: real})
            
            # update info and plot
            if step % 10 == 0:
                logging.debug('End of training step %d, \
                prediction probability of real img is %f2, \
                prediction probability of fake img is %f2' % (step, np.mean(real_pred), np.mean(fake_pred)))
                if step % 1000 == 0:
                    self.visualize(gen_img, str(step))
        # final visualize
        self.visualize(gen_img, str(step+1))
    
    
    def visualize(self, imgs, name):
        selected = np.random.choice(imgs.shape[0], int(self.half_batch/2), replace=False)
        selected = imgs[selected]
        selected = (selected + 1) / 2.0
        selected = selected.reshape(-1, self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
        plt.figure()
        for i in range(8):
            loc = '24' + str(i+1)
            plt.subplot(loc)
            plt.imshow(selected[i])
        directory = './imgs/cifar_dcgan/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory+name)
        plt.close()

        
        
        
        
        
        
######## main
if __name__ == '__main__':
    # init
    gan = CIFAR_GAN()
    # load data
    gan.load_cifar()
    # build model
    gan.build()
    # train model
    gan.train()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        