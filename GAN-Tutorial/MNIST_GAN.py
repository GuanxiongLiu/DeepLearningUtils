######## libs
import tensorflow as tf
from utils import *
import numpy as np
from keras.datasets import mnist
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt










######## class
class MNIST_GAN:
    def __init__(self):
        # network parameters
        self.IMG_ROW = 28
        self.IMG_COL = 28
        self.IMG_CHA = 1
        self.g_in_shape = [None,100]
        self.g_in_dim = 100
        self.g_out_dim = 28*28*1
        self.d_in_shape = [None,28*28*1]
        self.label_shape = [None,1]
        # training parameters
        self.D_lr = 0.0002
        self.D_beta = 0.5
        self.G_lr = 0.0002
        self.G_beta = 0.5
        self.train_steps = 20000
        self.g_train_steps = 1
        self.half_batch = 32
        
        
    def load_mnist(self):
        # loading data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # 255 degree to [0,1]
        X_train = (X_train.astype('float32') - 127.5) / 127.5
        X_test = (X_test.astype('float32') - 127.5) / 127.5
        # input reshape 
        X_train = X_train.reshape(-1, self.g_out_dim)
        X_test = X_test.reshape(-1, self.g_out_dim)
        # assign to variable
        self.X_train = X_train
        np.random.shuffle(self.X_train)
        
        
    def generator(self, g_in):
        # define generator
        with tf.variable_scope('Generator'):
            G_dense1 = dense_layer(g_in, 256, tf.nn.leaky_relu)
            G_dense2 = dense_layer(G_dense1, 512, tf.nn.leaky_relu)
            G_dense3 = dense_layer(G_dense2, 1024, tf.nn.leaky_relu)
            G_out = dense_layer(G_dense3, self.g_out_dim, tf.nn.tanh)
        return G_out
    
    
    def discriminator(self, img, reuse):
        # define discriminator
        with tf.variable_scope('Discriminator'):
            D_dense1 = dense_layer(img, 1024, tf.nn.leaky_relu, reuse, 'dis_dense1')
            D_drop1 = drop_layer(D_dense1, 0.3)
            D_dense2 = dense_layer(D_drop1, 512, tf.nn.leaky_relu, reuse, 'dis_dense2')
            D_drop2 = drop_layer(D_dense2, 0.3)
            D_dense3 = dense_layer(D_drop2, 256, tf.nn.leaky_relu, reuse, 'dis_dense3')
            D_drop3 = drop_layer(D_dense3, 0.3)
            D_logit = dense_layer(D_drop3, 1, None, reuse, 'dis_logit') # the prob of real img
        return D_logit
        
    def build(self):
        # define inputs
        self.Noise = tf.placeholder(tf.float32, shape=self.g_in_shape)
        self.Image = tf.placeholder(tf.float32, shape=self.d_in_shape)
        self.N_gen_image = self.generator(self.Noise)
        self.N_dis_logit = self.discriminator(self.N_gen_image, False)
        self.I_dis_logit = self.discriminator(self.Image, True)
            
        # define loss functions
        D_loss1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.I_dis_logit), 
                                                          logits=self.I_dis_logit)
        D_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.N_dis_logit), 
                                                          logits=self.N_dis_logit)
        G_loss1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.N_dis_logit), 
                                                          logits=self.N_dis_logit)
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
        for step in range(self.train_steps):
            # get train data 1
            batch_index = step * self.half_batch % self.X_train.shape[0]
            batch_index = min(batch_index, self.X_train.shape[0] - self.half_batch)
            real = self.X_train[batch_index:(batch_index + self.half_batch)]
            fake = np.random.normal(size=(self.half_batch, self.g_in_dim))
            
            # train discriminator
            gen_img, real_pred, fake_pred, _ = sess.run(fetches=[self.N_gen_image, 
                                                                 self.N_dis_logit, 
                                                                 self.I_dis_logit, 
                                                                 self.D_train], 
                                                        feed_dict={self.Noise: fake, 
                                                                   self.Image: real})
            
            # get train data 2
            batch_index = step * self.half_batch % self.X_train.shape[0]
            batch_index = min(batch_index, self.X_train.shape[0] - self.half_batch)
            real = self.X_train[batch_index:(batch_index + self.half_batch)]
            fake = np.random.normal(size=(self.half_batch, self.g_in_dim))
            
            # train generator
            gen_img, real_pred, fake_pred, _ = sess.run(fetches=[self.N_gen_image, 
                                                                 self.N_dis_logit, 
                                                                 self.I_dis_logit, 
                                                                 self.G_train], 
                                                        feed_dict={self.Noise: fake, 
                                                                   self.Image: real})
            
            # update info and plot
            if step % 10 == 0:
                print('End of training step %d, prediction probability of real img is %f2, prediction probability of fake img is %f2' % (step, np.mean(real_pred), np.mean(fake_pred)))
                if step % 100 == 0:
                    self.visualize(gen_img, str(step))
    
    
    def visualize(self, imgs, name):
        selected = np.random.choice(imgs.shape[0], int(self.half_batch/2), replace=False)
        selected = imgs[selected]
        selected = selected.reshape(-1, self.IMG_ROW, self.IMG_COL)
        plt.figure()
        for i in range(8):
            loc = '24' + str(i+1)
            plt.subplot(loc)
            plt.imshow(selected[i], cmap='gray')
        plt.savefig('./imgs/vis_'+name)
        plt.close()

        
        
        
        
        
        
######## main
if __name__ == '__main__':
    # init
    gan = MNIST_GAN()
    # load data
    gan.load_mnist()
    # build model
    gan.build()
    # train model
    gan.train()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        