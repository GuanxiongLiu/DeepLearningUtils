import numpy as np
import networkx as nx
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from keras.models import Model, load_model
from keras.layers import Input, Dense
import tensorflow as tf







class TransferModel:
    def __init__(self, config):
        self.config = config
        
    def load_data(self):
        # load embedding results
        handle = open(self.config.emb, 'r')
        node_num, enc_dim = handle.readline()[:-1].split(' ')
        node_num, enc_dim = int(node_num), int(enc_dim)
        y = np.zeros((node_num, enc_dim))
        for l in handle.readlines():
            context = l[:-1].split()
            node = int(context[0])
            enc = np.array(context)[1:].astype('float')
            y[node-1] = enc
        y = normalize(y, axis=0, norm='l1')
        assert y.shape == (node_num, enc_dim)
        # load graph and generate adjacency matrix
        G = nx.read_edgelist(self.config.graph, create_using=nx.DiGraph())
        X = nx.to_numpy_matrix(G)
        assert X.shape == (node_num, node_num)
        # store data
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, test_size=0.2)
        self.node_num = node_num
        self.enc_dim = enc_dim
        
    def build(self):
        # input
        inputs = Input(shape=(self.node_num,))
        # encoder
        enc = Dense(512, activation='sigmoid', kernel_initializer='truncated_normal')(inputs)
        enc = Dense(256, activation='sigmoid', kernel_initializer='truncated_normal')(enc)
        enc = Dense(self.enc_dim, activation='tanh', kernel_initializer='truncated_normal')(enc)
        # decoder
        dec = Dense(256, activation='sigmoid', kernel_initializer='truncated_normal')(enc)
        dec = Dense(512, activation='sigmoid', kernel_initializer='truncated_normal')(dec)
        dec = Dense(self.node_num, activation='sigmoid', kernel_initializer='truncated_normal')(dec)
        # autoencoder
        self.autoencoder = Model(inputs, [enc, dec])
        # decoder
        encoded_input = Input(shape=(self.enc_dim,))
        decoded_out = self.autoencoder.layers[-3](encoded_input)
        decoded_out = self.autoencoder.layers[-2](decoded_out)
        decoded_out = self.autoencoder.layers[-1](decoded_out)
        self.decoder = Model(encoded_input, decoded_out)
        # compile
        def recon_loss(y_true, y_pred):
            B = y_true * (self.config.beta - 1) + 1
            return tf.reduce_mean(tf.pow((y_pred - y_true)* B, 2))
        self.autoencoder.compile(optimizer='sgd', loss=['mean_absolute_error', recon_loss])
        # visualize model
        self.autoencoder.summary()
        
    def train(self):
        if os.path.exists(self.config.save_path):
            self.restore()
        else:
            # fit model
            self.autoencoder.fit(self.train_X, [self.train_y, self.train_X], epochs = self.config.epochs, 
                                 batch_size = self.config.batch_size, shuffle = True, 
                                 validation_data = (self.test_X, [self.test_y, self.test_X]))
        # save model
        self.autoencoder.save(self.config.save_path)
        
    def restore(self):
        self.autoencoder.restore(self.config.save_path)




if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="Run test.")
    parser.add_argument('--emb', nargs='?', default='../node2vec/emb/BlogCatalog.emb', 
                        help='Input embedding result path')
    parser.add_argument('--graph', nargs='?', default='../data/BlogCatalog-dataset/BlogCatalog.edges', 
                        help='Input graph path')
    parser.add_argument('--save_path', nargs='?', default='./models/ae-MAE-bMSE', 
                        help='Input saving path of autoencoder')
    parser.add_argument('--epochs', nargs='?', default=100, 
                        help='Input training epochs')
    parser.add_argument('--batch_size', nargs='?', default=64, 
                        help='Input training batch size')
    parser.add_argument('--beta', nargs='?', default=10, 
                        help='Input scaler for reconstruction loss of existing connection')
    args = parser.parse_args()
    # init model
    model = TransferModel(args)
    # test functions
    model.load_data()
    model.build()
    model.train()