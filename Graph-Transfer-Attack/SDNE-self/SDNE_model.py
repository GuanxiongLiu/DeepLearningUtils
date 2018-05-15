######## libs ########
import numpy as np
from scipy.sparse import bsr_matrix
import networkx as nx
from networkx.convert_matrix import to_numpy_array

from utils import *
import tensorflow as tf





######## classes ########
class graph_data:
    def __init__(self, raw_data):
        # read raw data to graph
        f_handle = open(raw_data, 'r')
        V, E = f_handle.readline()[:-1].split(' ')
        self.V, self.E = int(V), int(E)
        self.G = nx.Graph()
        # loop through edge list
        for l in range(self.E):
            i,j = f_handle.readline()[:-1].split(' ')
            i,j = int(i),int(j)
            self.G.add_edge(i, j)
        # convert adjacent matrix to sparse matrix
        self.adj_mat = to_numpy_array(self.G)
        # assert adj matrix
        assert self.adj_mat.shape == (self.V, self.V)
        assert self.G.number_of_edges() == self.E

        
class SDNE:
    def __init__(self, graph):
        self.in_dim, self.out_dim = graph.V, graph.V
        self.hidden1_dim, self.hidden2_dim = 500, 100
    
    def encoder(self, adj_in):
        with tf.variable_scope('Encoder'):
            E_dense1 = dense_layer(adj_in, self.hidden1_dim, reuse=True, name='enc-dense-1', 
                                   init=tf.truncated_normal_initializer(stddev=0.02), 
                                   b_init=tf.truncated_normal_initializer(stddev=0.02), 
                                   reg=tf.contrib.layers.l2_regularizer(0.001))
            E_act1 = relu_layer(E_dense1, name='enc-act-1')
            #
            E_dense2 = dense_layer(E_act1, self.hidden2_dim, reuse=True, name='enc-dense-2', 
                                   init=tf.truncated_normal_initializer(stddev=0.02), 
                                   b_init=tf.truncated_normal_initializer(stddev=0.02))
            E_out = relu_layer(E_dense2, name='enc-out')
        return E_out
    
    def decoder(self, code):
        with tf.variable_scope('Decoder'):
            D_dense1 = dense_layer(code, self.hidden1_dim, reuse=True, name='dec-dense-1', 
                                   init=tf.truncated_normal_initializer(stddev=0.02), 
                                   b_init=tf.truncated_normal_initializer(stddev=0.02))
            D_act1 = relu_layer(D_dense1, name='dec-act-1')
            #
            D_dense2 = dense_layer(D_act1, self.out_dim, reuse=True, name='dec-dense-2', 
                                   init=tf.truncated_normal_initializer(stddev=0.02), 
                                   b_init=tf.truncated_normal_initializer(stddev=0.02))
            D_out = sigmoid_layer(D_dense2, name='dec-out')
        return D_out
        


        
        
        
        

        
        
        
######## main ########
if __name__ == '__main__':
    # init
    graph = graph_data('./data/ca-Grqc.txt')