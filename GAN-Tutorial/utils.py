import tensorflow as tf



######## parametric layers
def dense_layer(tensor_in, dim, act=None, reuse=None, name=None, init=None, b_init=None):
    return tf.layers.dense(inputs=tensor_in, units=dim, 
                           activation=act, reuse=reuse, name=name, 
                           kernel_initializer=init, bias_initializer=b_init)

def bnorm_layer(tensor_in, axis=-1, reuse=None, is_training=False, name=None, epsilon = 1e-5, momentum = 0.9):
    return tf.layers.batch_normalization(inputs=tensor_in, axis=axis, reuse=reuse, 
                                         training=is_training, epsilon=epsilon, momentum=momentum, 
                                         name=name)

def deconv2D_layer(tensor_in, out_dim, window, stride=[1,1], padding='valid', reuse=None, name=None, init=None, b_init=None):
    return tf.layers.conv2d_transpose(inputs=tensor_in, filters=out_dim, kernel_size=window, 
                                      strides=stride, reuse=reuse, name=name, padding=padding,
                                      kernel_initializer=init, bias_initializer=b_init)

def conv2D_layer(tensor_in, out_dim, window, stride=[1,1], padding='valid', reuse=None, name=None, init=None, b_init=None):
    return tf.layers.conv2d(inputs=tensor_in, filters=out_dim, kernel_size=window, 
                            strides=stride, reuse=reuse, name=name, padding=padding,
                            kernel_initializer=init, bias_initializer=b_init)



######## non-parametric layers
def flat_layer(tensor_in, name=None):
    return tf.layers.flatten(inputs=tensor_in, name=name)

def drop_layer(tensor_in, prob, name=None):
    return tf.nn.dropout(x=tensor_in, keep_prob=prob, name=name)

def reshape_layer(tensor_in, out_shape, name=None):
    return tf.reshape(tensor=tensor_in, shape=out_shape, name=name)




######## activation layers
def relu_layer(tensor_in, name=None):
    return tf.nn.relu(features=tensor_in, name=name)

def lrelu_layer(tensor_in, name=None):
    return tf.nn.leaky_relu(features=tensor_in, name=name)

def tanh_layer(tensor_in, name=None):
    return tf.nn.tanh(x=tensor_in, name=name)




######## optimizer
def Adam(lr, beta):
    return tf.train.AdamOptimizer(learning_rate=lr, beta1=beta)



