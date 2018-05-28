import tensorflow as tf



######## activation
def relu_layer(**kwargs):
    return tf.nn.relu(**kwargs)

def lrelu_layer(**kwargs):
    return tf.nn.leaky_relu(**kwargs)

def tanh_layer(**kwargs):
    return tf.nn.tanh(**kwargs)




######## optimizer
def Adam(**kwargs):
    return tf.train.AdamOptimizer(**kwargs)