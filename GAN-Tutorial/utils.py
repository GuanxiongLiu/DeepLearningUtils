import tensorflow as tf



######## predefined functions
def dense_layer(tensor_in, dim, act=None, reuse=False, name=None):
    return tf.layers.dense(inputs=tensor_in, units=dim, activation=act, reuse=reuse, name=name)

def flat_layer(tensor_in):
    return tf.layers.flatten(inputs=tensor_in)

def Adam(lr, beta):
    return tf.train.AdamOptimizer(learning_rate=lr, beta1=beta)

def drop_layer(tensor_in, prob):
    return tf.nn.dropout(x=tensor_in, keep_prob=prob)



