import tensorflow as tf



######## parametric layers
def dense_layer(**kwargs):
    return tf.layers.dense(**kwargs)

def bnorm_layer(**kwargs):
    return tf.layers.batch_normalization(**kwargs)

def deconv2D_layer(**kwargs):
    return tf.layers.conv2d_transpose(**kwargs)

def conv2D_layer(**kwargs):
    return tf.layers.conv2d(**kwargs)



######## non-parametric layers
def flat_layer(**kwargs):
    return tf.layers.flatten(**kwargs)

def drop_layer(**kwargs):
    return tf.nn.dropout(**kwargs)

def reshape_layer(**kwargs):
    return tf.reshape(**kwargs)



