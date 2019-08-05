import tensorflow as tf

def CAE(inputs,scope):
    '''
    [1,32,32,16,16,32,32,1]
    :param inputs: [batch_size,64,64,1]
    :return: [batch_size,64,64,1]
    '''

    with tf.variable_scope(scope+'_encoder'):
        x=tf.layers.conv2d(inputs,32,3,activation=tf.nn.relu())
        x=tf.layers.max_pooling2d(x,2)#[32,32,32]
        x=tf.layers.conv2d(x,32,3,activation=tf.nn.relu())
        x=tf.layers.max_pooling2d(x,2)#[16,16,32]
        x=tf.layers.conv2d(x,16,3,activation=tf.nn.relu())
        x=tf.layers.max_pooling2d(x,2)#[8,8,16]

    with tf.variable_scope(scope+'_decoder'):
        x=tf.layers.conv2d(x,16,3,activation=tf.nn.relu())
        x=tf.image.resize_images(x,[16,16])
        x=tf.layers.conv2d(x,32,3,activation=tf.nn.relu())
        x=tf.image.resize_images(x,[32,32])
        x=tf.layers.conv2d(x,32,3,activation=tf.nn.relu())
        x=tf.image.resize_images(x,[64,64])
        outputs=tf.layers.conv2d(x,1,3)

    return outputs

def CAE_encoder(inputs,scope):
    with tf.variable_scope(scope+'_encoder'):
        x=tf.layers.conv2d(inputs,32,3,activation=tf.nn.relu())
        x=tf.layers.max_pooling2d(x,2)#[32,32,32]
        x=tf.layers.conv2d(x,32,3,activation=tf.nn.relu())
        x=tf.layers.max_pooling2d(x,2)#[16,16,32]
        x=tf.layers.conv2d(x,16,3,activation=tf.nn.relu())
        features=tf.layers.max_pooling2d(x,2)#[8,8,16]
    return features

def pixel_wise_L2_loss(pred,gt):
    return tf.reduce_mean(tf.square(gt-pred),axis=[1,2,3])