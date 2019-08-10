import tensorflow as tf

def CAE(inputs,scope,bn=False,training=True):
    '''
    [1,32,32,16,16,32,32,1]
    :param inputs: [batch_size,64,64,1]
    :return: [batch_size,64,64,1]
    '''

    with tf.variable_scope(scope+'_encoder'):
        x=tf.layers.conv2d(inputs=inputs,filters=32,kernel_size=3,activation=tf.nn.relu,padding='SAME')
        if bn:
            x=tf.layers.batch_normalization(x,training=training)
        x=tf.layers.max_pooling2d(inputs=x,pool_size=2,strides=2)#[32,32,32]
        x=tf.layers.conv2d(inputs=x,filters=32,kernel_size=3,activation=tf.nn.relu,padding='SAME')
        if bn:
            x=tf.layers.batch_normalization(x,training=training)
        x=tf.layers.max_pooling2d(inputs=x,pool_size=2,strides=2)#[16,16,32]
        x=tf.layers.conv2d(inputs=x,filters=16,kernel_size=3,activation=tf.nn.relu,padding='SAME')
        if bn:
            x=tf.layers.batch_normalization(x,training=training)
        x=tf.layers.max_pooling2d(inputs=x,pool_size=2,strides=2)#[8,8,16]

    with tf.variable_scope(scope+'_decoder'):
        x=tf.layers.conv2d(inputs=x,filters=16,kernel_size=3,activation=tf.nn.relu,padding='SAME')
        if bn:
            x=tf.layers.batch_normalization(x,training=training)
        x=tf.image.resize_nearest_neighbor(x,[16,16])
        x=tf.layers.conv2d(inputs=x,filters=32,kernel_size=3,activation=tf.nn.relu,padding='SAME')
        if bn:
            x=tf.layers.batch_normalization(x,training=training)
        x=tf.image.resize_nearest_neighbor(x,[32,32])
        x=tf.layers.conv2d(inputs=x,filters=32,kernel_size=3,activation=tf.nn.relu,padding='SAME')
        if bn:
            x=tf.layers.batch_normalization(x,training=training)
        x=tf.image.resize_nearest_neighbor(x,[64,64])
        outputs=tf.layers.conv2d(inputs=x,filters=1,kernel_size=3,activation=None,padding='SAME')

    return outputs

def CAE_encoder(inputs,scope,bn=False,training=False):
    with tf.variable_scope(scope+'_encoder'):
        x=tf.layers.conv2d(inputs=inputs,filters=32,kernel_size=3,activation=tf.nn.relu,padding='SAME')
        if bn:
            x=tf.layers.batch_normalization(x,training=training)
        x=tf.layers.max_pooling2d(inputs=x,pool_size=2,strides=2)#[32,32,32]
        x=tf.layers.conv2d(inputs=x,filters=32,kernel_size=3,activation=tf.nn.relu,padding='SAME')
        if bn:
            x=tf.layers.batch_normalization(x,training=training)
        x=tf.layers.max_pooling2d(inputs=x,pool_size=2,strides=2)#[16,16,32]
        x=tf.layers.conv2d(inputs=x,filters=16,kernel_size=3,activation=tf.nn.relu,padding='SAME')
        if bn:
            x=tf.layers.batch_normalization(x,training=training)
        features=tf.layers.max_pooling2d(inputs=x,pool_size=2,strides=2)#[8,8,16]
    return features

def pixel_wise_L2_loss(pred,gt):
    return tf.reduce_mean(tf.square(gt-pred))