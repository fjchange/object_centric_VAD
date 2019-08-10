import tensorflow as tf

# CHECK
reg_weight = 0.01

class TemporalRegularityDetector(object):
  """Fully convolutional autoencoder for temporal-regularity detection.
     For simplicity, fully convolutional autoencoder structure is
     changed to be fixed as symmetric.
     Reference:
     [1] Learning Temporal Regularity in Video Sequences
         (http://arxiv.org/abs/1604.04574)
     [2] https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py
  """

  def __init__(self, sess, input_shape):
    """
    Args:
      sess : TensorFlow session
      input_shape : Shape of the input data. [n, h, w, c]
    """
    self._sess = sess
    self._input_shape = input_shape

    self._x = tf.placeholder(tf.float32, self._input_shape)

    self._var_list = []
    self._build()

    self._saver = tf.train.Saver(self._var_list)
    self._sess.run(tf.initialize_all_variables())

  def _build(self):
    conv_h1 = self._conv2d(self._x, 512, 11, 11, 4, 4, "conv_h1")
    conv_h1 = tf.nn.relu(conv_h1)
    conv_h1 = tf.contrib.layers.batch_norm(conv_h1)

    conv_h2 = self._conv2d(conv_h1, 512, 2, 2, 2, 2, "conv_h2")
    conv_h2 = tf.nn.relu(conv_h2)
    conv_h2 = tf.contrib.layers.batch_norm(conv_h2)

    conv_h3 = self._conv2d(conv_h2, 256, 5, 5, 1, 1, "conv_h3")
    conv_h3 = tf.nn.relu(conv_h3)
    conv_h3 = tf.contrib.layers.batch_norm(conv_h3)

    conv_h4 = self._conv2d(conv_h3, 256, 2, 2, 2, 2, "conv_h4")
    conv_h4 = tf.nn.relu(conv_h4)
    conv_h4 = tf.contrib.layers.batch_norm(conv_h4)

    conv_h5 = self._conv2d(conv_h4, 128, 3, 3, 1, 1, "conv_h5")
    conv_h5 = tf.nn.relu(conv_h5)
    conv_h5 = tf.contrib.layers.batch_norm(conv_h5)

    deconv_h4 = self._deconv2d(conv_h5, tf.shape(conv_h4),
                              256, 3, 3, 1, 1, "deconv_h5")
    deconv_h4 = tf.nn.relu(deconv_h4)
    deconv_h4 = tf.contrib.layers.batch_norm(deconv_h4)

    deconv_h3 = self._deconv2d(deconv_h4, tf.shape(conv_h3),
                              256, 2, 2, 2, 2, "deconv_h4")
    deconv_h3 = tf.nn.relu(deconv_h3)
    deconv_h3 = tf.contrib.layers.batch_norm(deconv_h3)

    deconv_h2 = self._deconv2d(deconv_h3, tf.shape(conv_h2),
                              512, 5, 5, 1, 1, "deconv_h3")
    deconv_h2 = tf.nn.relu(deconv_h2)
    deconv_h2 = tf.contrib.layers.batch_norm(deconv_h2)

    deconv_h1 = self._deconv2d(deconv_h2, tf.shape(conv_h1),
                              512, 2, 2, 2, 2, "deconv_h2")
    deconv_h1 = tf.nn.relu(deconv_h1)
    deconv_h1 = tf.contrib.layers.batch_norm(deconv_h1)

    self._y = self._deconv2d(deconv_h1, self._x.get_shape(),
                            10, 11, 11, 4, 4, "output")
    self._y = tf.clip_by_value(self._y, 0., 1.)

    self._reconstruct_loss = 0.5 * tf.reduce_mean(
      tf.reduce_sum(tf.square(self._x - self._y), [1,2,3]))

    self._regularize_loss = tf.constant(0., dtype=tf.float32)
    for var in self._var_list:
      if 'conv2d' in var.name:
        self._regularize_loss += tf.reduce_sum(tf.square(var))
    self._regularize_loss *= reg_weight

    self._loss = self._reconstruct_loss + self._regularize_loss
    self._train = tf.train.AdamOptimizer(1e-4).minimize(
      self._loss, var_list=self._var_list)

    self._pixel_error = tf.reduce_sum(
      tf.square(self._x-self._y), [1,2,3])
    pixel_error_max = tf.reduce_max(self._pixel_error)
    pixel_error_min = tf.reduce_min(self._pixel_error)

    self._regularity = \
      1. - (self._pixel_error - pixel_error_max) / pixel_error_min

  def save(self, ckpt_path):
    self._saver.save(self._sess, ckpt_path)

  def load(self, ckpt_path):
    self._saver.restore(self._sess, ckpt_path)

  def fit(self, input_):
    # CHECK
    _, rec_loss, reg_loss = self._sess.run(
      [self._train, self._reconstruct_loss, self._regularize_loss],
      {self._x:input_})
    print(" reconstruct_loss : {:09f}\tregularize_loss : {:09f}".format(rec_loss, reg_loss))

  def reconstruct(self, input_):
    return self._sess.run(self._y, {self._x:input_})

  def get_regularity(self, input_):
    return self._sess.run(self._regularity, {self._x:input_})

  def get_pixel_error(self, input_):
    return self._sess.run(self._pixel_error, {self._x:input_})

  def _conv2d(self, input_, output_dim,
             k_h=3, k_w=3, s_h=2, s_w=2,
             name="conv2d", stddev=0.01):
    with tf.variable_scope(name):
      k = tf.get_variable('conv2d',
        [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=tf.truncated_normal_initializer(stddev=stddev))
      conv = tf.nn.conv2d(input_, k, [1, s_h, s_w, 1], "VALID")

      b = tf.get_variable('biases', [1, 1, 1, output_dim],
        initializer=tf.constant_initializer(0.0))

      self._var_list.append(k)
      self._var_list.append(b)
    return conv + b

  def _deconv2d(self, input_, output_shape, output_dim,
                k_h=3, k_w=3, s_h=2, s_w=2,
                name="deconv2d", stddev=0.01):
    with tf.variable_scope(name):
      k = tf.get_variable('deconv2d',
        [k_h, k_w, output_dim, input_.get_shape()[-1]],
        initializer=tf.random_normal_initializer(stddev=stddev))
      deconv = tf.nn.conv2d_transpose(
        input_, k, output_shape, [1, s_h, s_w, 1], "VALID")

      b = tf.get_variable('biases', [1, 1, 1, output_dim],
        initializer=tf.constant_initializer(0.0))

      self._var_list.append(k)
      self._var_list.append(b)
    return deconv + b