import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.utils import register_keras_serializable

def spatial_softmax(x):
    y = tf.exp(x - tf.reduce_max(x, axis=(1,2), keepdims=True))
    y = y / tf.reduce_sum(y, axis=(1,2), keepdims=True)
    return y

@register_keras_serializable()
class SpatialSoftmax(Layer):
    def __init__(self, **kwargs):
        super(SpatialSoftmax, self).__init__(**kwargs)
        self.scale = 1.0

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        initializer = Constant(100.0)
        self.K = self.add_weight(
            shape=(input_shape[-1],),
            initializer=initializer,
            trainable=True,
            name="K"
        )
        super(SpatialSoftmax, self).build(input_shape)

    def call(self, x):
        z = self.K * x
        return spatial_softmax(z)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(SpatialSoftmax, self).get_config()
        return config

@register_keras_serializable()
class SoftArgMaxConv(Layer):
    def __init__(self, **kwargs):
        super(SoftArgMaxConv, self).__init__(**kwargs)

    def call(self, x):
        batch_size, H, W, K = x.shape
        i = np.arange(H) / H
        j = np.arange(W) / W
        coords = np.meshgrid(i, j, indexing='ij')
        image_coords = np.stack(coords, axis=-1).astype(np.float32)
        kernel = np.tile(image_coords[:, :, np.newaxis, :], [1, 1, K, 1])
        kernel = tf.constant(kernel, dtype=tf.float32)
        y = tf.nn.depthwise_conv2d(x, kernel, strides=[1,1,1,1], padding='VALID')
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, 1, input_shape[3] * 2)

    def get_config(self):
        config = super(SoftArgMaxConv, self).get_config()
        return config
