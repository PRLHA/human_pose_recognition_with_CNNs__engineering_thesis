import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="hrnet_backbone")
class HRNetBackboneLayer(layers.Layer):
    def __init__(self, module_path, **kwargs):
        super().__init__(**kwargs)
        self.module_path = module_path
        self.module = None
        self._out_hw = None          # to store (H, W)
        self._out_channels = None    # to store C

    def build(self, input_shape):
        # 1) load module
        self.module = tf.saved_model.load(self.module_path)

        # 2) run a dummy pass to get static shape info
        dummy = tf.zeros([1,
                           input_shape[1] or 512,
                           input_shape[2] or 512,
                           input_shape[3] or 3], dtype=tf.float32)
        out = self.module(dummy)
        # If it’s a dict, grab the first tensor
        tensor = list(out.values())[0] if isinstance(out, dict) else out
        shape_list = tensor.shape.as_list()  # [1, H, W, C]
        _, H, W, C = shape_list

        # 3) store them
        self._out_hw = (H, W)
        self._out_channels = C

        super().build(input_shape)

    def call(self, inputs):
        out = self.module(inputs)
        tensor = list(out.values())[0] if isinstance(out, dict) else out

        # force the static shape: [batch, H, W, C]
        return tf.ensure_shape(
            tensor,
            [None, self._out_hw[0], self._out_hw[1], self._out_channels]
        )

    def compute_output_shape(self, input_shape):
        batch = input_shape[0]
        H, W = self._out_hw
        C = self._out_channels
        return (batch, H, W, C)

    def get_config(self):
        config = super().get_config()
        config.update({"module_path": self.module_path})
        return config

    @classmethod
    def from_config(cls, config):
        path = config.pop("module_path")
        return cls(module_path=path, **config)

