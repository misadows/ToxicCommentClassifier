import os
import time

import tensorflow as tf


class CustomModule(tf.Module):
    def __init__(self):
        super(CustomModule, self).__init__()
        self.v = tf.Variable(2.)

    @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
    def __call__(self, x):
        return x * self.v


module = CustomModule()

model_version = str(int(time.time()))
model_path = os.path.join("saved_model_basic", model_version)
tf.saved_model.save(module, model_path, signatures=module.__call__)


