"""TensorFlow 1.x compatibility helpers for running under TensorFlow 2.x."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
