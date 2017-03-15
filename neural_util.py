"""
This file contains functions for tensorflow neural networks in general.
"""
from operator import mul

import numpy as np
import tensorflow as tf
from typing import Union, Tuple, List, Dict
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import math_ops


def get_tensor_num_elements(tensor):
    # type: (tf.Tensor) -> int
    tensor_shape = map(lambda i: i.value, tensor.get_shape())
    return reduce(mul, tensor_shape, 1)


def decode_image(contents, channels=None, name=None):
    """Convenience function for `decode_gif`, `decode_jpeg`, and `decode_png`.
    Detects whether an image is a GIF, JPEG, or PNG, and performs the appropriate
    operation to convert the input bytes `string` into a `Tensor` of type `uint8`.
    Note: `decode_gif` returns a 4-D array `[num_frames, height, width, 3]`, as
    opposed to `decode_jpeg` and `decode_png`, which return 3-D arrays
    `[height, width, num_channels]`. Make sure to take this into account when
    constructing your graph if you are intermixing GIF files with JPEG and/or PNG
    files.
    Args:
      contents: 0-D `string`. The encoded image bytes.
      channels: An optional `int`. Defaults to `0`. Number of color channels for
        the decoded image.
      name: A name for the operation (optional)
    Returns:
      `Tensor` with type `uint8` with shape `[height, width, num_channels]` for
        JPEG and PNG images and shape `[num_frames, height, width, 3]` for GIF
        images.
    """
    with ops.name_scope(name, 'decode_image') as scope:
        if channels not in (None, 0, 1, 3):
            raise ValueError('channels must be in (None, 0, 1, 3)')
        substr = string_ops.substr(contents, 0, 4)

        def _gif():
            # Create assert op to check that bytes are GIF decodable
            is_gif = math_ops.equal(substr, b'\x47\x49\x46\x38', name='is_gif')
            decode_msg = 'Unable to decode bytes as JPEG, PNG, or GIF'
            assert_decode = control_flow_ops.Assert(is_gif, [decode_msg])
            # Create assert to make sure that channels is not set to 1
            # Already checked above that channels is in (None, 0, 1, 3)
            gif_channels = 0 if channels is None else channels
            good_channels = math_ops.not_equal(gif_channels, 1, name='check_channels')
            channels_msg = 'Channels must be in (None, 0, 3) when decoding GIF images'
            assert_channels = control_flow_ops.Assert(good_channels, [channels_msg])
            with ops.control_dependencies([assert_decode, assert_channels]):
                return gen_image_ops.decode_gif(contents)

        def _png():
            return gen_image_ops.decode_png(contents, channels)

        def check_png():
            is_png = math_ops.equal(substr, b'\211PNG', name='is_png')
            return control_flow_ops.cond(is_png, _png, _gif, name='cond_png')

        def _jpeg():
            return gen_image_ops.decode_jpeg(contents, channels)

        is_jpeg = math_ops.equal(substr, b'\xff\xd8\xff\xe0', name='is_jpeg')
        return control_flow_ops.cond(is_jpeg, _jpeg, check_png, name='cond_jpeg')
