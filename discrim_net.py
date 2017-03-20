#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file loads a checkpoint from trained 128x128 model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import urllib

from general_util import imread, get_all_image_paths_in_dir
from neural_util import decode_image

EPS = 1e-12
CLIP_VALUE = 0.04  # 0.01
APPROXIMATE_NUMBER_OF_TOTAL_PARAMETERS = 98218176

SKETCH_VAR_SCOPE_PREFIX = "sketch_"

Model = collections.namedtuple("Model", "loss, outputs, train, accuracy, precision, recall")


def conv(batch_input, out_channels, stride, shift=4, pad = 1, trainable=True):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [shift, shift, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02), trainable=trainable)
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        if pad > 0:
            padded_input = tf.pad(batch_input, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="CONSTANT")
        else:
            assert pad == 0
            padded_input = batch_input
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input, trainable=True):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer, trainable=trainable)
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02), trainable=trainable)
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels, stride = 2, shift = 4, trainable=True):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [shift, shift, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02), trainable=trainable)
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * stride, in_width * stride, out_channels], [1, stride, stride, 1], padding="SAME")
        return conv

def fc_layer(input, in_size, out_size, trainable=True):
    biases = tf.get_variable("biases", [out_size], dtype=tf.float32, initializer=tf.zeros_initializer, trainable=trainable)
    weights = tf.get_variable("weights", [in_size, out_size], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02), trainable=trainable)

    x = tf.reshape(input, [-1, in_size])
    fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

    return fc

def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    # Input range [0, 255]
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    # Input range for l is 0 ~ 100 and ab is -110 ~ 110
    # Output range is 0 ~ 1....???
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))


def create_model(inputs, targets, config):

    def create_discriminator(discrim_inputs, output_classes, label_classifier = True, fc_feature_num=1024):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        # input = tf.concat_v2([discrim_inputs, discrim_targets], axis=3)
        input = discrim_inputs

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            # convolved = conv(input, a.ndf, stride=2)
            convolved = conv(input, config.ndf, stride=2, shift=4)
            normed = batchnorm(convolved)
            rectified = lrelu(normed, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        layer_specs = [
            config.ndf,  # encoder_2: [batch, 256, 256, ngf] => [batch, 128, 128, ngf * 2]
            config.ndf * 2,  # encoder_2: [batch, 256, 256, ngf] => [batch, 128, 128, ngf * 2]
            config.ndf * 2,  # encoder_3: [batch, 128, 128, ngf * 2] => [batch, 128, 128, ngf * 2]
            config.ndf * 4,  # encoder_4: [batch, 128, 128, ngf * 2] => [batch, 64, 64, ngf * 4]
            config.ndf * 4,  # encoder_5: [batch, 64, 64, ngf * 4] => [batch, 64, 64, ngf * 4]
            config.ndf * 8,  # encoder_6: [batch, 64, 64, ngf * 4] => [batch, 32, 32, ngf * 8]
        ]
        for out_channels in layer_specs:
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                if (len(layers) + 1) % 2 == 0:
                    # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                    convolved = conv(layers[-1], out_channels, stride=1, shift=3)
                else:
                    # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                    convolved = conv(layers[-1], out_channels, stride=2, shift=4)

                normed = batchnorm(convolved)
                # rectified = lrelu(normed, 0.2)
                rectified = tf.nn.relu(normed)
                layers.append(rectified)
        # for i in range(n_layers):
        #     with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        #         out_channels = a.ndf * min(2**(i+1), 8)
        #         stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
        #         convolved = conv(layers[-1], out_channels, stride=stride)
        #         normalized = batchnorm(convolved)
        #         rectified = lrelu(normalized, 0.2)
        #         layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            # convolved = conv(rectified, out_channels=1, stride=1)
            # output = tf.sigmoid(convolved)
            # layers.append(output)
            # With WGAN, sigmoid for the last layer is no longer needed
            convolved = conv(rectified, out_channels=1, stride=1, shift=3)
            layers.append(convolved)

        if label_classifier:
            # Not sure how many features I should have, so I used a variable fc_feature_num instead of 4096 like vgg19.
            batch_num, height, width, num_features = layers[-1].get_shape().as_list()
            with tf.variable_scope("fc_%d" % (len(layers) + 1)):
                fc1 = fc_layer(layers[-1], height*width*num_features, fc_feature_num)
                rectified1 = tf.nn.relu(fc1)
                layers.append(rectified1)
            with tf.variable_scope("fc_%d" % (len(layers) + 1)):
                fc2 = fc_layer(rectified1, fc_feature_num, fc_feature_num)
                rectified2 = tf.nn.relu(fc2)
                layers.append(rectified2)
            with tf.variable_scope("fc_%d" % (len(layers) + 1)):
                output = fc_layer(rectified2, fc_feature_num, output_classes)
                layers.append(output)

        return layers[-1]
    
    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):

            predict = create_discriminator(inputs, targets.get_shape().as_list()[1])

    with tf.name_scope("loss"):


        # This is for one single right answer.
        # targets_indices = tf.argmax(targets,axis=1)
        # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(predict, targets_indices))
        #
        # predict_indices = tf.argmax(predict, 1)
        # correct_pred = tf.equal(predict_indices, targets_indices)
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # For multi class.
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, targets))

        cutoff = tf.constant(0.5)
        predictions = tf.greater_equal(predict, cutoff)
        targets_bool = tf.greater_equal(targets, cutoff)
        correct_pred = tf.equal(predictions, targets_bool)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        num_positive = tf.reduce_sum(tf.cast(targets_bool, tf.float32))
        true_positive = tf.reduce_sum(tf.cast(tf.logical_and(targets_bool, correct_pred), tf.float32))
        # true_negative = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(targets_bool), correct_pred), tf.float32))
        false_positive =tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(targets_bool), tf.logical_not(correct_pred)), tf.float32))
        false_negative = num_positive - true_positive
        precision = true_positive / (true_positive + false_positive + 0.0000001)
        recall = true_positive / (true_positive + false_negative + 0.0000001)



    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        non_fc_tvars = [var for var in discrim_tvars if "fc_" not in var.name]
        # discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        # WGAN does not use momentum based optimizer
        discrim_optim = tf.train.RMSPropOptimizer(config.lr)
        # discrim_train = discrim_optim.minimize(discrim_loss, var_list=discrim_tvars)

        # WGAN adds a clip and train discriminator 5 times
        discrim_min = discrim_optim.minimize(loss, var_list=discrim_tvars)
        # Only clip the non-fully connected layer variables.
        # discrim_clips = [var.assign(tf.clip_by_value(var, -CLIP_VALUE, CLIP_VALUE)) for var in non_fc_tvars]
        # # No difference between control dependencies and group.
        # # with tf.control_dependencies([discrim_min] + discrim_clips):
        # #     discrim_train = tf.no_op("discrim_train")
        # with tf.control_dependencies([discrim_min]):
        #     discrim_train = tf.group(*discrim_clips)
        discrim_train = discrim_min

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([loss,accuracy,precision, recall])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        loss=ema.average(loss),
        outputs=predict,
        train=tf.group(update_losses, incr_global_step, discrim_train),
        accuracy= ema.average(accuracy), # If this does not work, take out the ema.
        precision= ema.average(precision),
        recall= ema.average(recall),
    )
