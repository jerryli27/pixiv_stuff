#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is an experiment. It attempts to use features in deep nn to cluster different painting styles. (Unsupervised)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import glob
import random
import math
import time
import collections
import urllib
import re
import sys
import traceback
import dateparser
import shutil
import argparse
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
import scipy.sparse
from collections import Counter

from general_util import *
from neural_util import decode_image
from data_util import calc_k_mean
from vgg19 import Vgg19


Examples = collections.namedtuple("Examples", "paths, inputs, count, steps_per_epoch")

def gramian(layer):
    # type: (tf.Tensor) -> tf.Tensor
    """
    :param layer: tensor with shape (batch_size, height, width, num_features)
    :return: The gramian of the layer -- a tensor with dimension gramians of dimension (batches, channels, channels)
    """
    # Instead of iterating over #channels width by height matrices and computing similarity, we vectorize and compute
    # the entire gramian in a single matrix multiplication.
    _, height, width, number = map(lambda i: i.value, layer.get_shape())
    size = height * width * number
    layer_unpacked = tf.unpack(layer)
    grams = []
    for single_layer in layer_unpacked:
        feats = tf.reshape(single_layer, (-1, number))
        # Note: the normalization factor might be wrong. I've seen many different forms of normalization. The current
        # one works though.
        grams.append(tf.matmul(tf.transpose(feats), feats) / size)
    return tf.pack(grams)


def create_vgg_net(inputs, vgg19_npy_path, output_classes = 1000):
    vgg = Vgg19(vgg19_npy_path=vgg19_npy_path)  # Read model from pretrained vgg
    train_mode = tf.constant(False, dtype=tf.bool, name='train_mode')
    vgg.build(inputs, output_classes, train_mode=train_mode)
    vgg_19_net = vgg.net()
    return vgg_19_net

def load_examples(input_paths, batch_size, scale_size, decoder=decode_image):
    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)
    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=False)
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decoder(contents, channels=3)

        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])
        def transform(image):
            r = image
            # area produces a nice downscaling, but does nearest neighbor for upscaling
            # assume we're going to be doing downscaling here
            r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
            return r

        input_images = transform(raw_input)

        paths, inputs, = tf.train.batch([paths, input_images,], batch_size=batch_size)
        steps_per_epoch = int(math.ceil(len(input_paths) / batch_size))
        return Examples(
            paths=paths,
            inputs=inputs,
            count=len(input_paths),
            steps_per_epoch=steps_per_epoch,
        )

def create_model(inputs, vgg19_npy_path):
    with tf.variable_scope("vgg19") as scope:
        net = create_vgg_net(inputs, vgg19_npy_path)
        conv_layer = net["conv2_1"]
        # conv_layer = inputs
        gram = gramian(conv_layer)
    return gram

def create_labels(image_paths, vgg19_npy_path, batch_size, scale_size, gpu_percentage, decoder, flat=True):
    examples = load_examples(image_paths, batch_size, scale_size, decoder)
    output = create_model(examples.inputs,vgg19_npy_path)
    labels = []

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_percentage
    sv = tf.train.Supervisor(logdir=None, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=config) as sess:
        start_time = time.time()
        max_steps = examples.steps_per_epoch
        for step in range(max_steps):

            # # First get input and target
            # io_fetches = {
            #     "inputs": examples.inputs,
            #     "targets": examples.targets
            # }
            # # if should(a.display_freq):
            # #     io_fetches["display"] = display_fetches
            # io_results = sess.run(io_fetches, options=options, run_metadata=run_metadata)
            #
            # print(np.sum(io_results["inputs"][...,1:]))
            # print(np.sum(io_results["targets"]))


            fetches = {
                "image_path": examples.paths,
                "output": output,
            }

            results = sess.run(fetches)
            print("image_path: %s" % (", ".join(results["image_path"])))
            labels.append(results["output"])
            # Now write the labels to some file? Or store it?

            if sv.should_stop():
                break

    labels = np.concatenate(labels, axis=0)
    if flat:
        labels = np.reshape(labels, (labels.shape[0], -1))
    return labels

def cluster_images(input_dir, output_dir, vgg19_npy_path, batch_size, scale_size, gpu_percentage, num_clusters, do_copy = False):
    if not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")
    if not os.path.exists(output_dir):
        print("output_dir does not exist. Creating it.")
        os.mkdir(output_dir)

    # image_paths = get_all_image_paths_in_dir(input_dir)

    image_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(image_paths) == 0:
        image_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    num_images = len(image_paths)
    print("Number of images: %s. Start creating labels." %(num_images))

    if num_images == 0:
        raise Exception("input_dir contains no image files")

    image_labels = create_labels(image_paths, vgg19_npy_path, batch_size, scale_size, gpu_percentage, decoder=decode)
    # image_clusters, components_features = calc_pca(image_labels, n_components = num_clusters)
    image_clusters, components_features = calc_k_mean(image_labels, n_clusters = num_clusters)
    assert image_clusters.shape[0] == len(image_paths)

    # Print the component features

    cluster_summary_file_name = os.path.join(output_dir, "cluster_summary.txt")
    with open(cluster_summary_file_name, 'w'):
        pass
    # Output a series of files, each containing paths to images within the same cluster.
    for i in range(num_clusters):
        current_cluster_file_name = os.path.join(output_dir, "cluster_%d.txt" %(i))
        if do_copy:
            current_cluster_path_name = os.path.join(output_dir, "cluster_%d" % (i))
            if not os.path.exists(current_cluster_path_name):
                os.mkdir(current_cluster_path_name)
        current_cluster_num_images = 0
        with open(current_cluster_file_name, 'w') as f:
            for image_i in range(image_clusters.shape[0]):
                if image_clusters[image_i] == i:
                    f.write(image_paths[image_i] + "\n")
                    current_cluster_num_images += 1
                    if do_copy:
                        image_basename = os.path.basename(image_paths[image_i])
                        shutil.copy(image_paths[image_i], os.path.join(current_cluster_path_name, image_basename))
        print("Image cluster %d has %d images." %(i, current_cluster_num_images))

        with open(cluster_summary_file_name, 'a') as f:
            f.write("Image cluster %d has %d images.\n" %(i, current_cluster_num_images))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="path to folder containing images")
    parser.add_argument("--vgg19_npy_path", default = "vgg19.npy",
                        help="path to a pretrained vgg network.")
    parser.add_argument("--output_dir", default = "pixiv_1T_feature_clustered/", help="output path")
    parser.add_argument("--tag_max_count", type=int, default=10000,
                        help="number of the most popular tags to be taken into account.")
    parser.add_argument("--num_clusters", type=int, default=5, help="Number of clusters to generate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of clusters to generate.")
    parser.add_argument("--scale_size", type=int, default= 256,
                        help="scale images to this size")
    parser.add_argument("--gpu_percentage", type=float, default=1.0, help="Number of clusters to generate.")
    parser.add_argument("--do_copy", dest="do_copy", action="store_true",
                        help="Copy the images into their corresponding clusters.")
    parser.set_defaults(do_copy=False)
    a = parser.parse_args()
    cluster_images(a.input_dir, a.output_dir, a.vgg19_npy_path, a.batch_size, a.scale_size, a.gpu_percentage, a.num_clusters, a.do_copy)