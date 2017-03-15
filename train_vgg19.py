#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file trains a vgg 19 model on labeled image dataset. The overall framework is taken from
https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
Dataset from : http://foodcam.mobi/dataset256.html
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

import data_util
import vgg19
import discrim_net
from general_util import *

def main(a):
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "save":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization", "gray_input"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).iteritems():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = a.crop_size
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = data_util.load_examples(a)

    print("examples count = %d" % examples.count)

    #########
    if a.model == "vgg19":
        model = vgg19.create_model(examples.images, examples.labels, a)
    elif a.model == "discrim_net":
        model = discrim_net.create_model(examples.images, examples.labels, a)
    else:
        raise AssertionError("Not supported model %s" %(a.model))


    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("deprocess_inputs"):
        deprocessed_inputs = data_util.deprocess(examples.images, a)

    with tf.name_scope("argmax_labels"):
        argmax_labels = tf.argmax(examples.labels,axis=1)

    with tf.name_scope("argmax_outputs"):
        argmax_outputs = tf.argmax(model.outputs,axis=1)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, deprocessed_inputs, dtype=tf.string, name="input_pngs"),
            "labels": examples.labels,
            "outputs": model.outputs,
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", deprocessed_inputs)

    with tf.name_scope("labels_summary"):
        tf.summary.histogram("labels", argmax_labels)

    with tf.name_scope("outputs_summary"):
        tf.summary.histogram("outputs", argmax_outputs)


    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.accuracy)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = a.gpu_percentage
    with sv.managed_session(config=tf_config) as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        if a.mode == "save":
            model.save_npy(sess, npy_path = os.path.join(a.checkpoint,"vgg19_save.npy"))
        elif a.mode == "test":
            # testing
            # run a single epoch over all input data
            for step in range(examples.steps_per_epoch):
                results = sess.run(display_fetches)
                filesets = data_util.save_results(results, image_dir, examples.unique_labels)
                for i, path in enumerate(results["paths"]):
                    print(step * a.batch_size + i + 1, "evaluated image", os.path.basename(path))
                index_path = data_util.append_index(filesets, config=a)

            print("wrote index at", index_path)
        else:
            # training
            max_steps = 2**32
            if a.max_epochs is not None:
                max_steps = examples.steps_per_epoch * a.max_epochs
            if a.max_steps is not None:
                max_steps = a.max_steps

            start_time = time.time()
            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["loss"] = model.loss
                    fetches["accuracy"] = model.accuracy

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets =  data_util.save_results(results["display"], image_dir, examples.index2tag, step=results["global_step"])
                    data_util.append_index(filesets, a, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    global_step = results["global_step"]
                    print("progress  epoch %d  step %d  image/sec %0.1f" % (global_step // examples.steps_per_epoch, global_step % examples.steps_per_epoch, global_step * a.batch_size / (time.time() - start_time)))
                    print("loss", results["loss"])
                    print("accuracy", results["accuracy"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="path to folder containing images")
    parser.add_argument("--mode", required=True, choices=["train", "test", "save"])
    parser.add_argument("--output_dir", required=True, help="where to put output files")
    parser.add_argument("--db_dir", default = "pixiv_1T_db.pkl",
                        help="path to an already existing database or to store the new database.")
    parser.add_argument("--model", required=True, choices=["vgg19", "discrim_net", ])
    parser.add_argument("--vgg19_npy_path",  help="where to find pretrained the vgg19 file.")  #default='vgg19.npy',

    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_tags", type=int, default=1000)
    parser.add_argument("--checkpoint", default=None,
                        help="directory with checkpoint to resume training from or use for testing")

    parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
    parser.add_argument("--max_epochs", type=int, help="number of training epochs")
    parser.add_argument("--summary_freq", type=int, default=10, help="update summaries every summary_freq steps")
    parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
    # to get tracing working on GPU, LD_LIBRARY_PATH may need to be modified:
    # LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64
    parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
    parser.add_argument("--display_freq", type=int, default=0,
                        help="write current training images every display_freq steps")
    parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

    parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
    parser.add_argument("--gray_input", action="store_true", help="Treat input image as grayscale image.")
    parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
    parser.add_argument("--ngf", type=int, default=32, help="number of generator filters in first conv layer")
    parser.add_argument("--ndf", type=int, default=32, help="number of discriminator filters in first conv layer")
    parser.add_argument("--scale_size", type=int, default=286,
                        help="scale images to this size before cropping to 256x256")
    parser.add_argument("--crop_size", type=int, default=256, help="size to crop image into.")
    parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
    parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
    parser.set_defaults(flip=True)
    parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
    parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
    parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
    parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
    parser.add_argument("--trainable_layer", default="conv1_1",
                        choices=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1", "fc6"])
    parser.add_argument("--gpu_percentage", type=float, default=0.45, help="precent of gpu memory allocated.")
    a = parser.parse_args()
    main(a)

"""
python train_vgg19.py --mode train --output_dir sanity_check_train --model=discrim_net --max_epochs 200 --input_dir /home/xor/datasets/UECFOOD100 --display_freq=5000
python train_vgg19.py --mode train --output_dir sanity_check_train --max_epochs 20 --input_dir /mnt/tf_drive/home/ubuntu/datasets/UECFOOD256_sanity_check/ --display_freq=5000
python train_vgg19.py --mode test --output_dir sanity_check_test --input_dir /mnt/tf_drive/home/ubuntu/datasets/UECFOOD256_sanity_check/ --checkpoint sanity_check_train
python train_vgg19.py --mode train --output_dir UECFOOD256_train_iter_cont --max_epochs 50 --input_dir /mnt/tf_drive/home/ubuntu/datasets/UECFOOD256/ --display_freq=5000 --checkpoint=UECFOOD256_train_iter --trainable_layer=conv1_1
python train_vgg19.py --mode train --output_dir sanity_check_train --model=discrim_net --max_epochs 20000 --input_dir /mnt/data_drive/home/ubuntu/pixiv_new_sanity_check_128 --display_freq=5000 --batch_size=50 --crop_size=128 --scale_size=143 --lr=0.000001
python train_vgg19.py --mode train --output_dir pixiv_downloaded_sketches_lnet_128_train --model=discrim_net --max_epochs 40 --input_dir /mnt/data_drive/home/ubuntu/pixiv_downloaded_sketches_lnet_128/color --display_freq=5000 --batch_size=50 --crop_size=128 --scale_size=143 --lr=0.000001 --gpu_percentage=0.25

"""