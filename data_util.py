#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
import scipy.sparse
from collections import Counter

from general_util import *
from neural_util import decode_image

Examples = collections.namedtuple("Examples", "paths, images, labels, index2tag, count, steps_per_epoch")
PixivInfo = collections.namedtuple("PixivInfo",
                                   "artist_id, artist_name, image_id, title, caption, tags, comments, image_mode, pages, date, resolution, tools, bookmark_count, link, ugoira_data, description_urls")
Comment = collections.namedtuple("Comment", "user_id, comment")
PixivInfoNames = [
    "artist_id, artist_name, image_id, title, caption, tags, comments, image_mode, pages, date, resolution, tools, bookmark_count, link, ugoira_data, description_urls".split(
        ', ')]

pixiv_image_name_pattern = r"(\d+)_p(\d+) - (.*)\.(\w+)"
pixiv_image_name_re = re.compile(pixiv_image_name_pattern)
pixiv_info_pattern = r"(.*?)=(.*)"  # The first match is non-greedy to avoid matching other "=".
pixiv_info_pattern_re = re.compile(pixiv_info_pattern)
comment_split_pattern = r"(?<!\\), "
comment_split_re = re.compile(comment_split_pattern)
single_comment_pattern = r"comment_author_id: (\d+); comment: (.*)"
single_comment_re = re.compile(single_comment_pattern)
URL_TEXT = "Urls       =\n"  # notice the \r\n has already been replaced by \n


def deprocess(image, config):
    if config.aspect_ratio != 1.0:
        # upscale to correct aspect ratio
        size = [config.crop_size, int(round(config.crop_size * config.aspect_ratio))]
        image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)
    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)


def labels_to_one_hot_vector(labels):
    unique_labels = list(set(labels))

    if all(label.isdigit() for label in unique_labels):
        unique_labels = sorted(unique_labels, key=lambda label: int(label))
    else:
        unique_labels = sorted(unique_labels)
    num_unique_labels = len(unique_labels)
    label_vector_dict = {}
    for i, label in enumerate(unique_labels):
        one_hot_vector = np.zeros(num_unique_labels)
        one_hot_vector[i] = 1
        label_vector_dict[label] = one_hot_vector
    ret = [label_vector_dict[label] for label in labels]
    return ret, unique_labels


def one_hot_vector_to_labels(ohv, unique_labels):
    if ohv.ndim == 1:
        label_indices = np.argmax(ohv)
        labels = unique_labels[label_indices]
        return labels
    elif ohv.ndim == 2:
        label_indices = np.argmax(ohv, axis=1)
        labels = [unique_labels[i] for i in label_indices]
        return labels
    else:
        raise AssertionError("one hot vector must have 1 or 2 dimensions. Current shape is %s" % str(ohv.shape))

def parse_image_name_info(image_filename):
    if "/" in image_filename:
        print("Please input only the base filename (no directory) to parse_image_name_info function.")
        image_filename = os.path.basename(image_filename)
    image_metainfo = pixiv_image_name_re.match(image_filename)
    if image_metainfo is None:
        return None
    image_id = int(image_metainfo.group(1))
    image_pagenum = int(image_metainfo.group(2))
    image_name = image_metainfo.group(3)
    image_ext = image_metainfo.group(4)
    return image_id, image_pagenum, image_name, image_ext

def parse_pixiv_info(image_path):
    # Because the "comments" section may contain unescaped newline, and I don't know whether other ones will contain
    # newline as well, I have to use regex expression as a last resort, which maybe slow.
    # First get the correct path to the txt file, because one txt file can correspond to multiple pages(images).
    image_filename = os.path.basename(image_path)
    image_name_info = parse_image_name_info(image_filename)
    if image_name_info is None:
        # This happens only when the image name contains a "/" so a new folder was created for that image.
        # It is rare, but it happens. I will ignore it for now since it happens in less than 1/10000.
        raise AssertionError("Failed to parse image file name: %s." %(image_filename))
    image_id, image_pagenum, image_name, image_ext = image_name_info
    txt_path = image_path + '.txt'

    if not os.path.isfile(txt_path):
        return image_id  # For now. In the future, maybe find the corresponding one in the database or something.
    else:
        with open(txt_path, 'r') as f:
            content = f.read()
            content = content.replace("\r\n", "\n").strip()
            if URL_TEXT in content:
                content_url_split = content.split(URL_TEXT)
                content = content_url_split[0].strip()
                content_urls = content_url_split[1].split("\n")
            else:
                content_urls = []
            content_lines = content.split("\n")
            # First try the fast version, checking that there is no newline anywhere except at the end of line..
            if len(content_lines) == 15:
                current_info = []
                for i, line in enumerate(content_lines):
                    current_subinfo = pixiv_info_pattern_re.match(line)
                    current_info.append(current_subinfo.group(2).strip())
            else:
                try:
                    # If that does not work, try to break the content into three parts: before comment, in comment, after comment.
                    before_comment, nxt = content.split("\nComments   = ")
                    in_comment, after_comment = nxt.split("\nImage Mode = ")
                    # Note incomment and after comment does not include the first "blablabla = "

                    current_info = []
                    content_lines = before_comment.split("\n")
                    assert len(content_lines) == 6
                    for i, line in enumerate(content_lines):
                        current_subinfo = pixiv_info_pattern_re.match(line)
                        current_info.append(current_subinfo.group(2).strip())

                    current_info.append(in_comment)

                    after_comment = "Image Mode = " + after_comment
                    content_lines = after_comment.split("\n")
                    assert len(content_lines) == 7
                    for i, line in enumerate(content_lines):
                        current_subinfo = pixiv_info_pattern_re.match(line)
                        current_info.append(current_subinfo.group(2).strip())
                except AssertionError:
                    # If that does not work, that means there are newlines elsewhere not in the comment area.
                    # I can use regex to get everything...
                    raise NotImplementedError

            # Sanity check that the info in the file name matches the info in content.
            if not image_id == int(current_info[2]):  # and image_name == current_info[3]
                raise AssertionError("Image id in the image file name is different from the one in the .txt file.")
            return PixivInfo(artist_id=int(current_info[0]),
                             artist_name=current_info[1],
                             image_id=image_id,
                             title=current_info[3],
                             caption=current_info[4],
                             tags=current_info[5].split(', '),
                             comments=parse_comments(current_info[6]),
                             image_mode=current_info[7],
                             pages=int(current_info[8]),
                             date=dateparser.parse(current_info[9]),
                             resolution=current_info[10],
                             tools=current_info[11],
                             bookmark_count=int(current_info[12]),
                             link=current_info[13],
                             ugoira_data=current_info[14],
                             description_urls = content_urls)


def parse_comments(line):
    ret = []
    if len(line.strip()) == 0:
        return ret
    # Use regex negative lookbehind
    comments = re.split(comment_split_re, line)
    for comment in comments:
        comment_info = single_comment_re.match(comment)
        if comment_info is None:
            raise AssertionError("Comment %s in comments line %s does not seem to follow the convention." %(comment, line))
        else:
            comment_user_id = comment_info.group(1)
            comment_content = comment_info.group(2)
        ret.append(Comment(user_id=int(comment_user_id), comment=comment_content))
    return ret

def create_database(input_dir):

    if not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")

    input_paths = get_all_image_paths_in_dir(input_dir)
    num_images = len(input_paths)
    print("Number of images: %s. Start creating database." %(num_images))

    if num_images == 0:
        raise Exception("input_dir contains no image files")

    start_time = time.time()
    database = {}
    for path_i, input_path in enumerate(input_paths):
        if path_i % 100 == 0:
            current_time = time.time()
            remaining_time = 0.0 if path_i == 0 else (num_images - path_i) * (float(current_time - start_time) / path_i)
            print('%.3f%% done. Remaining time: %.1fs' % (float(path_i) / num_images * 100, remaining_time))
        try:
            pixiv_info = parse_pixiv_info(input_path)
            if isinstance(pixiv_info, PixivInfo):
                image_id = pixiv_info.image_id
                if image_id in database:
                    print("Wierd... image_id %d already appeared in database." %image_id)
                else:
                    database[image_id] = pixiv_info
        except Exception as exc:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print("Exception %s occured during database construction for image %s" %(str(exc), input_path))
    return database

def get_tags_count(database, most_common_count = None):
    tags = []
    for pixiv_info in database.values():
        tags += pixiv_info.tags
    tags_count = Counter(tags)
    if most_common_count is None or most_common_count <= 0:
        return dict(tags_count)
    else:
        assert isinstance(most_common_count, int)
        most_common = dict(tags_count.most_common(most_common_count))
        assert len(most_common) == most_common_count
        return most_common

def get_image_paths_in_database(database, image_paths):
    ret = []
    for image_i, image_path in enumerate(image_paths):
        try:
            image_file_name = os.path.basename(image_path)
            image_name_info = parse_image_name_info(image_file_name)
            if image_name_info is None:
                # This happens only when the image name contains a "/" so a new folder was created for that image.
                # It is rare, but it happens. I will ignore it for now since it happens in less than 1/10000.
                raise AssertionError("Failed to parse image file name: %s." %(image_file_name))
            image_id, image_pagenum, image_name, image_ext = image_name_info
            if image_id in database:
                ret.append(image_path)

        except Exception as exc:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print("Exception %s occured during label construction for image %s" % (str(exc), image_path))
    return ret

def get_image_paths_subdir(image_paths, parent_dir):
    # This function takes out the common prefix parent directory
    len_parent_dir = len(parent_dir)
    return map(lambda path: path[len_parent_dir:], image_paths)

def create_labels(database, image_paths, most_common_count=10000):
    num_images = len(image_paths)
    print("Number of images: %s. Start creating labels." %(num_images))

    if num_images == 0:
        raise Exception("input_dir contains no image files")
    # First count the number of times each tag appears in the database. Then take out the ones with occurrence below
    # some threshold.
    tags_count = get_tags_count(database, most_common_count=most_common_count)
    tag2index = {label: i for i,label in enumerate(sorted(tags_count.keys()))}
    index2tag= {i: label for i,label in enumerate(sorted(tags_count.keys()))}
    image_labels = np.zeros((len(image_paths), most_common_count), dtype=np.bool)

    start_time = time.time()
    for image_i, image_path in enumerate(image_paths):
        if image_i % 10000 == 0:
            current_time = time.time()
            remaining_time = 0.0 if image_i == 0 else (num_images - image_i) * (float(current_time - start_time) / image_i)
            print('%.3f%% done. Remaining time: %.1fs' % (float(image_i) / num_images * 100, remaining_time))
        try:
            image_file_name = os.path.basename(image_path)
            image_name_info = parse_image_name_info(image_file_name)
            if image_name_info is None:
                # This happens only when the image name contains a "/" so a new folder was created for that image.
                # It is rare, but it happens. I will ignore it for now since it happens in less than 1/10000.
                raise AssertionError("Failed to parse image file name: %s." %(image_file_name))
            image_id, image_pagenum, image_name, image_ext = image_name_info
            if image_id in database:
                image_tags = database[image_id].tags
                for tag in image_tags:
                    if tag in tags_count:
                        image_labels[image_i, tag2index[tag]] = True
            else:
                raise AttributeError("image %s not in database." %(image_path))
        except Exception as exc:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print("Exception %s occured during label construction for image %s" %(str(exc), image_path))
    return image_labels, tag2index, index2tag

# TODO: finish the test case for create_labels
# TODO: write the pipeline for loading pixiv data.
def load_examples(config):
    if not os.path.exists(config.input_dir):
        raise Exception("input_dir does not exist")
    if not os.path.isfile(config.db_dir):
        print("Database does not exist, creating new one.")
        database = create_database(config.input_dir)
        save_pickle(database, config.db_dir)
    else:
        database = load_pickle(config.db_dir)


    # input_paths = get_files_with_ext(config.input_dir, "jpg")  # glob.glob(os.path.join(config.input_dir, "*.jpg"))
    # decode = tf.image.decode_jpeg
    # if len(input_paths) == 0:
    #     input_paths = get_files_with_ext(config.input_dir, "png")  # glob.glob(os.path.join(config.input_dir, "*.png"))
    #     decode = tf.image.decode_png
    input_paths = get_all_image_paths_in_dir(config.input_dir)
    input_paths = get_image_paths_in_database(database,input_paths)
    # By taking out the common prefix parent directory, it saves some memory when the input_paths is huge.
    input_paths = get_image_paths_subdir(input_paths, config.input_dir)

    decode = decode_image

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    # Assume the subdirectories containing the input images is their categories/labels.
    labels, tag2index, index2tag = create_labels(database, image_paths=input_paths, most_common_count=config.num_tags)
    # labels, unique_labels = labels_to_one_hot_vector(labels)

    with tf.name_scope("load_examples"):
        # path_queue = tf.train.string_input_producer(input_paths, shuffle=config.mode == "train")
        # The slice input producer can produce as many queus as the user wants.
        # labels = tf.constant(np.array(labels, dtype=np.bool))
        labels = tf.constant(labels)
        input_queue = tf.train.slice_input_producer([input_paths, labels], shuffle=config.mode == "train")
        path_queue = tf.string_join([config.input_dir, input_queue[0]])
        label_queue = tf.to_float(input_queue[1])

        # reader = tf.WholeFileReader()
        # paths, contents = reader.read(path_queue)
        # Can't use whole file reader, so use tf.read_file instead.
        contents = tf.read_file(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        if config.gray_input:
            images = tf.image.rgb_to_grayscale(raw_input)
        else:
            images = raw_input

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2 ** 31 - 1)

    def transform(image):
        r = image
        if config.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [config.scale_size, config.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, config.scale_size - config.crop_size + 1, seed=seed)),
                         dtype=tf.int32)
        if config.scale_size > config.crop_size:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], config.crop_size, config.crop_size)
        elif config.scale_size < config.crop_size:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        images = transform(images)
    images = tf.image.resize_images(images, [config.crop_size, config.crop_size], method=tf.image.ResizeMethod.AREA)

    paths, images, labels = tf.train.batch([path_queue, images, label_queue], batch_size=config.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / config.batch_size))

    return Examples(
        paths=paths,
        images=images,
        labels=labels,
        index2tag=index2tag,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )

def logits_to_labels(logits, index2tag):
    # Turn a probability distribution into a list of actual tags (if the tag has probability over 50%)
    assert isinstance(logits, np.ndarray) and len(logits.shape) == 1 or len(logits.shape) == 2

    ret = []
    if len(logits.shape) == 1:
        tag_indices = (logits.astype(np.float) >= 0.5)
        for index in range(tag_indices.shape[0]):
            if tag_indices[index] == True:
                ret.append(index2tag[index])
    elif len(logits.shape) == 2:
        for i in range(logits.shape[0]):
            tag_indices = (logits[i].as_type(np.float) >= 0.5)
            current_tags = []
            for index in range(tag_indices.shape[0]):
                if tag_indices[index] == True:
                    current_tags.append(index2tag[index])
            ret.append(current_tags)
    return ret


def save_results(fetches, image_dir, index2tag, step=None):
    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path))
        fileset = {"name": name, "step": step}
        for kind in ["inputs"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "w") as f:
                f.write(contents)
        for kind in ["labels", "outputs"]:
            contents = fetches[kind][i]
            labels = logits_to_labels(contents, index2tag)
            fileset[kind] = labels
        filesets.append(fileset)
    return filesets

def append_index(filesets, config, step=False):
    index_path = os.path.join(config.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write(
            "<html><meta content=\"text/html;charset=utf-8\" http-equiv=\"Content-Type\"><meta content=\"utf-8\" http-equiv=\"encoding\"><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")
        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs"]:
            index.write("<td><img src=\"images/%s\"></td>" % urllib.quote(fileset[kind]))
        for kind in ["outputs", "labels"]:
            index.write("<td>%s</td>" % str(', '.join(fileset[kind])))

        index.write("</tr>")
    return index_path

def calc_pca(data, n_components = 20):
    # assume data is in numpy array format with boolean dtype.
    if len(data.shape) != 2:
        raise AssertionError("The calc_pca function is assuming data is in numpy array format with shape "
                             "(n_samples, n_features). Now it is %s" %(str(data.shape)))
    pca = PCA(n_components=n_components)
    pca.fit(data)
    data_pca_components = pca.transform(data)  # Shape (n_samples, n_components)

    # Return shape: (n_samples), (n_components, n_features)
    return np.argmax(data_pca_components, axis=1), pca.components_

def calc_sparse_pca(data, n_components = 20):
    # assume data is in numpy array format with boolean dtype.
    # if len(data.shape) != 2:
    #     raise AssertionError("The calc-pca function is assuming data is in numpy array format with boolean dtype.")
    sparse_data = scipy.sparse.csr_matrix(data)
    # pca = PCA(n_components=n_components)
    # pca.fit(sparse_data)
    # svd = TruncatedSVD()
    pca = TruncatedSVD(n_components=n_components)
    pca.fit(sparse_data)
    data_pca_components = pca.transform(sparse_data)  # Shape (n_samples, n_components)

    # Return shape: (n_samples), (n_components, n_features)
    return np.argmax(data_pca_components, axis=1), pca.components_

def cluster_images(input_dir, db_dir, output_dir, tag_max_count, num_clusters):
    if not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")
    if not os.path.exists(output_dir):
        print("output_dir does not exist. Creating it.")
        os.mkdir(output_dir)
    if not os.path.isfile(db_dir):
        print("Database does not exist, creating new one.")
        database = create_database(input_dir)
        save_pickle(database, db_dir)
    else:
        database = load_pickle(db_dir)

    input_paths = get_all_image_paths_in_dir(input_dir)
    input_paths = get_image_paths_in_database(database,input_paths)
    image_labels , tag2index, index2tag = create_labels(database, input_paths , most_common_count=tag_max_count)
    image_clusters, components_features = calc_sparse_pca(image_labels, n_components = num_clusters)
    assert image_clusters.shape[0] == len(input_paths)

    # Print the component features
    NUM_INFLUENCIAL_TAGS_TO_PRINT = min(5, tag_max_count)

    cluster_summary_file_name = os.path.join(output_dir, "cluster_summary.txt")
    with open(cluster_summary_file_name, 'w') as f:
        for i in range(num_clusters):
            cluster_components = []
            # for j in range(tag_max_count):
            #     cluster_components.append("%s: %.4f" %(index2tag[j], components_features[i,j]))
            # print("Cluster %d is made up of the following tag components: \n\t%s" %(i, ', '.join(cluster_components)))
            # Or I can print say the first 5 tags that most influence the cluster... Maybe this is better.
            sorted_features = np.argsort(components_features[i])
            for j in range(-1, -NUM_INFLUENCIAL_TAGS_TO_PRINT-1 ,-1):
                cluster_components.append("%s: %.4f" % (index2tag[sorted_features[j]], components_features[i, sorted_features[j]]))

            print("Cluster %d is most influenced by the following tags: \n\t%s" %(i, ', '.join(cluster_components)))
            f.write("Cluster %d is most influenced by the following tags: \n\t%s\n" %(i, ', '.join(cluster_components)))

    # Output a series of files, each containing paths to images within the same cluster.
    for i in range(num_clusters):
        current_cluster_file_name = os.path.join(output_dir, "cluster_%d.txt" %(i))
        current_cluster_num_images = 0
        with open(current_cluster_file_name, 'w') as f:
            for image_i in range(image_clusters.shape[0]):
                if image_clusters[image_i] == i:
                    f.write(input_paths[image_i] + "\n")
                    current_cluster_num_images += 1
        print("Image cluster %d has %d images." %(i, current_cluster_num_images))

        with open(cluster_summary_file_name, 'a') as f:
            f.write("Image cluster %d has %d images.\n" %(i, current_cluster_num_images))

def print_common_labels(db_dir, tag_max_count):
    if not os.path.isfile(db_dir):
        raise AssertionError("Database does not exist.")
    database = load_pickle(db_dir)
    most_common_tags_count = get_tags_count(database, tag_max_count)
    sorted_tags_count = sorted(most_common_tags_count.iteritems(), key=lambda tag_item: tag_item[1], reverse=True)
    for tag_name, tag_count in sorted_tags_count:
        print(tag_name + " appeared " + str(tag_count) + " times.")


