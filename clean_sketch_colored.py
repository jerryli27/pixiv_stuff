
import argparse
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.preprocessing import binarize
from PIL import ImageStat, Image

from general_util import get_all_image_paths_in_dir, imread, imsave

IMG_TYPE_GRAYSCALE = 1
IMG_TYPE_COLOR = 2
IMG_TYPE_BW = 3
IMG_TYPE_UK = 4

def png_path(path):
    basename, _ = os.path.splitext(os.path.basename(path))
    return os.path.join(os.path.dirname(path), basename + ".png")
def grayscale(img, keep_dim = False):
    img = img / 255.0
    img = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    if keep_dim:
        return (np.expand_dims(img, axis=2) * 255).astype(np.uint8)
    else:
        return (img * 255).astype(np.uint8)
def gray2rgb(img):
    return np.repeat(np.expand_dims(img, axis=2),3, axis=2)

def detect_bw(img, thumb_size=40, MSE_cutoff=22, adjust_color_bias=True):
    # type: (str, int, int, bool) -> int
    # Mainly copied from
    # http://stackoverflow.com/questions/14041562/python-pil-detect-if-an-image-is-completely-black-or-white
    # pil_img = Image.open(file)
    pil_img = Image.fromarray(img)
    bands = pil_img.getbands()
    if bands == ('R', 'G', 'B') or bands == ('R', 'G', 'B', 'A'):
        thumb = pil_img.resize((thumb_size, thumb_size))
        SSE, bias = 0, [0, 0, 0]
        if adjust_color_bias:
            bias = ImageStat.Stat(thumb).mean[:3]
            bias = [b - sum(bias) / 3 for b in bias]
        for pixel in thumb.getdata():
            mu = sum(pixel) / 3
            SSE += sum((pixel[i] - mu - bias[i]) * (pixel[i] - mu - bias[i]) for i in [0, 1, 2])
        MSE = float(SSE) / (thumb_size * thumb_size)
        if MSE <= MSE_cutoff:
            # print "grayscale\t",
            return IMG_TYPE_GRAYSCALE
        else:
            return IMG_TYPE_COLOR
            # print "Color\t\t\t",
        # print "( MSE=", MSE, ")"
    elif len(bands) == 1:
        # print "Black and white", bands
        return IMG_TYPE_BW
    else:
        # print "Don't know...", bands
        return IMG_TYPE_UK

# def find(d):
#     result = []
#     num_illegal_lines = 0
#     with open(d, 'r') as f:
#         for line in f.readlines():
#             line_splitted = line.split("\t")
#             if len(line_splitted) == 4:
#                 result.append((line_splitted[1], line_splitted[3].strip()))
#             else:
#                 num_illegal_lines += 1
#     assert num_illegal_lines < len(result)
#     result.sort()
#     return result

def get_sketch_img(img):
    width = img.shape[1]  # [height, width, channels]
    return img[:, :width // 2, :]

# def detect_wierd_distribution(sketch_img):
#     height, width, _ = sketch_img.shape
#     above_20 =  binarize(grayscale(sketch_img), threshold=20)
#     below_235 = 1 - binarize(grayscale(sketch_img), threshold=235)
#     percent_pixels_not_bw = np.sum(np.sum(above_20 * below_235)) / float(height * width)
#     if percent_pixels_not_bw >= 0.20:
#         print("percent_pixels_not_bw: %.3f" %(percent_pixels_not_bw))
#         # plt.hist(np.expand_dims(sketch_img,axis=2).ravel(), 256, [-1, 256]);
#         # plt.show(block=True)
#         return True
#     else:
#         return False


def detect_need_negate(sketch_img):
    if np.mean(sketch_img) < 40:
        return True
    else:
        return False

def combined_image_sketch_negate(img):
    width = img.shape[1]  # [height, width, channels]
    a_images = img[:, :width // 2, :]
    b_images = img[:, width // 2:, :]
    # Now negate a.
    a_images = 255 - a_images
    img = np.concatenate([a_images, b_images], axis=1)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="path to folder containing images")
    parser.add_argument("--last_stopped", type=int, default=0, help="Index to the image that you last stopped at.")
    # parser.add_argument("--output_path", default = "sketch_colored_cleaned", help="output path")
    a = parser.parse_args()

    all_image_paths = get_all_image_paths_in_dir(a.input_path)
    num_images = len(all_image_paths)

    img_shape = (128,256)
    # fig = plt.figure()
    # ax = plt.axes(xlim=(0, img_shape[1]), ylim=(0, img_shape[0]))
    # myimg = ax.imshow(np.zeros(img_shape, np.uint8))
    # plt.show()
    myimg = plt.imshow(np.zeros(img_shape, np.uint8))
    fig = plt.gcf()

    image_i = max(a.last_stopped, 0)
    last_image_i = -1
    image_i_after_cleaning = image_i
    while image_i < num_images:
        src_path = all_image_paths[image_i]
        # src_img = mpimg.imread(src_path)
        src_img = imread(src_path, shape=img_shape,dtype=np.uint8)
        sketch_part = get_sketch_img(src_img)
        # Only negate when the human did not perform action 1 or 2.
        if last_image_i != image_i and detect_need_negate(sketch_part):
            src_img = combined_image_sketch_negate(src_img)
            imsave(src_path, src_img)
            src_img = imread(src_path, shape=img_shape, dtype=np.uint8)
            sketch_part = get_sketch_img(src_img)

        assert src_img.shape[0] == img_shape[0] and src_img.shape[1] == img_shape[1]
        # if detect_wierd_distribution(sketch_img=sketch_part):
        # last_image_i = image_i
        # image_i += 1
        myimg.set_data(src_img)
        plt.pause(0.1)
        # If something is wrong, just rename the image...
        """
        1. Inversed sketch image so the sketch is white.
        2. incorrect sketch image and/or corresponding colored image
        3. Maybe padding is a little bit better than cropping
        4. Sketch is not totally either 0 or 255 (if the author say scanned their sketch)
        5. Sketch may have a little bit more than sketch -- that is the sketch might be colored a little or the sketch
        might have shade indicators etc.
        """
        action = raw_input("Image %d. after clean: %d Enter anything to rename the image." %(image_i, image_i_after_cleaning)).strip()
        if len(action) != 0:
            if action == "1":
                # width = src_img.shape[1]  # [height, width, channels]
                # a_images = src_img[:, :width // 2, :]
                # b_images = src_img[:, width // 2:, :]
                # # Now negate a.
                # a_images = 255 - a_images
                # src_img = np.concatenate([a_images, b_images], axis=1)
                src_img = combined_image_sketch_negate(src_img)
                imsave(src_path, src_img)
                # os.system('convert "%s" -negate "%s"' %(src_path, src_path))
            elif action == "2":
                width = src_img.shape[1]  # [height, width, channels]
                a_images = src_img[:, :width // 2, :]
                b_images = src_img[:, width // 2:, :]
                a_images = gray2rgb(binarize(grayscale(a_images), threshold=127) * 255)
                src_img = np.concatenate([a_images, b_images], axis=1)
                imsave(src_path, src_img)
            elif action == "p":
                pass
            else:
                shutil.move(src_path, src_path+".cleaned")
                image_i_after_cleaning -= 1
        image_i_after_cleaning += 1

        last_image_i = image_i
        if action == "p":
            image_i -= 1
        elif action== "1" or action == "2":
            pass
        else:
            image_i += 1
        # else:
        #     last_image_i = image_i
        #     image_i += 1


