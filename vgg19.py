import os
import tensorflow as tf

import numpy as np
import time
import inspect
import collections

VGG_MEAN_BGR = [103.939, 116.779, 123.68]
VGG_MEAN_RGB = [123.68, 116.779, 103.939]
Model = collections.namedtuple("Model", "loss, outputs, train, accuracy, precision, recall")

class Vgg19:
    """
    A trainable version VGG19.
    """

    def __init__(self, vgg19_npy_path=None, trainable=True):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable

    def build(self, rgb, output_classes, trainable_layer='fc6', train_mode=None, subtract_mean = True):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """
        self.trainable = False
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if subtract_mean:
            bgr = tf.concat(3, [
                blue - VGG_MEAN_BGR[0],
                green - VGG_MEAN_BGR[1],
                red - VGG_MEAN_BGR[2],
            ])
        else:
            bgr = tf.concat(3, [
                blue,
                green,
                red,
            ])
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        height = bgr.get_shape().as_list()[1]
        width = bgr.get_shape().as_list()[2]
        if trainable_layer == "conv1_1":
            self.trainable = True
        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        if trainable_layer == "conv2_1":
            self.trainable = True
        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        if trainable_layer == "conv3_1":
            self.trainable = True
        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        if trainable_layer == "conv4_1":
            self.trainable = True
        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        if trainable_layer == "conv5_1":
            self.trainable = True
        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        # self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 / (2 ** 5)) ** 2) * 512
        # self.relu6 = tf.nn.relu(self.fc6)
        # if train_mode is not None:
        #     self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, 0.5), lambda: self.relu6)
        # elif self.trainable:
        #     self.relu6 = tf.nn.dropout(self.relu6, 0.5)
        #
        # self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        # self.relu7 = tf.nn.relu(self.fc7)
        # if train_mode is not None:
        #     self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, 0.5), lambda: self.relu7)
        # elif self.trainable:
        #     self.relu7 = tf.nn.dropout(self.relu7, 0.5)
        #
        # self.fc8 = self.fc_layer(self.relu7, 4096, output_classes, "fc8")
        #
        # self.prob = tf.nn.softmax(self.fc8, name="prob")

        if output_classes is not None:
            self.trainable = True
            self.fc6 = self.fc_layer(self.pool5, ((height / (2 ** 5)) * (width / (2 ** 5))) * 512, 4096, "fc6")  # 25088 = ((224 / (2 ** 5)) ** 2) * 512
            self.relu6 = tf.nn.relu(self.fc6)
            if train_mode is not None:
                self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, 0.5), lambda: self.relu6)
            elif self.trainable:
                self.relu6 = tf.nn.dropout(self.relu6, 0.5)

            self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
            self.relu7 = tf.nn.relu(self.fc7)
            if train_mode is not None:
                self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, 0.5), lambda: self.relu7)
            elif self.trainable:
                self.relu7 = tf.nn.dropout(self.relu7, 0.5)

            self.fc8 = self.fc_layer(self.relu7, 4096, output_classes, "fc8")

            self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
            value_shape = value.shape
            if list(value.shape) != initial_value.get_shape().as_list():
                print('Warning. Stored variable %s has a different shape than current setting. '
                      'Stored shape is %s while current setting shape is %s. Using current setting.'
                      %(var_name, str(value.shape), str(initial_value.get_shape().as_list())))
                value = initial_value
                value_shape = initial_value.get_shape().as_list()
        else:
            value = initial_value
            value_shape = initial_value.get_shape().as_list()

        if self.trainable:
            # var = tf.Variable(value, name=var_name)
            var = tf.get_variable(name=var_name, initializer=value)
        else:
            # var = tf.Variable(value, name=var_name, trainable=False) # tf.constant(value, dtype=tf.float32, name=var_name)
            var = tf.get_variable(name=var_name, initializer=value, trainable=False)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in self.var_dict.items():
            var_out = sess.run(var)
            if not data_dict.has_key(name):
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in self.var_dict.values():
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
    def net(self):
        return {
            "conv1_1": self.conv1_1,
            "conv1_2": self.conv1_2,
            "conv2_1": self.conv2_1,
            "conv2_2": self.conv2_2,
            "conv3_1": self.conv3_1,
            "conv3_2": self.conv3_2,
            "conv3_3": self.conv3_3,
            "conv3_4": self.conv3_4,
            "conv4_1": self.conv4_1,
            "conv4_2": self.conv4_2,
            "conv4_3": self.conv4_3,
            "conv4_4": self.conv4_4,
            "conv5_1": self.conv5_1,
            "conv5_2": self.conv5_2,
            "conv5_3": self.conv5_3,
            "conv5_4": self.conv5_4,
        }

def create_model(inputs, targets, config):
    def create_classifier(inputs, targets):
        vgg = Vgg19(vgg19_npy_path=config.vgg19_npy_path) # Read model from pretrained vgg
        train_mode = tf.constant(config.mode=='train',dtype=tf.bool, name='train_mode')
        # train_mode = tf.constant(False,dtype=tf.bool, name='train_mode')
        output_classes = targets.get_shape().as_list()[1]
        vgg.build(inputs, output_classes, config.trainable_layer, train_mode)
        vgg_19_net = vgg.fc8
        return vgg_19_net

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("classifier"):
        with tf.variable_scope("classifier"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict = create_classifier(inputs, targets)

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

    with tf.name_scope("train"):
        classifier_tvars = [var for var in tf.trainable_variables() if var.name.startswith("classifier")]
        # classifier_optim = tf.train.AdamOptimizer(config.lr, config.beta1)
        # I am forced to use this optimizer because Adam optimizer creates its own variable which is not suitable for
        # modifying the network for each stage of the training process.
        # TODO: change it back once i'm sure I don't need separate training processes.
        classifier_optim = tf.train.GradientDescentOptimizer(config.lr)
        # classifier_train = classifier_optim.minimize(loss, var_list=classifier_tvars)
        classifier_train = classifier_optim.minimize(loss)


    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([loss,accuracy,precision, recall])
    # update_losses = ema.apply([loss,])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        loss=ema.average(loss),
        outputs=predict,
        train=tf.group(update_losses, incr_global_step, classifier_train),
        accuracy= ema.average(accuracy), # If this does not work, take out the ema.
        precision= ema.average(precision),
        recall= ema.average(recall),
    )

def preprocess(image, mean_pixel =VGG_MEAN_RGB):
    return image - mean_pixel

def unprocess(image, mean_pixel = VGG_MEAN_RGB):
    return image + mean_pixel