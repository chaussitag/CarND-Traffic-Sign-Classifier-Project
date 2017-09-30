#!/usr/bin/env python
# coding=utf8

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
from tensorflow.contrib import layers
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler


def nomalize_images(images_data):
    # normalize each pixel value to [-1, 1]
    norm_data = (images_data - 128.0) / 128.0
    return norm_data.astype(np.float32)


def rgb_to_gray(rgb_image_batch):
    # Y = 0.2125 R + 0.7154 G + 0.0721 B
    gray_images = 0.2125 * rgb_image_batch[:, :, :, 0] + \
                  0.7154 * rgb_image_batch[:, :, :, 1] + \
                  0.0721 * rgb_image_batch[:, :, :, 2]
    gray_images = np.expand_dims(gray_images, axis=3)
    return gray_images.astype(np.float32)


class Dataset(object):
    def __init__(self, training_file_path, validation_file_path, testing_file_path):
        assert os.path.isfile(training_file_path), "training file %s not exist" % training_file_path
        assert os.path.isfile(validation_file_path), "validation file %s not exist" % validation_file_path
        assert os.path.isfile(testing_file_path), "testing file %s not exist" % testing_file_path
        # Load the data
        with open(training_file_path, mode='rb') as f:
            train_data = pickle.load(f)
            self._train_images = train_data['features']
            self._train_labels = train_data['labels']
            self._classes, uniq_train_imgs_incides, self._origin_train_labels_counts = np.unique(self._train_labels,
                                                                                                 return_index=True,
                                                                                                 return_counts=True)
            self._resampled_train_labels_counts = self._origin_train_labels_counts
            self._class_images = self._train_images[uniq_train_imgs_incides].copy()
            self._num_classes = len(self._classes)
        with open(validation_file_path, mode='rb') as f:
            validation_data = pickle.load(f)
            self._validation_images = validation_data['features']
            self._validation_labels = validation_data['labels']
            _, self._validation_labels_counts = np.unique(self._validation_labels, return_counts=True)
        with open(testing_file_path, mode='rb') as f:
            test_data = pickle.load(f)
            self._test_images = test_data['features']
            self._test_labels = test_data['labels']
            _, self._test_labels_counts = np.unique(self._test_labels, return_counts=True)

            # # one-hot the labels
            # lb = LabelBinarizer()
            # lb.fit(range(0, self._num_classes))
            # self._train_labels = lb.transform(self._train_labels).astype(np.float32)
            # self._validation_labels = lb.transform(self._validation_labels).astype(np.float32)
            # self._test_labels = lb.transform(self._test_labels).astype(np.float32)

    def preprocess(self):
        # convert to gray image
        self._train_images = rgb_to_gray(self._train_images)
        self._validation_images = rgb_to_gray(self._validation_images)
        self._test_images = rgb_to_gray(self._test_images)

        # normalize pixel value to [-1, 1]
        self._train_images = nomalize_images(self._train_images)
        self._validation_images = nomalize_images(self._validation_images)
        self._test_images = nomalize_images(self._test_images)

        # resample
        resampler = SMOTE()
        img_h, img_w, img_depth = self._train_images.shape[1:4]
        self._train_images = self._train_images.reshape(self.num_train_image, -1)
        print("78 self._train_images.shape %s" % (self._train_images.shape,))
        print("79 self._train_labels.shape %s" % (self._train_labels.shape,))
        self._train_images, self._train_labels = resampler.fit_sample(self._train_images, self._train_labels)
        self._train_images = self._train_images.reshape(-1, img_h, img_w, img_depth)

        # update the counts for each class
        _, self._resampled_train_labels_counts = np.unique(self._train_labels, return_counts=True)

        # one-hot the labels
        lb = LabelBinarizer()
        lb.fit(range(0, self._num_classes))
        self._train_labels = lb.transform(self._train_labels).astype(np.float32)
        self._validation_labels = lb.transform(self._validation_labels).astype(np.float32)
        self._test_labels = lb.transform(self._test_labels).astype(np.float32)

    def shuffle_train_data(self):
        # randomly shuffle the training set
        self._train_images, self._train_labels = shuffle(self.train_images, self.train_labels)

    @property
    def origin_train_labels_counts(self):
        return self._origin_train_labels_counts

    @property
    def resampled_train_labels_counts(self):
        return self._resampled_train_labels_counts

    @property
    def validation_labels_counts(self):
        return self._validation_labels_counts

    @property
    def test_labels_counts(self):
        return self._test_labels_counts

    @property
    def test_labels_count(self):
        return self

    @property
    def classes(self):
        return self._classes

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_train_image(self):
        return self._train_images.shape[0]

    @property
    def num_validation_image(self):
        return self._validation_images.shape[0]

    @property
    def num_test_image(self):
        return self._test_images.shape[0]

    @property
    def image_height(self):
        return self._train_images.shape[1]

    @property
    def image_width(self):
        return self._train_images.shape[2]

    @property
    def image_depth(self):
        return self._train_images.shape[3]

    @property
    def train_images(self):
        return self._train_images

    @property
    def train_labels(self):
        return self._train_labels

    @property
    def validation_images(self):
        return self._validation_images

    @property
    def validation_labels(self):
        return self._validation_labels

    @property
    def test_images(self):
        return self._test_images

    @property
    def test_labels(self):
        return self._test_labels

    @property
    def class_images(self):
        """

        return exactly one image for each class in a list, used only for debug, in rbg format
        :return: a list contains exactly one image for each class
        """
        return self._class_images

    def get_random_train_image(self):
        index = np.random.randint(0, self.num_train_image)
        img = self._train_images[index]
        return img


data_set = Dataset("dataset/train.p", "dataset/valid.p", "dataset/test.p")
# preprocess the dataset
data_set.preprocess()

print("Number of training examples =", data_set.num_train_image)
print("Number of validating examples =", data_set.num_validation_image)
print("Number of testing examples =", data_set.num_test_image)
print("Image data shape = [%d, %d, %d]" % (data_set.image_height, data_set.image_width, data_set.image_depth))
print("Number of classes =", data_set.num_classes)
print("classes: %s" % data_set.classes)

### Data exploration visualization.
class_images = data_set.class_images
num_of_class_images = len(class_images)
img_per_row = 10
num_of_plot_rows = (num_of_class_images // img_per_row) + 1
fig = plt.gcf()
fig_default_size = fig.get_size_inches()
fig.set_size_inches(fig_default_size[0] * 2.5, fig_default_size[1] * 2.5)
for plot_index, img in enumerate(class_images):
    axes = fig.add_subplot(num_of_plot_rows, img_per_row, plot_index + 1)
    axes.imshow(img)

# get counts of each trafic sign from the training set
counts_for_train_origin = data_set.origin_train_labels_counts
counts_for_train_resampled = data_set.resampled_train_labels_counts
counts_for_validation = data_set.validation_labels_counts
counts_for_test = data_set.test_labels_counts
# draw the counts of each trafic sign for all of the three data splits
fig1, axes1 = plt.subplots(1)
fig1_default_size = fig1.get_size_inches()
fig1.set_size_inches(fig1_default_size[0] * 2.5, fig1_default_size[1] * 2.5)
axes1.set_title("count of each sign in the training set")
axes1.set_yticks(range(0, np.max(counts_for_train_origin), 100))
axes1.set_xticks(range(0, 4 * data_set.num_classes, 4))
axes1.set_xticklabels(range(0, data_set.num_classes))
bar_width = 0.8
x_indices = data_set.classes * 4
rects_train_resampled = axes1.bar(x_indices, counts_for_train_resampled, width=bar_width, color='y')
rects_train_origin = axes1.bar(x_indices + bar_width, counts_for_train_origin, width=bar_width, color='r')
rects_valid = axes1.bar(x_indices + 2 * bar_width, counts_for_validation, width=bar_width, color='g')
rects_test = axes1.bar(x_indices + 3 * bar_width, counts_for_test, width=bar_width, color='b')
axes1.legend((rects_train_resampled[0], rects_train_origin[0], rects_valid[0], rects_test[0]),
             ('resampled-train-set', 'original-train-set', 'validation-set', 'test-set'))

plt.show()


### the model
def LeNet5(input_x, num_input_img_channel, num_cls, mu=0.0, sigma=0.1, regularizer_factor=1.0e-3,
           keep_prob=None):
    regularizer_ = layers.l2_regularizer(regularizer_factor)
    # use the L2-norm to regularize all parameters
    with tf.variable_scope("lenet", regularizer=regularizer_,
                           initializer=tf.truncated_normal_initializer(mean=mu, stddev=sigma)):
        # Layer 1: convolutional layer, 32x32xnum_input_img_channel ==> 28x28x6
        with tf.variable_scope("conv1"):
            conv1_w = tf.get_variable("conv1_w", [5, 5, num_input_img_channel, 6])
            conv1_b = tf.get_variable("conv1_b", [6], initializer=tf.zeros_initializer)
            conv1 = tf.nn.conv2d(input_x, conv1_w, strides=[1, 1, 1, 1], padding="VALID")
            conv1 = tf.nn.bias_add(conv1, conv1_b)
            # activation
            conv1 = tf.nn.relu(conv1)

        # max pooling layer 1: 28x28x4 ==> 14x14x6
        max_pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        # Layer 2: convolutional layer, 14x14x6 ==> 10x10x16
        with tf.variable_scope("conv2"):
            conv2_w = tf.get_variable("conv2_w", [5, 5, 6, 16])
            conv2_b = tf.get_variable("conv2_b", [16], initializer=tf.zeros_initializer)
            conv2 = tf.nn.conv2d(max_pool_1, conv2_w, strides=[1, 1, 1, 1], padding="VALID")
            conv2 = tf.nn.bias_add(conv2, conv2_b)
            # activation
            conv2 = tf.nn.relu(conv2)

        # max pooling layer 2: 10x10x16 ==> 5x5x16
        max_pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        # flatten: 5x5x16 ==> 400
        max_pool_2_flattened = flatten(max_pool_2)

        # Layer 3: fully connected layer, 400 ==> 120
        with tf.variable_scope("fc1"):
            fc1_w = tf.get_variable("fc1_w", [400, 120])
            fc1_b = tf.get_variable("fc1_b", [120], initializer=tf.zeros_initializer)
            fc1 = tf.add(tf.matmul(max_pool_2_flattened, fc1_w), fc1_b)
            # activation
            fc1 = tf.nn.relu(fc1)
            # dropout
            if keep_prob is not None:
                fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

        # Layer 4: full connected layer, 120 ==> 84
        with tf.variable_scope("fc2"):
            fc2_w = tf.get_variable("fc2_w", [120, 84])
            fc2_b = tf.get_variable("fc2_b", [84], initializer=tf.zeros_initializer)
            fc2 = tf.add(tf.matmul(fc1, fc2_w), fc2_b)
            # activation
            fc2 = tf.nn.relu(fc2)
            # dropout
            if keep_prob is not None:
                fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob)

        # Layer 5: fully connected layer, 84 ==> num_cls, no activation
        with tf.variable_scope("fc3"):
            fc3_w = tf.get_variable("fc3_w", [84, num_cls])
            fc3_b = tf.get_variable("fc3_b", [num_cls], initializer=tf.zeros_initializer)
            logits_ = tf.add(tf.matmul(fc2, fc3_w), fc3_b)

    return logits_


### the model
def LeNet5_ext(input_x, num_input_img_channel, num_cls, mu=0.0, sigma=0.1, regularizer_factor=1.0e-3,
           keep_prob=None):
    regularizer_ = layers.l2_regularizer(regularizer_factor)
    # use the L2-norm to regularize all parameters
    with tf.variable_scope("lenet_ext", regularizer=regularizer_,
                           initializer=tf.truncated_normal_initializer(mean=mu, stddev=sigma)):
        # Layer 1: convolutional layer, 32x32xnum_input_img_channel ==> 28x28x6
        with tf.variable_scope("conv1"):
            conv1_w = tf.get_variable("conv1_w", [5, 5, num_input_img_channel, 6])
            conv1_b = tf.get_variable("conv1_b", [6], initializer=tf.zeros_initializer)
            conv1 = tf.nn.conv2d(input_x, conv1_w, strides=[1, 1, 1, 1], padding="VALID")
            conv1 = tf.nn.bias_add(conv1, conv1_b)
            # activation
            conv1 = tf.nn.relu(conv1)

        # Layer 2: convolutional layer, 28x28x6 ==> 24x24x12
        with tf.variable_scope("conv2"):
            conv2_w = tf.get_variable("conv2_w", [5, 5, 6, 12])
            conv2_b = tf.get_variable("conv2_b", [12], initializer=tf.zeros_initializer)
            conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1, 1, 1, 1], padding='VALID')
            conv2 = tf.nn.bias_add(conv2, conv2_b)
            # activation
            conv2 = tf.nn.relu(conv2)

        # Layer 3: max pooling layer, 24x24x12 ==> 12x12x12
        max_pool_1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        # Layer 4: convolutional layer, 12x12x12 ==> 8x8x16
        with tf.variable_scope("conv3"):
            conv3_w = tf.get_variable("conv3_w", [5, 5, 12, 16])
            conv3_b = tf.get_variable("conv3_b", [16], initializer=tf.zeros_initializer)
            conv3 = tf.nn.conv2d(max_pool_1, conv3_w, strides=[1, 1, 1, 1], padding="VALID")
            conv3 = tf.nn.bias_add(conv3, conv3_b)
            # activation
            conv3 = tf.nn.relu(conv3)

        # Layer 5: max pooling layer, 8x8x16 ==> 4x4x16
        max_pool_2 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        # flatten: 4x4x16 ==> 256
        max_pool_2_flattened = flatten(max_pool_2)

        # Layer 6: fully connected layer, 256 ==> 120
        with tf.variable_scope("fc1"):
            fc1_w = tf.get_variable("fc1_w", [256, 120])
            fc1_b = tf.get_variable("fc1_b", [120], initializer=tf.zeros_initializer)
            fc1 = tf.add(tf.matmul(max_pool_2_flattened, fc1_w), fc1_b)
            # activation
            fc1 = tf.nn.relu(fc1)
            # dropout
            if keep_prob is not None:
                fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

        # Layer 7: full connected layer, 120 ==> 84
        with tf.variable_scope("fc2"):
            fc2_w = tf.get_variable("fc2_w", [120, 84])
            fc2_b = tf.get_variable("fc2_b", [84], initializer=tf.zeros_initializer)
            fc2 = tf.add(tf.matmul(fc1, fc2_w), fc2_b)
            # activation
            fc2 = tf.nn.relu(fc2)
            # dropout
            if keep_prob is not None:
                fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob)

        # Layer 8: fully connected layer, 84 ==> num_cls, no activation
        with tf.variable_scope("fc3"):
            fc3_w = tf.get_variable("fc3_w", [84, num_cls])
            fc3_b = tf.get_variable("fc3_b", [num_cls], initializer=tf.zeros_initializer)
            logits_ = tf.add(tf.matmul(fc2, fc3_w), fc3_b)

    return logits_

# learning rate
BASE_LEARNING_RATE = 0.01
LEARNING_RATE_DECAY_BASE = 0.96
# batch_size
BATCH_SIZE = 512
# epochs
EPOCHS = 120
#EPOCHS = 1
# keep probablity for dropout during training
KEEP_PROB = 0.80

num_train_images = data_set.num_train_image

x = tf.placeholder(tf.float32, [None, data_set.image_height, data_set.image_width, data_set.image_depth])
y = tf.placeholder(tf.float32, [None, data_set.num_classes])
dropout_keep_prob = tf.placeholder(tf.float32)

# logits = LeNet5(x, data_set.image_depth, data_set.num_classes, keep_prob=dropout_keep_prob)
logits = LeNet5_ext(x, data_set.image_depth, data_set.num_classes, keep_prob=dropout_keep_prob)

cross_entropy_loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
regularization_loss_op = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
loss_op = cross_entropy_loss_op + regularization_loss_op

# set up a variable that's incremented once per batch and controls the learning rate decay.
global_step = tf.Variable(0.0, dtype=tf.float32)
# Decay the learning rate once per two epochs, using an exponential schedule starting at 0.01.
learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE, global_step,
                                           2 * num_train_images, LEARNING_RATE_DECAY_BASE, staircase=True)

training_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(images, one_hot_labels):
    num_images_ = len(images)
    total_accuracy_ = 0.0
    sess_ = tf.get_default_session()
    for offset_ in range(0, num_images_, BATCH_SIZE):
        batch_images_ = images[offset_:offset_ + BATCH_SIZE]
        batch_labels_ = one_hot_labels[offset_:offset_ + BATCH_SIZE]
        accuracy_ = sess_.run(accuracy_op, feed_dict={x: batch_images_, y: batch_labels_, dropout_keep_prob: 1.0})
        total_accuracy_ += (accuracy_ * len(batch_images_))
    return total_accuracy_ / num_images_


# a list to store validation accuracy for each epoch
validation_accuracy_list = []
saver = tf.train.Saver()
# train the model
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for epoch in range(0, EPOCHS):
        print("==========================================================================================")
        print("epoch %d:" % epoch)
        data_set.shuffle_train_data()
        batch_num = 0
        for offset in range(0, num_train_images, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_images = data_set.train_images[offset:end]
            batch_labels = data_set.train_labels[offset:end]
            accuracy_value, total_loss_value, regularization_loss_value, _ = \
                session.run([accuracy_op, loss_op, regularization_loss_op, training_op],
                            feed_dict={x: batch_images, y: batch_labels, dropout_keep_prob: KEEP_PROB})
            if batch_num % 100 == 0:
                print("    training step %3d: accuracy %6f, total-loss %6f, regularization-loss %f" \
                      % (batch_num, accuracy_value, total_loss_value, regularization_loss_value))
            batch_num += 1

        valid_accuracy_value = evaluate(data_set.validation_images, data_set.validation_labels)
        validation_accuracy_list.append(valid_accuracy_value)
        print("    validation accuracy %6f" % valid_accuracy_value)
        print("==========================================================================================")

    # after training, save the model
    saver.save(session, "./lenet")
    print("model saved!")

# draw the validation accuracy
fig2, axes2 = plt.subplots(1)
fig2_default_size = fig2.get_size_inches()
fig2.set_size_inches(fig2_default_size[0] * 2.5, fig2_default_size[1] * 2.5)
axes2.set_title("validation accuracy for all epochs")
axes2.set_xlabel("epoch")
axes2.set_ylabel("validation accuracy")
axes2.set_xticks(range(0, EPOCHS), 2)
y_ticks = np.arange(np.min(validation_accuracy_list), np.max(validation_accuracy_list), 0.002)
axes2.set_yticks(y_ticks)
axes2.plot(range(0, EPOCHS), validation_accuracy_list)
plt.grid(True)
plt.show()

with tf.Session() as session:
    saver.restore(session, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(data_set.test_images, data_set.test_labels)
    print("test Accuracy %f" % test_accuracy)
