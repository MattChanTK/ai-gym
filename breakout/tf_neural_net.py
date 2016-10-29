import tensorflow as tf
import numpy as np


class PixelsCNN(object):

    def __init__(self, image_size, num_outputs, num_frames=1, num_samples=None):

        if not isinstance(image_size, (tuple, list, np.ndarray)) and len(image_size) == 2:
            raise TypeError("image_size must be an array of 2 dimension.")
        self.__image_size = np.asarray(image_size)

        if not isinstance(num_outputs, int):
            raise TypeError("num_outputs must be an integer.")
        self.__num_outputs = num_outputs

        if not isinstance(num_frames, int):
            raise TypeError("num_frames must be an integer.")
        self.__num_frames = num_frames

        if num_samples and not isinstance(num_samples, int):
            raise TypeError("num_samples must be an integer.")
        self.__num_samples = num_samples


    def create_network(self):

        # defining the input layer
        input_layer = tf.placeholder(tf.float32, shape=[self.num_samples, self.image_width, self.image_height, self.num_frames])

        # first convolutional layer
        patch_size_1 = (8, 8)
        num_features_1 = 32
        stride_size_1 = (1, 4, 4, 1)
        pooling_stride_size_1 = (1, 2, 2, 1)
        pooling_ksize_1 = (1, 2, 2, 1)

        weights_1  = self.weight_variable(patch_size_1 + (self.num_frames, num_features_1))
        bias_1 = self.bias_variable((num_features_1,))
        hidden_layer_1 = tf.nn.relu(self.convolution_2d(input_layer, weights_1, strides=stride_size_1) + bias_1)
        pooling_1 = self.max_pool(hidden_layer_1, ksize=pooling_ksize_1, strides=pooling_stride_size_1)

        # second convolutional layer
        patch_size_2 = (4, 4)
        num_features_2 = 64
        stride_size_2 = (1, 2, 2, 1)
        pooling_stride_size_2 = (1, 2, 2, 1)
        pooling_ksize_2 = (1, 2, 2, 1)

        weights_2  = self.weight_variable(patch_size_2 + (num_features_1, num_features_2))
        bias_2 = self.bias_variable((num_features_2,))
        hidden_layer_2 = tf.nn.relu(self.convolution_2d(pooling_1, weights_2, strides=stride_size_2) + bias_2)
        pooling_2 = self.max_pool(hidden_layer_2, ksize=pooling_ksize_2, strides=pooling_stride_size_2)

        # third convolutional layer
        patch_size_3 = (3, 3)
        num_features_3 = 64
        stride_size_3 = (1, 1, 1, 1)
        pooling_stride_size_3 = (1, 2, 2, 1)
        pooling_ksize_3 = (1, 2, 2, 1)

        weights_3 = self.weight_variable(patch_size_3 + (num_features_2, num_features_3))
        bias_3 = self.bias_variable((num_features_3,))
        hidden_layer_3 = tf.nn.relu(self.convolution_2d(pooling_2, weights_3, strides=stride_size_3) + bias_3)
        pooling_3 = self.max_pool(hidden_layer_3, ksize=pooling_ksize_3, strides=pooling_stride_size_3)

        # the final densely connected layer
        reduction_1 = stride_size_1[1] * stride_size_1[2]
        reduction_2 = stride_size_2[1] * stride_size_2[2]
        reduction_3 = stride_size_3[1] * stride_size_3[2]
        num_features_final = 1024
        reduced_image_size = int(self.image_width * self.image_height  / (reduction_1 * reduction_2 * reduction_3))
        num_features_3_flatten = reduced_image_size * self.num_frames * num_features_3

        weights_final = self.weight_variable([num_features_3_flatten, num_features_final])
        bias_final = self.bias_variable((num_features_final, ))
        pooling_3_flat = tf.reshape(pooling_3, [-1, num_features_3_flatten])
        hidden_layer_final = tf.nn.relu(tf.matmul(pooling_3_flat, weights_final) + bias_final)

        # drop out step to prevent overfitting during the training stage
        dropout_keep_prob = tf.placeholder(tf.float32)
        hidden_layer_final_with_dropout = tf.nn.dropout(hidden_layer_final, dropout_keep_prob)

        # the output layer
        weight_output = self.weight_variable([num_features_final, self.num_outputs])
        bias_output = self.bias_variable([self.num_outputs])
        output_layer = tf.matmul(hidden_layer_final_with_dropout, weight_output) + bias_output

        return input_layer, output_layer

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def convolution_2d(x, W, strides=(1, 1, 1, 1)):
        return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

    @staticmethod
    def max_pool(x, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1)):
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding='SAME')

    @property
    def image_size(self):
        return tuple(self.__image_size)

    @property
    def image_width(self):
        return int(self.image_size[0])

    @property
    def image_height(self):
        return int(self.image_size[1])

    @property
    def num_frames(self):
        return self.__num_frames

    @property
    def num_samples(self):
        return self.__num_samples

    @property
    def num_outputs(self):
        return self.__num_outputs
