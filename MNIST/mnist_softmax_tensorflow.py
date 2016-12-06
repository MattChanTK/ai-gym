import tensorflow as tf
from time import clock

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def weight_variable(shape):
    w_initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(w_initial)


def bias_variable(shape):
    b_initial = tf.constant(0.1, shape=shape)
    return tf.Variable(b_initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



x = tf.placeholder(tf.float32, shape=[None, 784])
y_truth = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

# First Convolutional Layer
W_1 = weight_variable([5, 5, 1, 32])
b_1 = bias_variable([32])

conv_1 = tf.nn.relu(conv2d(x_image, W_1))
pool_1 = max_pool_2x2(conv_1)

# Second Convolutional Layer
W_2 = weight_variable([5, 5, 32, 64])
b_2 = bias_variable([64])

conv_2 = tf.nn.relu(conv2d(pool_1, W_2) + b_2)
pool_2 = max_pool_2x2(conv_2)

# Final Convolutional layer
W_f = weight_variable([7*7*64, 1024])
b_f = bias_variable([1024])
pool_2_flat = tf.reshape(pool_2, [-1, 7*7*64])
conv_f = tf.nn.relu(tf.matmul(pool_2_flat, W_f) + b_f)

# Drop out layer
keep_prob = tf.placeholder(tf.float32)
drop_out = tf.nn.dropout(conv_f, keep_prob)

# Output Layer
W_out = weight_variable([1024, 10])
b_out = weight_variable([10])

y = tf.matmul(drop_out, W_out) + b_out


# Error function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_truth))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_truth, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


init_op = tf.initialize_all_variables()
training_start_time = clock()

with tf.Session() as sess:
    sess.run(init_op)

    for i in range(10000):
        batch = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch[0], y_truth: batch[1], keep_prob: 0.5})

        if i%100 == 0:
            t = clock() - training_start_time
            result = sess.run(accuracy, feed_dict={x: batch[0], y_truth: batch[1], keep_prob: 1.0})
            print("(%.2f s) Step %d, training accuracy %g" % (t, i, result))

    test_result = sess.run(accuracy, feed_dict={x: mnist.test.images, y_truth: mnist.test.labels, keep_prob: 1.0})
    print("Test accuracy %g" % (test_result))
