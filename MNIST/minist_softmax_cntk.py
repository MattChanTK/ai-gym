import gzip
import os
import struct
import numpy as np

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

import cntk
from cntk.layers import Convolution, MaxPooling, Dropout, Dense


'''
-----------------------------------------------------
Functions to load or download MNIST images and unpack
into training and testing sets.
-----------------------------------------------------
'''
# loading data from local path if possible. Otherwise download from online sources
def load_or_download_mnist_files(filename, num_samples, local_data_dir):

    if (local_data_dir):
        local_path = os.path.join(local_data_dir, filename)
    else:
        local_path = os.path.join(os.getcwd(), filename)

    if os.path.exists(local_path):
        gzfname = local_path
    else:
        local_data_dir = os.path.dirname(local_path)
        if not os.path.exists(local_data_dir):
            os.makedirs(local_data_dir)
        filename = "http://yann.lecun.com/exdb/mnist/" + filename
        print ("Downloading from" + filename, end=" ")
        gzfname, h = urlretrieve(filename, local_path)
        print ("[Done]")

    return gzfname

def get_mnist_data(filename, num_samples, local_data_dir):

    gzfname = load_or_download_mnist_files(filename, num_samples, local_data_dir)

    with gzip.open(gzfname) as gz:
        n = struct.unpack('I', gz.read(4))
        # Read magic number.
        if n[0] != 0x3080000:
            raise Exception('Invalid file: unexpected magic number.')
        # Read number of entries.
        n = struct.unpack('>I', gz.read(4))[0]
        if n != num_samples:
            raise Exception('Invalid file: expected {0} entries.'.format(num_samples))
        crow = struct.unpack('>I', gz.read(4))[0]
        ccol = struct.unpack('>I', gz.read(4))[0]
        if crow != 28 or ccol != 28:
            raise Exception('Invalid file: expected 28 rows/cols per image.')
        # Read data.
        res = np.fromstring(gz.read(num_samples * crow * ccol), dtype = np.uint8)

        return res.reshape((num_samples, crow * ccol))

# loading labels from local path if possible. Otherwise download from online sources
def get_mnist_labels(filename, num_samples, local_data_dir):

    gzfname = load_or_download_mnist_files(filename, num_samples, local_data_dir)

    with gzip.open(gzfname) as gz:
        n = struct.unpack('I', gz.read(4))
        # Read magic number.
        if n[0] != 0x1080000:
            raise Exception('Invalid file: unexpected magic number.')
        # Read number of entries.
        n = struct.unpack('>I', gz.read(4))
        if n[0] != num_samples:
            raise Exception('Invalid file: expected {0} rows.'.format(num_samples))
        # Read labels.
        res = np.fromstring(gz.read(num_samples), dtype = np.uint8)

        return res.reshape((num_samples, 1))

# loading mnist data and labels
def load_mnist_data(data_filename, labels_filename, number_samples, local_data_dir=None):
    data = get_mnist_data(data_filename, number_samples, local_data_dir)
    labels = get_mnist_labels(labels_filename, number_samples, local_data_dir)
    return np.hstack((data, labels))

# Save the data files into a format compatible with CNTK text reader
def save_as_txt(filename, ndarray):
    dir = os.path.dirname(filename)

    if not os.path.exists(dir):
        os.makedirs(dir)

    if not os.path.isfile(filename):
        print("Saving to ", filename, end=" ")
        with open(filename, 'w') as f:
            labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
            for row in ndarray:
                row_str = row.astype(str)
                label_str = labels[row[-1]]
                feature_str = ' '.join(row_str[:-1])
                f.write('|labels {} |features {}\n'.format(label_str, feature_str))
    else:
        print("File already exists", filename)


'''
--------------------------------------------------
Retrieve and process the training and testing data
--------------------------------------------------
'''
# Ensure we always get the same amount of randomness
np.random.seed(0)

# Define the data dimensions
image_shape = (1, 28, 28)
input_dim = int(np.prod(image_shape, dtype=int))
output_dim = 10

num_train_samples = 60000
num_test_samples = 10000

# The local path where the training and test data might be found or will be downloaded to.
training_data_path = os.path.join(os.getcwd(), "MNIST_data", "Train-28x28_cntk_text.txt")
testing_data_path = os.path.join(os.getcwd(), "MNIST_data", "Test-28x28_cntk_text.txt")

# Download the data if they don't already exist
if not os.path.exists(training_data_path):

    url_train_image = "train-images-idx3-ubyte.gz"
    url_train_labels = "train-labels-idx1-ubyte.gz"

    print("Loading training data")
    saved_data_dir = os.path.join(os.getcwd(), "MNIST_data")
    train = load_mnist_data(url_train_image, url_train_labels, num_train_samples, local_data_dir=saved_data_dir)

    print ("Writing training data text file...")
    save_as_txt(training_data_path, train)
    print("[Done]")

if not os.path.exists(testing_data_path):

    url_test_image = "t10k-images-idx3-ubyte.gz"
    url_test_labels = "t10k-labels-idx1-ubyte.gz"

    print("Loading testing data")
    saved_data_dir = os.path.join(os.getcwd(), "MNIST_data2")
    test = load_mnist_data(url_test_image, url_test_labels, num_test_samples, saved_data_dir)

    print ("Writing testing data text file...")
    save_as_txt(testing_data_path, test)
    print("[Done]")

feature_stream_name = 'features'
labels_stream_name = 'labels'

# Convert to CNTK MinibatchSource
train_minibatch_source = cntk.text_format_minibatch_source(training_data_path, [
    cntk.StreamConfiguration(feature_stream_name, input_dim),
    cntk.StreamConfiguration(labels_stream_name, output_dim)])
training_features = train_minibatch_source[feature_stream_name]
training_labels = train_minibatch_source[labels_stream_name]

print("Training data from file %s successfully read." % training_data_path)

test_minibatch_source = cntk.text_format_minibatch_source(testing_data_path, [
    cntk.StreamConfiguration(feature_stream_name, input_dim),
    cntk.StreamConfiguration(labels_stream_name, output_dim)])
test_features = test_minibatch_source[feature_stream_name]
test_labels = test_minibatch_source[labels_stream_name]

print("Test data from file %s successfully read." % testing_data_path)


'''
---------------------------------------------
Constructing the Convolutional Neural Network
---------------------------------------------
'''
def create_convolutional_neural_network(input_vars, out_dims, dropout_prob=0.0):

    convolutional_layer_1 = Convolution((5, 5), 32, strides=1, activation=cntk.ops.relu, pad=True)(input_vars)
    pooling_layer_1 = MaxPooling((2, 2), strides=(2, 2))(convolutional_layer_1)

    convolutional_layer_2 = Convolution((5, 5), 64, strides=1, activation=cntk.ops.relu, pad=True)(pooling_layer_1)
    pooling_layer_2 = MaxPooling((2, 2), strides=(2, 2))(convolutional_layer_2)

    fully_connected_layer = Dense(1024)(pooling_layer_2)
    dropout_layer = Dropout(dropout_prob)(fully_connected_layer)
    output_layer = Dense(out_dims, activation=None)(dropout_layer)

    return output_layer

# Define the input to the neural network
input_vars = cntk.ops.input_variable(image_shape, np.float32)

# Create the convolutional neural network
output = create_convolutional_neural_network(input_vars, output_dim, dropout_prob=0.5)


'''
----------------------
Setting up the trainer
----------------------
'''
# Define the label as the other input parameter of the trainer
labels = cntk.ops.input_variable(output_dim, np.float32)

#Initialize the parameters for the trainer
train_minibatch_size = 100
learning_rate = 1e-4
momentum = 0.9 ** (1.0 / train_minibatch_size)

# Define the loss function
loss = cntk.ops.cross_entropy_with_softmax(output, labels)

# Define the function that calculates classification error
label_error = cntk.ops.classification_error(output, labels)

# Instantiate the trainer object to drive the model training
learner = cntk.adam_sgd(output.parameters, learning_rate, momentum)
trainer = cntk.Trainer(output, loss, label_error, [learner])


'''
-----------------------------------------
Training the Convolutional Neural Network
-----------------------------------------
'''
num_training_epoch = 5
training_progress_output_freq = 10

for epoch in range(num_training_epoch):

    sample_count = 0
    num_minibatch = 0

    # loop over minibatches in the epoch
    while sample_count < num_train_samples:

        minibatch = train_minibatch_source.next_minibatch(min(train_minibatch_size, num_train_samples - sample_count))

        # Specify the mapping of input variables in the model to actual minibatch data to be trained with
        data = {input_vars: minibatch[training_features],
                labels: minibatch[training_labels]}
        trainer.train_minibatch(data)

        sample_count += data[labels].num_samples
        num_minibatch += 1

        # Print the training progress data
        if num_minibatch % training_progress_output_freq == 0:
            training_loss = cntk.get_train_loss(trainer)
            eval_error = cntk.get_train_eval_criterion(trainer)
            print("Epoch %d  |  # of Samples: %6d  |  Loss: %.6f  |  Error: %.6f" % (epoch, sample_count, training_loss, eval_error))

print("Training Completed.", end="\n\n")


'''
-------------------
Classification Test
--------------------
'''
test_minibatch_size = 1000

sample_count = 0
test_results = []

while sample_count < num_test_samples:

    minibatch = test_minibatch_source.next_minibatch(min(test_minibatch_size, num_test_samples - sample_count))

    # Specify the mapping of input variables in the model to actual minibatch data to be tested with
    data = {input_vars: minibatch[test_features],
            labels: minibatch[test_labels]}
    eval_error = trainer.test_minibatch(data)
    test_results.append(eval_error)

    sample_count += data[labels].num_samples

# Printing the average of evaluation errors of all test minibatches
print("Average errors of all test minibatches: %.3f%%" % (float(np.mean(test_results, dtype=float))*100))
