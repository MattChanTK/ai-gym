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
input_dim = 28*28
output_dim = 10

num_train_samples = 60000
num_test_samples = 10000

# Loading the MNIST training and testing data
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
train_mb_source = cntk.text_format_minibatch_source(training_data_path, [
    cntk.StreamConfiguration(feature_stream_name, input_dim),
    cntk.StreamConfiguration(labels_stream_name, output_dim)])
features_train = train_mb_source[feature_stream_name]
labels_train = train_mb_source[labels_stream_name]

print("Training data from file {0} successfully read.".format(training_data_path))

test_mb_source = cntk.text_format_minibatch_source(testing_data_path, [
    cntk.StreamConfiguration(feature_stream_name, input_dim),
    cntk.StreamConfiguration(labels_stream_name, output_dim)])
features_test = test_mb_source[feature_stream_name]
labels_test = test_mb_source[labels_stream_name]

print("Test data from file {0} successfully read".format(testing_data_path))


'''
---------------------------------------------
Constructing the Convolutional Neural Network
---------------------------------------------
'''
image_shape = (1, 28, 28)
input = cntk.ops.input_variable(image_shape, np.float32)
label = cntk.ops.input_variable(output_dim, np.float32)

def create_convolutional_neural_network(input, out_dims, dropout_prob=0.0):

    convolutional_layer_1 = Convolution((5, 5), 32, activation=cntk.ops.relu, pad=True)(input)
    pooling_layer_1 = MaxPooling((2, 2), strides=(2, 2))(convolutional_layer_1)

    convolutional_layer_2 = Convolution((5, 5), 64, activation=cntk.ops.relu, pad=True)(pooling_layer_1)
    pooling_layer_2 = MaxPooling((2, 2), strides=(2, 2))(convolutional_layer_2)

    fully_connected_layer = Dense(1024)(pooling_layer_2)
    dropout_layer = Dropout(dropout_prob)(fully_connected_layer)
    output_layer = Dense(out_dims, activation=None)(dropout_layer)

    return output_layer

# Scale the input to 0-1 range by dividing each pixel by 256
scaled_input = cntk.ops.element_times(cntk.ops.constant(float(1/256)), input)

# Create the convolutional neural network
output = create_convolutional_neural_network(scaled_input, output_dim, dropout_prob=0.5)


'''
----------------------
Setting up the trainer
----------------------
'''
#Initialize the parameters for the trainer
train_minibatch_size = 64
learning_rate = 1e-4
momentum = 0.9 ** (1.0 / train_minibatch_size)

# Define the loss function
loss = cntk.ops.cross_entropy_with_softmax(output, label)

# Define the function that calculates classification error
label_error = cntk.ops.classification_error(output, label)

# Instantiate the trainer object to drive the model training
learner = cntk.adam_sgd(output.parameters, learning_rate, momentum)
trainer = cntk.Trainer(output, loss, label_error, [learner])


'''
-----------------------------------------
Training the Convolutional Neural Network
-----------------------------------------
'''
train_epoch_size = num_train_samples
train_max_epochs = 1
training_progress_output_freq = 10

# loop over epochs
for epoch in range(train_max_epochs):
    sample_count = 0
    num_minibatch = 0

    # loop over minibatches in the epoch
    while sample_count < train_epoch_size:

        minibatch = train_mb_source.next_minibatch(min(train_minibatch_size, train_epoch_size - sample_count))

        # Specify the mapping of input variables in the model to actual minibatch data to be trained with
        data = {input: minibatch[features_train],
                label: minibatch[labels_train]}
        trainer.train_minibatch(data)

        sample_count += data[label].num_samples
        num_minibatch += 1

        # Print the training progress data
        if num_minibatch % training_progress_output_freq == 0:
            training_loss = cntk.get_train_loss(trainer)
            eval_error = cntk.get_train_eval_criterion(trainer)
            print("# Samples: %d - Loss: %f   Error: %f" % (sample_count, training_loss, eval_error))

print("Training Completed.")


'''
-------------------
Classification Test
--------------------
'''
test_epoch_size = num_test_samples
test_minibatch_size = 512

sample_count = 0
test_result = 0.0
while sample_count < test_epoch_size:

    minibatch = test_mb_source.next_minibatch(min(test_minibatch_size, test_epoch_size - sample_count))

    # Specify the mapping of input variables in the model to actual minibatch data to be tested with
    data = {input: minibatch[features_test],
            label: minibatch[labels_test]}
    eval_error = trainer.test_minibatch(data)
    test_result = test_result + eval_error

    sample_count += data[label].num_samples

# Printing the average of evaluation errors of all test minibatches
print("\nAverage errors of all test minibaches: {0:.6f}%".format(test_result*100 / test_epoch_size))
