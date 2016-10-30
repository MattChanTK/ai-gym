# Import the relevant components
import gzip
import os
import struct
import numpy as np
import cntk
from cntk.layers import Convolution, MaxPooling, AveragePooling, Dropout, BatchNormalization, Dense

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


# Functions to load MNIST images and unpack into train and test set.
# - loadData reads image data and formats into a 28x28 long array
# - loadLabels reads the corresponding labels data, 1 for each image
# - load packs the downloaded image and labels data into a combined format to be read later by CNTK text reader

def loadData(src, cimg):
    local_path = os.path.join(os.getcwd(), "MNIST_data", src)
    if os.path.exists(local_path):
        gzfname = local_path
    else:
        src = "http://yann.lecun.com/exdb/mnist/" + src
        print ('Downloading ' + src)
        gzfname, h = urlretrieve(src, local_path)
        print ('Done.')

    with gzip.open(gzfname) as gz:
        n = struct.unpack('I', gz.read(4))
        # Read magic number.
        if n[0] != 0x3080000:
            raise Exception('Invalid file: unexpected magic number.')
        # Read number of entries.
        n = struct.unpack('>I', gz.read(4))[0]
        if n != cimg:
            raise Exception('Invalid file: expected {0} entries.'.format(cimg))
        crow = struct.unpack('>I', gz.read(4))[0]
        ccol = struct.unpack('>I', gz.read(4))[0]
        if crow != 28 or ccol != 28:
            raise Exception('Invalid file: expected 28 rows/cols per image.')
        # Read data.
        res = np.fromstring(gz.read(cimg * crow * ccol), dtype = np.uint8)

        return res.reshape((cimg, crow * ccol))


def loadLabels(src, cimg):
    local_path = os.path.join(os.getcwd(), "MNIST_data", src)
    if os.path.exists(local_path):
        gzfname = local_path
    else:
        src = "http://yann.lecun.com/exdb/mnist/" + src
        print ('Downloading ' + src)
        gzfname, h = urlretrieve(src, local_path)
        print ('Done.')

    with gzip.open(gzfname) as gz:
        n = struct.unpack('I', gz.read(4))
        # Read magic number.
        if n[0] != 0x1080000:
            raise Exception('Invalid file: unexpected magic number.')
        # Read number of entries.
        n = struct.unpack('>I', gz.read(4))
        if n[0] != cimg:
            raise Exception('Invalid file: expected {0} rows.'.format(cimg))
        # Read labels.
        res = np.fromstring(gz.read(cimg), dtype = np.uint8)

        return res.reshape((cimg, 1))


def try_download(dataSrc, labelsSrc, cimg):
    data = loadData(dataSrc, cimg)
    labels = loadLabels(labelsSrc, cimg)
    return np.hstack((data, labels))

# Save the data files into a format compatible with CNTK text reader
def savetxt(filename, ndarray):
    dir = os.path.dirname(filename)

    if not os.path.exists(dir):
        os.makedirs(dir)

    if not os.path.isfile(filename):
        print("Saving", filename)
        with open(filename, 'w') as f:
            labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
            for row in ndarray:
                row_str = row.astype(str)
                label_str = labels[row[-1]]
                feature_str = ' '.join(row_str[:-1])
                f.write('|labels {} |features {}\n'.format(label_str, feature_str))
    else:
        print("File already exists", filename)


# Ensure we always get the same amount of randomness
np.random.seed(0)

# Define the data dimensions
input_dim = 784
output_dim = 10

# Loading the MNIST training and testing data
training_data_path = os.path.join(os.getcwd(), "MNIST_data", "Train-28x28_cntk_text.txt")
testing_data_path = os.path.join(os.getcwd(), "MNIST_data", "Test-28x28_cntk_text.txt")

if not os.path.exists(training_data_path):

    url_train_image = 'train-images-idx3-ubyte.gz'
    url_train_labels = 'train-labels-idx1-ubyte.gz'
    num_train_samples = 60000

    print("Loading train data")
    train = try_download(url_train_image, url_train_labels, num_train_samples)

    print ('Writing train text file...')
    savetxt(training_data_path, train)
    print('Done')

if not os.path.exists(testing_data_path):

    url_test_image = 't10k-images-idx3-ubyte.gz'
    url_test_labels = 't10k-labels-idx1-ubyte.gz'
    num_test_samples = 10000

    print("Loading test data")
    test = try_download(url_test_image, url_test_labels, num_test_samples)

    print ('Writing test text file...')
    savetxt(testing_data_path, test)
    print('Done')

feature_stream_name = 'features'
labels_stream_name = 'labels'

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



image_shape = (1, 28, 28)
input = cntk.ops.input_variable(image_shape, np.float32)
label = cntk.ops.input_variable(output_dim, np.float32)


def create_basic_model(input, out_dims, dropout=0.0):
    net = Convolution((5, 5), 32, init=cntk.initializer.glorot_uniform(scale=0.1557 / 256), activation=cntk.ops.relu, pad=True)(input)
    net = MaxPooling((3, 3), strides=(2, 2))(net)

    net = Convolution((5, 5), 32, init=cntk.initializer.glorot_uniform(scale=0.2), activation=cntk.ops.relu, pad=True)(net)
    net = MaxPooling((3, 3), strides=(2, 2))(net)

    net = Convolution((5, 5), 64, init=cntk.initializer.glorot_uniform(scale=0.2), activation=cntk.ops.relu, pad=True)(net)
    net = MaxPooling((3, 3), strides=(2, 2))(net)

    net = Dense(64, init=cntk.initializer.glorot_uniform(scale=1.697))(net)
    net = Dropout(dropout)(net)
    net = Dense(out_dims, init=cntk.initializer.glorot_uniform(scale=0.212), activation=None)(net)

    return net


# Create the fully connected classfier but first we scale the input to 0-1 range by dividing each pixel by 256.
scaled_input = cntk.ops.element_times(cntk.ops.constant(0.00390625), input)
y = create_basic_model(scaled_input, output_dim, dropout=0.5)

loss = cntk.ops.cross_entropy_with_softmax(y, label)
label_error = cntk.ops.classification_error(y, label)

#Initialize the parameters for the trainer
train_epoch_size = 20000
train_minibatch_size = 64
train_max_epochs = 1

# For basic model
lr_per_sample = [0.00015625]*10+[0.000046875]*10+[0.0000156]
momentum_per_sample = 0
l2_reg_weight = 0.03

# trainer ob# Instantiate the trainer object to drive the model training
lr_schedule = cntk.learning_rate_schedule(lr_per_sample, units=train_epoch_size)
learner = cntk.momentum_sgd(y.parameters,  lr_schedule, momentum_per_sample, l2_regularization_weight = l2_reg_weight)
trainer = cntk.Trainer(y, loss, label_error, [learner])

# Run the trainer on and perform model training
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

# Test data for trained model
test_epoch_size = 10000
test_minibatch_size = 512

sample_count = 0
test_result = 0.0
while sample_count < test_epoch_size:

    minibatch = test_mb_source.next_minibatch(min(test_minibatch_size, test_epoch_size - sample_count))

    # Specify the mapping of input variables in the model to actual
    # minibatch data to be tested with
    data = {input: minibatch[features_test],
            label: minibatch[labels_test]}
    eval_error = trainer.test_minibatch(data)
    test_result = test_result + eval_error

    sample_count += data[label].num_samples

# Average of evaluation errors of all test minibatches
print("Average errors of all test minibaches: {0:.2f}%".format(test_result*100 / test_epoch_size))
