# Import the relevant components
import gzip
import os
import struct
import cntk
from cntk.ops import *

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


num_hidden_layers = 2
hidden_layers_dim = 400

input = input_variable(input_dim, np.float32)
label = input_variable(output_dim, np.float32)

# Define a fully connected feedforward network
def linear_layer(input_var, output_dim):

    input_dim = input_var.shape[0]
    times_param = parameter(shape=(input_dim, output_dim), init=cntk.initializer.glorot_uniform())
    bias_param = parameter(shape=(output_dim))

    t = times(input_var, times_param)
    return bias_param + t

def dense_layer(input, output_dim, nonlinearity):
    r = linear_layer(input, output_dim)
    r = nonlinearity(r)
    return r

def fully_connected_classifier_net(input, num_output_classes, hidden_layer_dim, num_hidden_layers, nonlinearity):
    h = dense_layer(input, hidden_layer_dim, nonlinearity)
    for i in range(1, num_hidden_layers):
        h = dense_layer(h, hidden_layer_dim, nonlinearity)
    r = linear_layer(h, num_output_classes)
    return r

# Create the fully connected classfier but first we scale the input to 0-1 range by dividing each pixel by 256.
scaled_input = element_times(constant(0.00390625), input)
y = fully_connected_classifier_net(scaled_input, output_dim, hidden_layers_dim, num_hidden_layers, relu)

loss = cntk.ops.cross_entropy_with_softmax(y, label)
label_error = cntk.ops.classification_error(y, label)

# Instantiate the trainer object to drive the model training
learning_rate_per_sample = 0.003125
learner = cntk.sgd(y.parameters, lr=learning_rate_per_sample)
trainer = cntk.Trainer(y, loss, label_error, [learner])

#Initialize the parameters for the trainer
minibatch_size = 64
num_samples_per_sweep = 60000
num_sweeps_to_train_with = 10
num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

# Run the trainer on and perform model training
training_progress_output_freq = 500

for i in range(0, int(num_minibatches_to_train)):
    mb = train_mb_source.next_minibatch(minibatch_size)

    # Specify the mapping of input variables in the model to actual minibatch data to be trained with
    arguments = {input: mb[features_train],
                 label: mb[labels_train]}
    trainer.train_minibatch(arguments)

    # Print the training progress data
    if i % training_progress_output_freq == 0:
        training_loss = cntk.get_train_loss(trainer)
        eval_error = cntk.get_train_eval_criterion(trainer)
        print("%d - Loss: %f   Error: %f" % (i, training_loss, eval_error))

# Evaluate on test data
# Test data for trained model
test_minibatch_size = 512
num_samples = 10000
num_minibatches_to_test = num_samples / test_minibatch_size
test_result = 0.0
for i in range(0, int(num_minibatches_to_test)):
    mb = test_mb_source.next_minibatch(test_minibatch_size)

    # Specify the mapping of input variables in the model to actual
    # minibatch data to be tested with
    arguments = {input: mb[features_test],
                 label: mb[labels_test]}
    eval_error = trainer.test_minibatch(arguments)
    test_result = test_result + eval_error

# Average of evaluation errors of all test minibatches
print("Average errors of all test minibaches: {0:.2f}%".format(test_result*100 / num_minibatches_to_test))
