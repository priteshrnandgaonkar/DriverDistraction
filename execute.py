import load_data
import classification
from classification import Network
from classification import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from random import shuffle
import cPickle as pickle

(image_name_list, image_matrix_list, classification_list) = load_data.load_training_data()
data = zip(image_name_list,image_matrix_list, classification_list)
shuffle(data)
shuffled_data = zip(*data)
data_string = pickle.dumps(shuffled_data)
f = open("processed_data.txt","w")
f.write(data_string)
f.close()
training_data = data[:20000]
vdata = data[20000:]
test_data= vdata[:5000]
validation_data = vdata[5000:]
mini_batch_size = 10
net = Network([FullyConnectedLayer(n_in=784, n_out=100), SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)