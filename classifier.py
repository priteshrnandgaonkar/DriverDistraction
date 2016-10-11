from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
from keras.utils import np_utils
import load_data
from random import shuffle
import cPickle as pickle

np.random.seed(1337)  # for reproducibility

batch_size = 50
num_epochs = 30
nb_classes = 10

# (image_name_list, image_matrix_list, classification_list) = load_data.load_training_data()
# data = zip(image_name_list, image_matrix_list, classification_list)
# shuffle(data)
# suffled_data = zip(*data)
f = open("processed_data.txt","r")
data = pickle.loads(f.read())
f.close()
print "data unpacked"
shuffled_image_name_list = np.array(data[0])
shuffled_image_matrix_list = np.array(data[1])
shuffled_image_matrix_list = shuffled_image_matrix_list.astype('float32')
shuffled_image_matrix_list /= 255
shuffled_classification_list = np.array(data[2])
# train_data = (shuffled_image_name_list[:15000], shuffled_image_matrix_list[:15000], shuffled_classification_list[:15000])
train_data = (shuffled_image_name_list[:-1000], shuffled_image_matrix_list[:-1000], shuffled_classification_list[:-1000])
validation_data = (shuffled_image_name_list[-1000:], shuffled_image_matrix_list[-1000:], shuffled_classification_list[-1000:])

# validation_data = (shuffled_image_name_list[15000:-2000], shuffled_image_matrix_list[15000:-2000], shuffled_classification_list[15000:-2000])
# test_data = (shuffled_image_name_list[-2000:], shuffled_image_matrix_list[-2000:], shuffled_classification_list[-2000:])

# print("Train Data" + len(train_data[1]) + "Validation data" + len(validation_data[1]))
print "data training begins"
model = Sequential()
model.add(Dense(500, input_shape=(len(shuffled_image_matrix_list[0]),), init='glorot_uniform'))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(500))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))


sgd = SGD(lr=0.01, delay=1e-6)
# Configure the learning process.
model.compile(loss='categorical_crossentropy', optimizer=sgd)


#Lets learn
model.fit(train_data[1], train_data[2], batch_size=batch_size, nb_epoch=num_epochs, show_accuracy=True, verbose=2, validation_data=(validation_data[1], validation_data[2]))
#Lets Evaluate
# score = model.evaluate(test_data[1], test_data[2], show_accuracy=True, verbose=0)

test_file = open("processed_test_data.txt","r")
t_data = pickle.loads(test_file.read())
test_file.close()
td = np.array(t_data[1])
td_name = t_data[0]
td = td.astype('float32')
td = td / 255
predictions = model.predict(td, batch_size=batch_size, verbose=2)
pred_list = predictions.tolist()
td_name.insert(0,"img")
pred_header = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
pred_list.insert(0,pred_header)
# final_pred =  np.vstack((pred_header,predictions))
output_data = np.column_stack([td_name,pred_list])
np.savetxt('driver_output.csv',output_data,delimiter=',', fmt='%s')
print('Test accuracy:', score[1])
print('Predictions length', len(predictions))


