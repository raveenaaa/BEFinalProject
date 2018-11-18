#import the necessary package
#tensorflow, keras 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, TensorBoard
import h5py
#set max level of img pixel
size = 512
#Model initialized as Sequential
classifier = Sequential()


classifier.add(Conv2D(32, (3, 3), input_shape = (size, size, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#three layers of hidden 

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(256, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(512, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 4, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/train',
                                                 target_size = (size, size),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/test',
                                            target_size = (size, size),
                                            batch_size = 16,
                                            class_mode = 'categorical')
#construct fit generator
classifier.fit_generator(training_set,
                         steps_per_epoch = 3500,
                         epochs = 2,
                         validation_data = test_set,
                         validation_steps = 1500)
confusion = tf.confusion_matrix(['average', 'best', 'raw', 'worst'], ['average', 'best', 'raw', 'worst'], num_classes=4)
print(confusion)

tensorboard = TensorBoard(log_dir='/output/Graph', histogram_freq=0, write_graph=True, write_images=True)

#save the model
model_json =classifier.to_json()
with open("/output/model.json","w" ) as json_file:
    json_file.write(model_json)
classifier.save_weights("/output/classifier.h5")
print("Training done successfully and model is saved")
classifier.save_weights("/output/classifier.hdf5",overwrite=True)
