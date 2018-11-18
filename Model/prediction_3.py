import numpy as np
import os
from keras.preprocessing import image
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout

size = 512

##################### Adding the layers of the Model###################################
classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (size, size, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(256, (3, 3), activation = 'relu'))
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
####################################################################################################


fname = "I:\\Project\\Model\\3 layers\\classifier.hdf5"
classifier.load_weights(fname)  #Load the weights

#################### Prediction ##################################
for img in os.listdir('I:\\Project\\try\\'):
    test_image = image.load_img('I:\\Project\\try\\'+img, target_size = (512, 512))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    if result[0][0] == 1:
        prediction = 'Average orange'
    elif result[0][1] == 1:
        prediction = 'Best orange'
    elif result[0][2] == 1:
        prediction = 'Raw Orange'
    else:
        prediction = 'Worst'    
    
    print(img)
    print(prediction)
    print(result)
    print("\n")
