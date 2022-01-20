#%%
import tensorflow as tf
from sklearn.metrics import f1_score 
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from keras.models import Sequential,model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
import tensorflow.keras.backend as K
from keras.utils import np_utils

#%%
train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)
path = 'data2/data2'
train_generator = train_datagen.flow_from_directory(
        path+'/training_data',  
        target_size=(32,32),  
        batch_size=1,
        class_mode='sparse',
        color_mode="grayscale")

validation_generator = train_datagen.flow_from_directory(
        path+'/testing_Data',  
        target_size=(32,32),  
        class_mode='sparse',
        color_mode="grayscale")

model = Sequential()
model.add(Conv2D(16, (22,22), input_shape=(32, 32, 1), activation='relu', padding='same'))
model.add(Conv2D(32, (16,16), input_shape=(32, 32, 1), activation='relu', padding='same'))
model.add(Conv2D(64, (8,8), input_shape=(32, 32, 1), activation='relu', padding='same'))
model.add(Conv2D(128, (4,4), input_shape=(32, 32, 1), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(4,4)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dense(36, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001))
model.fit(
      train_generator, 
      epochs = 25)

model.save("try1.h5")

