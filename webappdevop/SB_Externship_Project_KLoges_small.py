# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 10:42:52 2021

@author: Logu
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen= ImageDataGenerator(rescale=1./255)

x_train=train_datagen.flow_from_directory(r'C:\Users\Logu\Downloads\BreastCancer_Dataset_Small\Train', 
                                          target_size=(128,128), batch_size=32, class_mode="categorical")


x_test=train_datagen.flow_from_directory(r'C:\Users\Logu\Downloads\BreastCancer_Dataset_Small\Test', 
                                          target_size=(128,128), batch_size=32, class_mode="categorical")

print(x_train.class_indices)



"""builing the model"""

model = Sequential()
#add cnn layer
model.add(Convolution2D(32,(5,5),input_shape=(128,128,3),activation="relu"))

#add maxpooling layer
model.add(MaxPooling2D(2,2))

#add flatten layer
model.add(Flatten())

#add hidden layer
model.add(Dense(units=128,activation="relu"))

#add output layer
model.add(Dense(units=2, activation="softmax"))

print(model.summary())

#configure the learning process
model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])


# steps_per_epoch=no.of images in train data/batch_size
#16178/32=506
# validation steps=no.of images in test data/batch _size
#3488/32=109

#fit the model
model.fit(x_train,steps_per_epoch= 506, epochs=30, validation_data=x_test,validation_steps=109 )



model.save("Breast_Cancer_CNN_Model_Small.h5")