import os
import csv

samples = []
with open('data_my/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

import sklearn
sklearn.utils.shuffle(samples)
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print("Number of traing samples: ",len(train_samples))
print("Number of validation samples: ",len(validation_samples))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Lambda, Cropping2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import Adam

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # center image
                name = 'data_my/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                flip_center_image = np.fliplr(center_image)
                flip_center_angle = -1.0 * center_angle
                images.append(flip_center_image)
                angles.append(flip_center_angle)
                #left image
                name = 'data_my/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_image = cv2.cvtColor(left_image,cv2.COLOR_BGR2RGB)
                left_angle = float(batch_sample[3]) + 0.2
                images.append(left_image)
                angles.append(left_angle)
                flip_left_image = np.fliplr(left_image)
                flip_left_angle = -1.0 * left_angle
                images.append(flip_left_image)
                angles.append(flip_left_angle)
                #right image
                name = 'data_my/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_image = cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB)
                right_angle = float(batch_sample[3]) - 0.2
                images.append(right_image)
                angles.append(right_angle)
                flip_right_image = np.fliplr(right_image)
                flip_right_angle = -1.0 * right_angle
                images.append(flip_right_image)
                angles.append(flip_right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

batch_size = 32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()

# crop image to only see section with road
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

# centered around zero with small standard deviation
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

#Nvidia model
model.add(Conv2D(24, (5, 5), activation="relu", padding="valid", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="relu", padding="valid", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", padding="valid", strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu", padding="valid", strides=(1, 1)))
model.add(Conv2D(64, (3, 3), activation="relu", padding="valid", strides=(1, 1)))

model.add(Dropout(0.5))
model.add(Flatten())

model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# Fit model
model.fit_generator(train_generator, steps_per_epoch=(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=(len(validation_samples)/batch_size), verbose=1, epochs=1)

# Save model
model.save('model.h5')

print('Done!')