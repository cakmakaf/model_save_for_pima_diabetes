#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 20:17:59 2021

@author: ahmetcakmak
"""

# Beep Learning MLP model for Pima Indians Dataset 
# Serialize to JSON and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import pandas as pd
import os
# Fix random seed for reproducibility
numpy.random.seed(10)

# Load PIMA Indians diabetes dataset
dataset = pd.read_csv("diabetes.csv")
# dataset.head()

# Split into the data as input (X) and output (Y) variables
X = dataset.iloc[:, 0:8]
Y = dataset.iloc[:, 8]

# Create the MLP model
model = Sequential()
model.add(Dense(256, input_dim=8, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=500, batch_size=64, verbose=0)

# Evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
# Serialize the weights to HDF5
model.save_weights("model.h5")
print("Saved the model to disk")


# Load JSON and create the model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load the weights into the new model
loaded_model.load_weights("model.h5")
print("Loaded the model from disk")

# Evaluate loaded model on the test data
loaded_model.compile(loss='binary_crossentropy',
                     optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
