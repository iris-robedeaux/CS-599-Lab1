""" 
author:-aam35 & ir496
"""
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import utils

#set random seed
tf.random.set_seed(
    599
)

# Define paramaters for the model
learning_rate = 0.001
batch_size = 500
n_epochs = 100
n_train = 60000
n_test = 10000
img_size = 28
num_features = img_size ** 2
n_classes = 10 
img_shape = (img_size, img_size)

# Step 1: Read in data
#Create dataset load function
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

#flatten and normalize the data
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
x_train, x_test = x_train / 255., x_test / 255.

#get the random forest model
#rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
#rf.fit(x_train, y_train)

#get svm model
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_clf.fit(x_train, y_train)

# Predictions
y_pred = svm_clf.predict(x_test)

# Predictions
#y_pred = rf.predict(x_test) # for random forest
y_pred = svm_clf.predict(x_test) #for svm
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")