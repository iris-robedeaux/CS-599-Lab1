# -*- coding: utf-8 -*-
"""
Author:-aam35
Analyzing Forgetting in neural networks
"""

import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam, RMSprop 

#load the mnist data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#set the random seed
tf.random.set_seed(
    599
)

num_tasks_to_run = 10
num_epochs_per_task = 20

minibatch_size = 32
learning_rate = 0.001

##Permuted MNIST
# Generate the tasks specifications as a list of random permutations of the input pixels.
task_permutation = []
for task in range(num_tasks_to_run):
	task_permutation.append( np.random.permutation(784) )

# Flatten the data first
x_train_flat = x_train.reshape(-1, 784)  # shape: (60000, 784)
x_test_flat  = x_test.reshape(-1, 784)   # shape: (10000, 784)

# Generate the tasks specifications as a list of random permutations of the input pixels.
permuted_datasets = []
for edit in task_permutation:
    # Apply permutation
    x_train_edit = x_train_flat[:, edit]
    x_test_edit  = x_test_flat[:, edit]
    
    # Reshape back to (28,28) for Keras
    x_train_edit = x_train_edit.reshape(-1, 28, 28)
    x_test_edit  = x_test_edit.reshape(-1, 28, 28)
    
    permuted_datasets.append((x_train_edit, y_train, x_test_edit, y_test))

#calculate ACC
def get_acc(taskMatrix):
    numTasks = taskMatrix.shape[0]
    sumCount = 0
    for index in range(taskMatrix.shape[1]):
        sumCount += taskMatrix[numTasks-1][index]  # last row
    return sumCount / numTasks

#calculate bwt
def get_bwt(taskMatrix):
    numTasks = taskMatrix.shape[0]
    sumCount = 0
    for index in range(numTasks-1):
        sumCount += (taskMatrix[numTasks-1][index] - taskMatrix[index][index])
    return sumCount / (numTasks-1)

def l1_l2_loss(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return mae + mse

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='sigmoid'),  
    Dense(128, activation='sigmoid'), 
    Dense(10, activation='softmax'),  
])

#function to make the model
def make_mlp(input_shape=(28, 28), hidden_layers=2, activation='sigmoid', output_units=10, optimizerChoice = "Adam", loss_choice = "NLL", dropout_rate = 0):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))

    # Add hidden layers dynamically
    for i in range(hidden_layers):
        model.add(Dense(256, activation=activation))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Output layer 
    model.add(Dense(output_units, activation='softmax'))

    #choose the optimizer (default, Adam)
    if optimizerChoice == "RMSprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizerChoice == "SGD":
        optimizer = SGD(learning_rate=learning_rate)
    else:
        optimizer = Adam(learning_rate=learning_rate)

    #choose the loss (default, NLL)
    if loss_choice == "L1+L2":
        loss_fn = l1_l2_loss
    elif loss_choice == "L1":
        loss_fn = tf.keras.losses.MeanAbsoluteError()
    elif loss_choice == "L2":
        loss_fn = tf.keras.losses.MeanSquaredError()
    else:
        loss_fn = 'sparse_categorical_crossentropy'
	
    #compile the model
    model.compile(
        optimizer = optimizer,
        loss=loss_fn,
        metrics=['accuracy']
        )
    return model

#train and test the model
taskMatrix = np.zeros((num_tasks_to_run, num_tasks_to_run))

#actually initiate the model:
model = make_mlp( hidden_layers=2, optimizerChoice="Adam", loss_choice= "NLL"   )

#create a loop to go through the permuted data
for i, (x_train, y_train, x_test, y_test) in enumerate(permuted_datasets):

    # Training epochs: 50 for first task, 20 for the rest
    if i == 0:
         epochs = 50
    else:
         epochs = num_epochs_per_task

    #print the current permutation/task
    print(f"\nTraining on Task {i+1}/{num_tasks_to_run} for {epochs} epochs.")

    model.fit(x_train, y_train, epochs=epochs, batch_size=minibatch_size, verbose=0)

    # Evaluate on all tasks seen so far
    for j in range(i + 1):
        x_test_t, y_test_t = permuted_datasets[j][2], permuted_datasets[j][3]
        loss, acc = model.evaluate(x_test_t, y_test_t, verbose=0)
        taskMatrix[i, j] = acc
        print(f"\nTest Accuracy after Task {i+1} on Task {j+1}: {acc:.4f}.\n")

#get performance metrics
finalACC = get_acc(taskMatrix)
finalBWT = get_bwt(taskMatrix)

print(f"The final ACC is {finalACC}, and the final BWT is {finalBWT}.")