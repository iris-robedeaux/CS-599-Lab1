""" 
author:-aam35 & ir496
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
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

# Combine original train and test sets
X_all = np.concatenate([x_train, x_test], axis=0)
Y_all = np.concatenate([y_train, y_test], axis=0)

# Choose proportion for training set (both numerically and by percent)
#train_proportion = n_train / (n_test + n_train)
#train_proportion = 0.6

#x_train, x_test, y_train, y_test = train_test_split(
#    X_all, Y_all, train_size=train_proportion, random_state=42, shuffle=True)

# Step 3: create weights and bias
W = tf.Variable(tf.ones([num_features, n_classes]))
b = tf.Variable(tf.zeros([n_classes]))

#make sure that the variables are floating values
x_train = tf.cast(x_train, tf.float32)
W = tf.cast(W, tf.float32)
b = tf.cast(b, tf.float32)

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
def get_logits(x_vals, W, b):
    return tf.matmul(x_vals, W) + b

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
def get_loss(y_vals, logits):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_vals, logits=logits))

# Step 6: define optimizer
# using Adam Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.optimizers.Adagrad(learning_rate)

# Step 7: calculate accuracy with test set
def get_accuracy(preds, y_test):
    correct_preds = tf.equal(preds, y_test)  # No need to use tf.argmax on y_test
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)).numpy()
    return accuracy

# Create TensorFlow Dataset for training with batching
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=n_train).batch(batch_size)

# Training loop
for epoch in range(n_epochs):
    total_loss = 0
    total_correct = 0
    n_batches = 0
    
    # Training loop for each batch
    for batch_x, batch_y in train_dataset:
        batch_x = tf.cast(batch_x, tf.float32)  # Ensure batch_x is float32
        batch_y = tf.cast(batch_y, tf.int64)  # Ensure batch_y is int64
        with tf.GradientTape() as tape:
            # Flatten the inputs (images)
            batch_x = tf.reshape(batch_x, [-1, num_features])
            
            # Compute the logits and loss
            logits = get_logits(batch_x, W, b)
            loss_value = get_loss(batch_y, logits)
        
        # Compute gradients and apply them
        grads = tape.gradient(loss_value, [W, b])
        optimizer.apply_gradients(zip(grads, [W, b]))
        
        # Accumulate loss
        total_loss += loss_value.numpy()
        
        # Compute accuracy on this batch
        preds = tf.argmax(logits, axis=1)
        total_correct += get_accuracy(preds, batch_y)
        
        n_batches += 1

    # Print epoch results (Average loss & accuracy)
    avg_loss = total_loss / n_batches
    avg_accuracy = total_correct / n_train
    
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy * 100:.2f}%")
	
#Step 9: Get the Final test accuracy
x_test_flat = x_test.reshape([-1, num_features])
x_test_flat = tf.cast(x_test_flat, tf.float32)
logits_test = tf.matmul(x_test_flat, W) + b
preds_test = tf.argmax(logits_test, axis=1)
correct_preds_test = tf.equal(preds_test, y_test)
test_accuracy = tf.reduce_sum(tf.cast(correct_preds_test, tf.float32)) / n_test
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

#Step 10: Helper function to plot images in 3*3 grid
#You can change the function based on your input pipeline

def plot_images(images, y, yhat=None):
    assert len(images) == len(y) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if yhat is None:
            xlabel = "True: {0}".format(y[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(y[i], yhat[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

#Get image from test set 
images = x_test[0:9]

# Get the true classes for those images.
y = y_test[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, y=y)


#Second plot weights 
def plot_weights(w=None):
    # Get the values for the weights from the TensorFlow variable.
    W_values = W.numpy()
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = W_values.min()
    #TO DO## obtains these value from W
    w_max = W_values.max()

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = np.reshape(W_values[:, i], img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

plot_weights(W)