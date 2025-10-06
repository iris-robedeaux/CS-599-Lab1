"""
author:-aam35 & ir496
"""
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#set random seed
tf.random.set_seed(
    599
)


# Create data
NUM_EXAMPLES = 500

#define inputs and outputs with some noise 
X = tf.random.normal([NUM_EXAMPLES])  #inputs 
noise = tf.random.normal([NUM_EXAMPLES]) #noise 
#uniform noise!
#noise = tf.random.uniform([NUM_EXAMPLES], minval=-5, maxval=5)
y = X * 3 + 2 + noise  #true output

# Create variables.
W = tf.Variable(np.random.randn(), name = "W")
b = tf.Variable(np.random.randn(), name = "b")
losses = []

train_steps = 2001
learning_rate = 0.001

# Define the linear predictor.
def prediction(x):
  return tf.add(tf.multiply(x, W), b)

# Define loss functions of the form: L(y, y_predicted)
def squared_loss(y, y_predicted):
  return (( y - y_predicted ) **2 )

def l1_loss(y, y_predicted):
  return (abs(y - y_predicted))

def hybrid_loss( y, y_predicted):
  return (squared_loss( y, y_predicted) + l1_loss(y, y_predicted))

for epoch in range(train_steps):
  with tf.GradientTape() as tape:
    y_pred = prediction(X)
    loss = tf.reduce_mean(hybrid_loss(y, y_pred))  
  grads = tape.gradient(loss, [W, b])
  for g, v in zip(grads, [W, b]):
    v.assign_sub(learning_rate * g)

  if len(losses) != 0 and losses[-1] == loss:
    learning_rate = learning_rate / 2
  losses.append(loss)
  if epoch % 100 == 0:
    print(f'MSE for step {epoch}: {loss.numpy():0.3f}')
    #add noise to weights
    newVal = tf.random.normal([], mean=0.0, stddev=1.0)
    #W.assign_add(newVal)
  
plt.plot(X, y, 'bo',label='Generated Data')
plt.plot(X, W.numpy() * X + b.numpy(), 'r', label="Hyrbid Regression")
plt.legend()
plt.show()
print("The final equation is: ", W.numpy(), "x + ", b.numpy())
