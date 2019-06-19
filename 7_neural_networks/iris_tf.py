# Import the needed libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import urllib.request as request
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Create and train a tensorflow model of a neural network
def create_train_model(hidden_nodes, num_iters):
    # Reset the graph
    tf.reset_default_graph()

    # Placeholders for input and output data
    X = tf.placeholder(shape=(len(X_train), 4), dtype=tf.float64, name="X")
    y = tf.placeholder(shape=(len(y_train), 3), dtype=tf.float64, name="y")

    # Variables for two group of weights between the three layers of the network
    W1 = tf.Variable(np.random.rand(4, hidden_nodes), dtype=tf.float64)
    W2 = tf.Variable(np.random.rand(hidden_nodes, 3), dtype=tf.float64)

    # Create the neural net graph
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Define a loss function
    deltas = tf.square(y_est - y)
    loss = tf.reduce_sum(deltas)

    # Define a train operation to minimize the loss
    optimizer = tf.train.GradientDescentOptimizer(0.005) #0.005
    train = optimizer.minimize(loss)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Go through num_iters iterations
    for i in range(num_iters):
        sess.run(train, feed_dict={X: X_train, y: y_train})
        loss_plot[hidden_nodes].append(sess.run(loss, feed_dict={X: X_train, y: y_train}))
        weights1 = sess.run(W1)
        weights2 = sess.run(W2)

    print("loss (hidden nodes: {0}, iterations: {1}): {2:0.2f}".format(hidden_nodes, num_iters, loss_plot[hidden_nodes][-1]))
    sess.close()
    return weights1, weights2


# # Download dataset
# IRIS_TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
# IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
# 
# names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "species"]
# train = pd.read_csv(IRIS_TRAIN_URL, names=names, skiprows=1)
# test = pd.read_csv(IRIS_TEST_URL, names=names, skiprows=1)
# 
# # Train and test input data
# X_train = train.drop("species", axis=1)
# X_test = test.drop("species", axis=1)
# 
# # Encode target values into binary ("one-hot" style) representation
# y_train = pd.get_dummies(train.species)
# y_test = pd.get_dummies(test.species)

iris_dataset = pd.read_csv("data/iris.csv")

X = np.array(iris_dataset[["sepal-length", "sepal-width", "petal-length", "petal-width"]]).reshape(-1, 4)
y = np.array(iris_dataset["class"]).reshape(-1, 1)

dummy_encoder = preprocessing.OneHotEncoder(sparse=False, categories="auto")
y = dummy_encoder.fit_transform(y)

standard_scaler = preprocessing.StandardScaler()
X = standard_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)


# Run the training for 3 different network architectures: (4-5-3) (4-10-3) (4-20-3)

# Plot the loss function over iterations
num_iters = 5000
num_hidden_nodes = [1, 10, 30] # [5, 10, 20]
loss_plot = {num_hidden_nodes[0]: [], num_hidden_nodes[1]: [], num_hidden_nodes[2]: []}
weights1 = {num_hidden_nodes[0]: None, num_hidden_nodes[1]: None, num_hidden_nodes[2]: None}
weights2 = {num_hidden_nodes[0]: None, num_hidden_nodes[1]: None, num_hidden_nodes[2]: None}

plt.figure(figsize=(12, 8))
for hidden_nodes in num_hidden_nodes:
    weights1[hidden_nodes], weights2[hidden_nodes] = create_train_model(hidden_nodes, num_iters)
    plt.plot(range(num_iters), loss_plot[hidden_nodes], label="nn: 4-{}-3".format(hidden_nodes))

plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(fontsize=12)

# Evaluate models on the test set
X = tf.placeholder(shape=(len(X_test), 4), dtype=tf.float64, name="X")
y = tf.placeholder(shape=(len(y_test), 3), dtype=tf.float64, name="y")

for hidden_nodes in num_hidden_nodes:

    # Forward propagation
    W1 = tf.Variable(weights1[hidden_nodes])
    W2 = tf.Variable(weights2[hidden_nodes])
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Calculate the predicted outputs
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        y_est_np = sess.run(y_est, feed_dict={X: X_test, y: y_test})

    # Calculate the prediction accuracy
    correct = [estimate.argmax(axis=0) == target.argmax(axis=0)
               for estimate, target in zip(y_est_np, y_test)]
    accuracy = 100 * sum(correct) / len(correct)
    print("Network architecture 4-{0}-3, accuracy: {1:0.2f}%".format(hidden_nodes, accuracy))

plt.show()
