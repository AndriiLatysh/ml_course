import pandas as pd
import numpy as np
import timeit
import sys
import matplotlib.pyplot as plt
import keras.datasets as keras_datasets
import keras.utils as keras_utils
import keras.models as keras_models
import keras.layers as keras_layers


np.set_printoptions(linewidth=200)

(X_train, y_train), (X_test, y_test) = keras_datasets.mnist.load_data()

# image_index = 3437
# print(y_train[image_index])
# print(X_train[image_index])
# plt.imshow(X_train[image_index], cmap="Greys")
# plt.show()

# sys.exit()

X_train_count = X_train.shape[0]
X_test_count = X_test.shape[0]
X_height = X_train.shape[1]
X_width = X_train.shape[2]

input_shape = (X_height, X_width, 1)

print("Train size: {}".format(X_train_count))
print("Test size: {}".format(X_test_count))

X_train = X_train.reshape(X_train_count, X_height, X_width, 1).astype("float32") / 255
X_test = X_test.reshape(X_test_count, X_height, X_width, 1).astype("float32") / 255

y_train = keras_utils.to_categorical(y_train)
y_test = keras_utils.to_categorical(y_test)

# print(y_train[image_index])
# sys.exit()

CNN_model = keras_models.Sequential()

CNN_model.add(keras_layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
CNN_model.add(keras_layers.MaxPooling2D(pool_size=(2, 2)))

CNN_model.add(keras_layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
CNN_model.add(keras_layers.MaxPooling2D(pool_size=(2, 2)))

CNN_model.add(keras_layers.Flatten())

CNN_model.add(keras_layers.Dense(64, activation="relu"))
CNN_model.add(keras_layers.Dropout(0.2))

CNN_model.add(keras_layers.Dense(10, activation="softmax"))

CNN_model.compile(loss="mean_squared_error",
                  optimizer="adam",
                  metrics=["accuracy"])

print(CNN_model.summary())

stating_time = timeit.default_timer()

CNN_model.fit(x=X_train, y=y_train, epochs=10, batch_size=100, shuffle=True)

ending_time = timeit.default_timer()

print("Training time: {} s".format(round(ending_time-stating_time)))

CNN_model_evaluation = CNN_model.evaluate(x=X_test, y=y_test, verbose=0)

print("Cross-validation accuracy: {}".format(CNN_model_evaluation[1]))

CNN_model.save("models/NMIST_CNN.h5")
