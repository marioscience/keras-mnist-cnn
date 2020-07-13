import math
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from random import seed
from random import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_classes = 10
input_shape = (28, 28, 1)
batch_size = 128
epochs = 50

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("===============================")
print("x_train.shape: ", x_train.shape)
print(x_train.shape[0], " train samples")
print(x_test.shape[0], " test samples")
print("===============================")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# build convolutional neural network
model = keras.Sequential(
    [
        keras.Input(shape=input_shape, name="image_input"),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ]
)

print("===============================")
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

training_metrics = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=1)

print("Test loss: %s\nTest accuracy: %s" % (score[0], score[1]))

seed(1)

plt_figure = plt.figure(2)
visual_samples = 16

for i in range(visual_samples):
    sample = math.floor(random() * len(x_test))
    image = np.array(x_test[sample])
    prediction = model.predict(np.expand_dims(image, axis=0))
    image = image.reshape(28, 28)
    figure = plt_figure.add_subplot(math.sqrt(visual_samples),math.sqrt(visual_samples),i+1)
    figure = plt.imshow(image, cmap="gray")
    print("\t<actual>:\t <predicted>")
    print("\t" + str(np.where(y_test[sample] == 1)[0][0].item()) + ":\t " + str(np.argmax(prediction, axis=1)))

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

learning_curve = plt.figure(1)
print(training_metrics.history.keys())
plt.plot(training_metrics.history["accuracy"])
plt.plot(training_metrics.history["val_accuracy"])
plt.title("Learning/CV Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["Train", "Test"], loc="upper right")

plt.show()

