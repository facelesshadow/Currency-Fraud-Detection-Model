import csv
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
EPOCHS = 10

# model = Perceptron()
# model = svm.SVC()
# model = KNeighborsClassifier(n_neighbors=1)

# Read data in from file
with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    labels = []
    for row in reader:
        data.append([float(cell) for cell in row[:4]])
        labels.append(int(row[4]))

        """data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == "0" else "Counterfeit"
        })"""

# Separate data into training and testing groups
"""holdout = int(0.40 * len(data))
random.shuffle(data)
testing = data[:holdout]
training = data[holdout:]"""



x_train, x_test, y_train, y_test = train_test_split(
    np.array(data), np.array(labels), test_size = TEST_SIZE
)

'''x_train.reshape(
    x_train.shape[0], x_train.shape[1], 1
)
x_test.reshape(
    x_test.shape[0], x_test.shape[1], 1
)'''


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])  

model.compile(
    optimizer="adam",
    loss="binary_crossentropy", 
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=EPOCHS)

model.evaluate(x_test, y_test, verbose=2)