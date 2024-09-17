import csv
from sklearn.model_selection import train_test_split
import numpy as np

TEST_SIZE = 0.2

with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    labels = []
    for row in reader:
        data.append([float(cell) for cell in row[:4]])
        labels.append(int(row[4]))

x_train, x_test, y_train, y_test = train_test_split(
    np.array(data), np.array(labels), test_size = TEST_SIZE
)

print(x_train)