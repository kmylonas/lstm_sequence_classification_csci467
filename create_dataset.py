import numpy as np
from sklearn.model_selection import train_test_split
import json
import csv

dataset = []
X = []
Y = []

with open("data.jsonl") as file:
    for line in file:
        tweet = json.loads(line)
        words = tweet["text"].strip().split(' ')
        label = tweet["label"]

        X.append(words)
        Y.append(label)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.33)

if (len(X_train) == len(y_train) and len(X_test) == len(y_test) and len(X_dev) == len(y_dev)):
    print("Splitting was successfull")

print(f'X_train: {len(X_train)}')
print(f'y_train: {len(y_train)}')
print(f'X_dev: {len(X_dev)}')
print(f'y_dev: {len(y_dev)}')
print(f'X_test: {len(X_test)}')
print(f'y_test: {len(y_test)}')


with open("X_train.tsv", "w", newline='') as tsvfile:
    csv_writer = csv.writer(tsvfile, delimiter='\t')
    csv_writer.writerows(X_train)

with open("X_dev.tsv", "w", newline='') as tsvfile:
    csv_writer = csv.writer(tsvfile, delimiter='\t')
    csv_writer.writerows(X_dev)

with open("X_test.tsv", "w", newline='') as tsvfile:
    csv_writer = csv.writer(tsvfile, delimiter='\t')
    csv_writer.writerows(X_test)

np.save("y_train", y_train)
np.save("y_dev", y_dev)
np.save("y_test", y_test)
