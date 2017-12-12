from csv import reader
import numpy as np
from random import seed
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score

## Pre-processing functions
# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup


## main routine to test bagging on the sonar dataset
seed(1)
# load and prepare data
filename = 'sonar.all-data.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5
max_depth = 6
sample_size = 0.50

import numpy as np
x = np.zeros((len(dataset),len(dataset[0])-1))
y = np.zeros(len(dataset))
for i in range(0, len(dataset)):
    x[i,:] = dataset[i][0:len(dataset[0])-1]
    y[i] = dataset[i][len(dataset[0])-1]


for n_trees in [1, 5, 10, 50]:
    clf = BaggingClassifier(RandomForestClassifier(), n_estimators = n_trees, max_samples=sample_size)
    scores = cross_val_score(clf, x, y, cv=n_folds, scoring='accuracy')
    print('Trees: %d' % n_trees)
    print('Scores: %s' % scores)
    print('Mean Accuracy: {:3f}%'.format(np.mean(scores)*100))