"""
This file is used to load and process data for hw1pr2a.py and hw1pr2b.py.
"""
import pandas as pd
import numpy as np


print('==>Loading data...')

# create the headers for data frame since original data dodes not have headers
name_list = ['pix_{}'.format(i + 1) for i in range(784)]
name_list = ['label'] + name_list

df_train = pd.read_csv('http://pjreddie.com/media/files/mnist_train.csv', \
	sep=',', engine='python', names = name_list)

df_test = pd.read_csv('http://pjreddie.com/media/files/mnist_test.csv', \
	sep=',', engine='python', names = name_list)

print('==>Data loaded succesfully.')

# Process training data, so that X (pixels) and y (labels) are seperated
X_train = np.array(df_train[:][[col for col in df_train.columns \
	if col != 'label']]) / 256.

y_train = np.array(df_train[:][['label']])

# Process test data, so that X (pixels) and y (labels) are seperated
X_test = np.array(df_test[:][[col for col in df_test.columns \
	if col != 'label']]) / 256.

y_test = np.array(df_test[:][['label']])
