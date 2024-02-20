#########################################
#			 Helper Functions	    	#
#########################################

import p2_data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import time


def NLL(X, y, W, reg=0.0):
	"""	Inputs: 
			X, the data matrix with dimension m x (n + 1)
			y, the label of the data with dimension m x 1
			W, a weight matrix
			reg, the parameter for regularization
		Output: Calculates and returns negative log likelihood for
		softmax regression.

		Strategy:
			1) We recall the negative log likelihood function for softmax
			regression and
			2) Use a.sum() to find the summation of all entries in a numpy
			array.
			3) When performing operations vertically across rows, we use axis=0.
			When perform operations horizontally across columns, we use
			axis=1.
			4) Use np.exp and np.log to calculate the exp and log of
			each entry of the input array
	"""
	mu = X @ W # m x k
	exp_mu = np.exp(mu)
	prob = exp_mu / exp_mu.sum(axis=1).reshape(-1, 1)
	groundTruth = y * np.log(prob)
	NLL = -groundTruth.sum(axis=1).sum() + reg * np.diag(W.T @ W).sum()
	return NLL


def grad_softmax(X, y, W, reg=0.0):
	"""	Inputs:
			X, the data matrix with dimension m x (n + 1)
			y, the label of the data with dimension m x 1
			W, a weight matrix
			reg, the parameter for regularization
		Output: Returns the gradient of W for softmax regression.

		Strategy:
			1) Recall the log likelihood function for softmax regression and
			   get the gradient with respect to the weight matrix, W
			2) Apply the regularization
	"""
	mu = X @ W
	exp_mu = np.exp(mu)
	prob = exp_mu / exp_mu.sum(axis=1).reshape(-1, 1)
	grad = X.T @ (prob - y) + reg * W
	return grad


def predict(X, W):
	"""	Inputs:
			X, the data matrix with dimension m x (n + 1)
			W, a weight matrix
		Output: Returns the predicted labels y_pred with
		dimension m x 1

		Strategy:
			1) Obtain the probablity matrix according to the softmax equation
			2) Use np.argmax to get the predicted label for each image
	"""
	mu = X @ W
	exp_mu = np.exp(mu)
	prob = exp_mu / exp_mu.sum(axis=1).reshape(-1, 1)
	y_pred = np.argmax(prob, axis=1).reshape(-1, 1)
	return y_pred


def get_accuracy(y_pred, y):
	"""	Inputs:
			y_pred, the predicted label of data with dimension m x 1
			y, the true label of data with dimension m x 1
		Output: Returns the accuracy of the prediction
	"""
	diff = (y_pred == y).astype(int)
	accu = 1. * diff.sum() / len(y)
	return accu


def grad_descent(X, y, reg=0.0, lr=1e-5, eps=1e-6, max_iter=500, print_freq=20):
	"""	Inputs:
			X, the data with dimension m x (n + 1)
			y, the label of data with dimension m x 1
			reg, the parameter for regularization
			lr, the learning rate
			eps, the threshold of the norm for the gradients
			max_iter, the maximum number of iterations
			print_freq, the frequency of printing the report
		Output: Returns W, the optimal weight by gradient descent,
		and nll_list, the corresponding learning objectives.
	"""
	# Gets the shape of the data, and initialize nll_list
	m, n = X.shape
	k = y.shape[1]
	nll_list = []

	# Initializes the weight and its gradient
	W = np.zeros((n, k))
	W_grad = np.ones((n, k))

	print('\n==> Running gradient descent...')

	# Starts iteration for gradient descent
	iter_num = 0
	t_start = time.time()

	# Running gradient descent algorithms (updates W; calculates LO)
	while iter_num < max_iter and np.linalg.norm(W_grad) > eps:
		# calculate NLL
		nll = NLL(X, y, W, reg=reg)
		if np.isnan(nll):
			break
		nll_list.append(nll)

		# calculate gradients and update W
		W_grad = grad_softmax(X, y, W, reg=reg)
		W -= lr * W_grad

		# Print statements for debugging
		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration {} - negative log likelihood {: 4.4f}'.format(\
					iter_num + 1, nll))

		# Goes to the next iteration
		iter_num += 1

	# benchmark
	t_end = time.time()
	print('-- Time elapsed for running gradient descent: {t:2.2f} seconds'.format(\
			t=t_end - t_start))

	return W, nll_list


def accuracy_vs_lambda(X_train, y_train_OH, X_test, y_test, lambda_list):
	"""	Inputs:
			X_train, the training data with dimension m x (n + 1)
			y_train_OH, the label of training data with dimension m x 1
			X_test, the validation data with dimension m x (n + 1)
			y_test, the label of validation data with dimension m x 1
			lambda_list, a list of different regularization paramters that
						we want to test
		Output: Generates a plot of accuracy of prediction vs lambda and
		returns the regularization parameter that maximizes the accuracy,
		reg_opt.

		STRATEGY: Generates the list of accuracies following the steps below:
			1) Run gradient descent with each parameter to obtain the optimal
			weight
			2) Predicted the label using the weights
			3) Use get_accuracy function provided to calculate the accuracy
	"""
	# initialize the list of accuracy
	accu_list = []

	# Determines corresponding accuracy values for each parameter
	for reg in lambda_list:
		W, nll_list = grad_descent(X_train, y_train_OH, reg=reg, lr=2e-5, \
		print_freq=50)
		y_pred = predict(X_test, W)
		accuracy = get_accuracy(y_pred, y_test)
		accu_list.append(accuracy)

		print('-- Accuracy is {:2.4f} for lambda = {:2.2f}'.format(accuracy, reg))

	# Plot accuracy vs lambda
	print('==> Printing accuracy vs lambda...')
	plt.style.use('ggplot')
	plt.plot(lambda_list, accu_list)
	plt.title('Accuracy versus Lambda in Softmax Regression')
	plt.xlabel('Lambda')
	plt.ylabel('Accuracy')
	plt.savefig('hw4pr2b_lva.png', format = 'png')
	plt.close()
	print('==> Plotting completed.')

	# Find the optimal lambda that maximizes the accuracy
	opt_lambda_index = np.argmax(accu_list)
	reg_opt = lambda_list[opt_lambda_index]
	return reg_opt


###########################################
#	    	Main Driver Function       	  #
###########################################

if __name__ == '__main__':

	# =============STEP 0: LOADING DATA=================
	# Loads code from p2_data.py #

	df_train = data.df_train
	df_test = data.df_test

	X_train = data.X_train
	y_train = data.y_train
	X_test = data.X_test
	y_test = data.y_test

	# stacking an array of ones
	X_train = np.hstack((np.ones_like(y_train), X_train))
	X_test = np.hstack((np.ones_like(y_test), X_test))

	# one hot encoder
	enc = OneHotEncoder()
	y_train_OH = enc.fit_transform(y_train.copy()).astype(int).toarray()
	y_test_OH = enc.fit_transform(y_test.copy()).astype(int).toarray()


	# =============STEP 1: Accuracy versus lambda=================
	print('\n\n==> Step 1: Finding optimal regularization parameter...')

	lambda_list = [0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
	reg_opt = accuracy_vs_lambda(X_train, y_train_OH, X_test, y_test, lambda_list)

	print('\n-- Optimal regularization parameter is {:2.2f}'.format(reg_opt))


	# =============STEP 2: Convergence plot=================
	# run gradient descent to get the nll_list
	W_gd, nll_list_gd = grad_descent(X_train, y_train_OH, reg=reg_opt,\
								    max_iter=1500, lr=2e-5, print_freq=100)

	print('\n==> Step 2: Plotting convergence plot...')

	# set up style for the plot
	plt.style.use('ggplot')

	# generate the convergence plot
	nll_gd_plot, = plt.plot(range(len(nll_list_gd)), nll_list_gd)
	plt.setp(nll_gd_plot, color = 'red')

	# add legend, title, etc and save the figure
	plt.title('Convergence Plot on Softmax Regression with $\lambda = {:2.2f}$'.format(reg_opt))
	plt.xlabel('Iteration')
	plt.ylabel('NLL')
	plt.savefig('hw4pr2b_convergence.png', format = 'png')
	plt.close()

	print('==> Plotting completed.')
