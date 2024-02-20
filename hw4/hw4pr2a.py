#########################################
#			 Helper Functions	    	#
#########################################

import p2_data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

#########################
#	    Step 1a`		#
#########################

def sigmoid(x):
	"""	Inputs:
			x, a numpy array
		Output: applies the sigmoid / logistic function on each entry of
		the input array and returns the new array.
	"""
	return 1. / (1. + np.exp(-x))


def grad_logreg(X, y, W, reg=0.0):
	"""	Inputs:
			X, the data matrix with dimension m x (n + 1)
			y, the label of the data with dimension m x 1
			W, a weight matrix with bias
			reg, the parameter for regularization
		Output: Calculates and returns the gradient of W for logistic regression.
	"""
	grad = X.T @ (sigmoid(X @ W) - y) + reg * W
	return grad


def NLL(X, y, W, reg=0.0):
	"""Inputs:
			X, the data matrix with dimension m x (n + 1)
			y, the label of the data with dimension m x 1
			W, a weight matrix with bias
			reg, the parameter for regularization
		Output: Calculates and returns the negative log likelihood 
		of the logistic regression with l2 regularization
	"""
	# Number of examples
	m = X.shape[0]
    
    # Compute the hypothesis
	h = sigmoid(X @ W)
    
    # Calculate the negative log likelihood without regularization
	nll = -(y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    
    # Compute the L2 norm of W excluding the bias term if included
	reg_term = (reg / (2 * m)) * np.linalg.norm(W[1:])**2  # Adjust if W includes bias
    
    # Add the regularization term to the NLL
	nll = nll + reg_term
    
    # Since we're working with matrices, sum up to get a scalar
	nll = nll.sum()
	
	return nll


def grad_descent(X, y, reg=0.0, lr=1e-4, eps=1e-6, max_iter=500, print_freq=20):
	"""Inputs:
			X, the data with dimension m x (n + 1)
			y, the label of data with dimension m x 1
			reg, the parameter for regularization
			lr, the learning rate
			eps, the threshold of the norm for the gradients
			max_iter, the maximum number of iterations
			print_freq, the frequency of printing the report
		Output: returns W, the optimal weight by gradient descent,
		        and nll_list, the corresponding learning objectives.
	"""
	# get the shape of the data, and initiate nll list
	m, n = X.shape
	nll_list = []

	# initialize the weight and its gradient
	W = np.zeros((n, 1))
	W_grad = np.ones_like(W)

	print('\n==> Running gradient descent...')

	# Start iteration for gradient descent
	iter_num = 0
	t_start = time.time()

	# Running gradient descent algorithms
	while iter_num < max_iter and np.linalg.norm(W_grad) > eps:
		# Calculate NLL
		nll = NLL(X, y, W, reg = reg)

		if np.isnan(nll):
			break

		nll_list.append(nll)

		# Calculates gradients and updates W
		W_grad = grad_logreg(X, y, W, reg = reg)
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
			t = t_end - t_start))

	return W, nll_list


#########################
#	    Step 1a`		#
#########################

def newton_step(X, y, W, reg=0.0):
	"""	Inputs:
			X, the data matrix with dimension m x (n + 1)
			y, the label of the data with dimension m x 1
			W, a weight matrix with bias
			reg, the parameter for regularization
		Output: Calculates and returns the change of W according
		        to the Newton's method
	"""
	mu = sigmoid(X @ W)

	# Calculates the gradient of log likelihood, grad, wrt W
	g = grad_logreg(X, y, W, reg = reg)

	# Creates a diagonal matrix
	diag = np.diag(np.squeeze(np.asarray(np.multiply(mu, 1. - mu))))

	# Calculates hessian matrix of log regression
	H = X.T @ diag @ X + reg * np.eye(X.shape[1])

	# Solves for d in the equation Hd = -grad
	d = np.linalg.solve(H, g)
	return d


def newton_method(X, y, reg=0.0, eps=1e-6, max_iter=20, print_freq=5):
	"""	Inputs:
			X, the data with dimension m x (n + 1)
			y, the label of data with dimension m x 1
			reg, the parameter for regularization
			eps, the threshold of the norm for the gradients
			max_iter, the maximum number of iterations
			print_freq, the frequency of printing the report
		Output: Returns W, the optimal weight by Newton's Method, and 
			    nll_list, the corresponding learning objectives.
	"""
	# get the shape of the data, and initiate nll list
	m, n = X.shape
	nll_list = []

	# initialize the weight and its gradient
	W = np.zeros((n, 1))
	step = np.ones_like(W)

	print('==> Running Newton\'s method...')

	# Start iteration for gradient descent
	iter_num = 0
	t_start = time.time()

	# Run gradient descent algorithms
	while iter_num < max_iter and np.linalg.norm(step) > eps:

		# Calculates NLL
		nll = NLL(X, y, W, reg = reg)

		if np.isnan(nll):
			break

		nll_list.append(nll)

		# Calculates gradients and updates W
		step = newton_step(X, y, W, reg = reg)
		W -= step

		# Print statements for debugging
		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration {} - negative log likelihood {: 4.4f}'.format(\
					iter_num + 1, nll))

		# Goes to the next iteration
		iter_num += 1

	# benchmark
	t_end = time.time()
	print('-- Time elapsed for running Newton\'s method: {t:2.2f} seconds'.format(\
			t = t_end - t_start))

	return W, nll_list


#########################
#		 Step 3			#
#########################

def predict(X, W):
	""" Inputs:
			W, a weight matrix with bias
			X, the data with dimension m x (n + 1)
		Output: Calculates and returns the predicted label.
	"""
	mu = sigmoid(X @ W)
	return (mu >= 0.5).astype(int)


def get_description(X, y, W):
	"""	Inputs:
			X, the data matrix with dimension m x (n + 1)
			y, the label of the data with dimension m x 1.
			W, the weight matrix with bias and dimension (n + 1) x 1
		Output: Calculates and returns the accuracy, precision,
		        recall and F-1 score of the prediction.

		Description:
			1) We get the predict lables using predict defined above
			2) Note that Accuracy = probability of correct prediction
			3) Precision = probability of true label being 1 given that the
			predicted label is 1
			4) Recall = probablity of predicted label being 1 given that the
			true label is 1
			5) F-1 = 2*p*r / (p + r), where p = precision and r = recall
	"""
	m, n = X.shape
	y_pred = predict(X, W)
	count_a, count_p, count_r = 0, 0, 0
	total_p, total_r = 0, 0

	for index in range(m):
		actual, pred = y.item(index), y_pred.item(index)
		if actual == pred:
			count_a += 1
		if actual == 1:
			total_r += 1
			if pred == 1:
				count_r += 1
		if pred == 1:
			total_p += 1
			if actual == 1:
				count_p += 1

	accuracy = 1. * count_a / m
	precision = 1. * count_p / total_p
	recall = 1. * count_r / total_r
	f1 = 2. * precision * recall / (precision + recall)

	return accuracy, precision, recall, f1


def plot_description(X_train, y_train, X_test, y_test):
	"""	Inputs:
			X_train, the training data with dimension m x (n + 1)
			y_train, the label of training data with dimension m x 1
			X_val, the validation data with dimension m x (n + 1)
			y_val, the label of validation data with dimension m x 1
		Output: Plots the accuracy/precision/recall/F-1 score versus
		        lambda and returns the lambda that maximizes accuracy.
	"""
	# We use 10 different values for lambda, including 0.
	reg_list = [0., 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
	reg_list.sort()
	a_list = []
	p_list = []
	r_list = []
	f1_list = []

	# STRATEGY:
	# 	1) First, generate/create a list of different lambda
	#   2) For each lambda, run gradient descent and obtain the optimal weights
	# 	3) Get accuracy, precision, recall and f1 with the data and W_opt
	#	   obtained, and append those into the corresponding list

	# Run Newton's method or gradient descent
	for index in range(len(reg_list)):
		reg = reg_list[index]
		W_opt, obj = grad_descent(X_train, y_train, reg = reg, \
			lr = 2e-4, print_freq = 100)
		accuracy, precision, recall, f1 = get_description(X_test, y_test, W_opt)
		a_list.append(accuracy)
		p_list.append(precision)
		r_list.append(recall)
		f1_list.append(f1)

	# Generate plots
	# plot accurary versus lambda
	a_vs_lambda_plot, = plt.plot(reg_list, a_list)
	plt.setp(a_vs_lambda_plot, color = 'red')

	# plot precision versus lambda
	p_vs_lambda_plot, = plt.plot(reg_list, p_list)
	plt.setp(p_vs_lambda_plot, color = 'green')

	# plot recall versus lambda
	r_vs_lambda_plot, = plt.plot(reg_list, r_list)
	plt.setp(r_vs_lambda_plot, color = 'blue')

	# plot f1 score versus lambda
	f1_vs_lambda_plot, = plt.plot(reg_list, f1_list)
	plt.setp(f1_vs_lambda_plot, color = 'yellow')

	# Set up the legend, titles, etc. for the plots
	plt.legend((a_vs_lambda_plot, p_vs_lambda_plot, r_vs_lambda_plot, \
		f1_vs_lambda_plot), ('accuracy', 'precision', 'recall', 'F-1'),\
		loc = 'best')
	plt.title('Testing descriptions')
	plt.xlabel('regularization parameter')
	plt.ylabel('Metric')
	plt.savefig('hw4pr2a_description.png', format = 'png')
	plt.close()

	print('==> Plotting completed.')

	# Determines the lambda, reg_opt, that maximizes accuracy
	opt_reg_index = np.argmax(a_list)
	reg_opt = reg_list[opt_reg_index]

	return reg_opt


###########################################
#	    	Main Driver Function       	  #
###########################################

if __name__ == '__main__':

	# =============STEP 0: LOADING DATA=================
	# This data is loaded using the code in p2_data.py.

	# data frame
	df_train = data.df_train
	df_test = data.df_test

	# training data
	X_train = data.X_train
	y_train = data.y_train

	# test data
	X_test = data.X_test
	y_test = data.y_test

	# =============STEP 1: Logistic regression=================
	print('\n\n==> Step 1: Running logistic regression...')

	# Splitting data for logistic regression
	# NOTE: For logistic regression, we only want images with label 0 or 1.
	df_train_logreg = df_train[df_train.label <= 1]
	df_test_logreg = df_test[df_test.label <= 1]

	# Training data for logistic regression
	X_train_logreg = np.array(df_train_logreg[:][[col for \
		col in df_train_logreg.columns if col != 'label']]) / 256.
	y_train_logreg = np.array(df_train_logreg[:][['label']])

	# Testing data for logistic regression
	X_test_logreg = np.array(df_test_logreg[:][[col for \
		col in df_test_logreg.columns if col != 'label']]) / 256.
	y_test_logreg = np.array(df_test_logreg[:][['label']])

	# Stacking a column of 1's to both training and testing data
	X_train_logreg = np.hstack((np.ones_like(y_train_logreg), X_train_logreg))
	X_test_logreg = np.hstack((np.ones_like(y_test_logreg), X_test_logreg))

	# ========STEP 1a: Gradient descent=========
	print('\n==> Step 1a: Running gradient descent...')
	W_gd, nll_list_gd = grad_descent(X_train_logreg, y_train_logreg, reg = 1e-6)

	# ========STEP 1b: Newton's method==========
	print('\n==> Step 1b: Running Newton\'s method...')
	W_newton, nll_list_newton = newton_method(X_train_logreg, y_train_logreg, \
		reg = 1e-6)

	# =============STEP 2: Generate convergence plot=================
	print('\n==> Step 2: Generate Convergence Plot...')
	print('==> Plotting convergence plot...')

	# set up the style for the plot
	plt.style.use('ggplot')

	# plot gradient descent and newton's method convergence plot
	nll_gd_plot, = plt.plot(range(len(nll_list_gd)), nll_list_gd)
	plt.setp(nll_gd_plot, color = 'red')

	nll_newton_plot, = plt.plot(range(len(nll_list_newton)), nll_list_newton)
	plt.setp(nll_newton_plot, color = 'green')

	# add legend, titles, etc. for the plots
	plt.legend((nll_gd_plot, nll_newton_plot), \
		('Gradient descent', 'Newton\'s method'), loc = 'best')
	plt.title('Convergence Plot on Binary MNIST Classification')
	plt.xlabel('Iteration')
	plt.ylabel('NLL')
	plt.savefig('hw4pr2a_convergence.png', format = 'png')
	plt.close()

	print('==> Plotting completed.')

	# =============STEP 3: Generate accuracy/precision plot=================
	print('\nStep 3: ==> Generating plots for accuracy, precision, recall, and F-1 score...')

	# Plot the graph and obtain the optimal regularization parameter
	reg_opt = plot_description(X_train_logreg, y_train_logreg, \
		X_test_logreg, y_test_logreg)

	print('\n==> Optimal regularization parameter is {:4.4f}'.format(reg_opt))
