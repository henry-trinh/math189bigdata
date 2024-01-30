import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


###########################################
#	    	Main Driver Function       	  #
###########################################


if __name__ == '__main__':

	# =============part c: Plot data and the optimal linear fit=================

	# load the four data points of X and y
	X = np.array([0, 2, 3, 4])
	y = np.array([1, 3, 6, 8])

	# plot four data points on the plot
	plt.style.use('ggplot')
	plt.plot(X, y, 'ro')

	# used m_opt and b_opt with the solution from part (a) 
	# note that y = mx + b
	m_opt = 62/35
	b_opt = 18/35

	# We then generate 100 points along the line of optimal linear fit.
	#	1) Used np.linspace to get the x-coordinate of 100 points
	#	2) Calculated the y-coordinate of those 100 points with the m_opt and
	#	   b_opt, since y = mx+b.
	#	3) Used a.reshape(-1,1), where a is a np.array, to reshape the array
	#	   to appropriate shape for generating plot

	# Documentation states that stop is exclusive, but the plot appears
	# to be inclusive of the stop value.
	X_space = np.linspace(start=0, stop=4, num=100).reshape(-1, 1)

	line = m_opt * X_space + b_opt
	y_space = line.reshape(-1, 1)

	# plots the optimal learn fit and save it to the current directory
	plt.plot(X_space, y_space)
	plt.title('Optimal Linear Fit')
	plt.savefig('hw1pr2c.png', format='png')
	plt.close()


	# =============part d: Optimal linear fit with random data points=================

	# variables for: mean, std dev, sample size
	mu, sigma, sampleSize = 0, 1, 100

	# Generates noise using np.random.normal
	noise = np.random.normal(mu, sigma, sampleSize).reshape(-1, 1)

	# Generates y-coordinate of the 100 points with noise
	#	1) Uses X_space created in the part (c) above as the x-coordinates
	#	2) In this case, y = mx + b + noise

	y_space_rand =  X_space * m_opt + b_opt + noise

	# Calculates the new parameters for optimal linear fit using the
	# 100 new points generated above
	#	1) Uses np.ones_like to create a column of 1
	#	2) Uses np.hstack to stack column of ones on X_space to create
	#	   X_space_stacked
	#	3) Uses np.linalg.solve to solve W_opt following the normal equation:
	#	   X.T * X * W_opt = X.T * y. Used np.dot for matrix multiplication.

	X_space_stacked = np.hstack((np.ones_like(y_space), X_space))
	W_opt = np.linalg.solve(np.dot(X_space_stacked.T, X_space_stacked),
							np.dot(X_space_stacked.T, y_space_rand))

	# Get the new m, and new b from W_opt obtained above
	b_rand_opt, m_rand_opt = W_opt.item(0), W_opt.item(1)

	# Generates the y-coordinate of 100 points with the new parameters obtained
	#	1) Uses X_space for x-coordinates (same)
	#	2) y = mx + b
	#	3) Ensures appropriate shape using a.reshape(-1,1)

	y_pred_rand = np.array([m_rand_opt * x + b_rand_opt for x in X_space]).reshape(-1, 1)

	# generate plot
	# plot original data points and line
	plt.plot(X, y, 'ro')
	orig_plot, = plt.plot(X_space, y_space, 'r')

	# plot the generated 100 points with white gaussian noise and the new line
	plt.plot(X_space, y_space_rand, 'bo')
	rand_plot, = plt.plot(X_space, y_pred_rand, 'b')

	# set up legend and save the plot to the current folder
	plt.legend((orig_plot, rand_plot), \
		('original fit', 'fit with noise'), loc = 'best')
	plt.savefig('hw1pr2d.png', format='png')
	plt.close()