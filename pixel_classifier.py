'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
import math

class PixelClassifier():
  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    self.theta = np.load('./trained_parameters/theta_rgb.npy')
    self.mu = np.load('./trained_parameters/mu_rgb.npy')
    self.sigma = np.load('./trained_parameters/sigma_rgb.npy')

  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue

	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''

    # YOUR CODE HERE
    theta = self.theta
    mu = self.mu
    sigma = self.sigma

    # Just a random classifier for now
    # Replace this with your own approach
    num_classes = 3
    num_X_rows = X.shape[0]
    num_X_cols = X.shape[1]
    y = np.empty((num_X_rows, 1)) # length = #rows in X
    pred_class_scores = np.empty((num_X_rows,num_classes)) #num_classes = 3
    #for each row
    for k in range(num_classes):  #0:2
      log_sum = 0
      #for each column of an rgb data in X
      for l in range(num_X_cols):  #0:2
        log_sum =  log_sum + math.log((sigma[k,l])**2) + ( ((X[:,l] - mu[k,l])**2) / (sigma[k,l]**2) )
      pred_class_scores[:,k] = math.log(1/(theta[k]**2)) + log_sum
    y = np.argmin(pred_class_scores, axis=1) + 1 #'+1' is to account for '-1' and make actual class value
    #print('y:', y)
    #print('y shape:', y.shape)  # length = #rows in X

    return y

