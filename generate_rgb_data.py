'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import os, cv2
import numpy as np
import math

def read_pixels(folder, verbose = False):
  '''
    Reads 3-D pixel value of the top left corner of each image in folder
    and returns an n x 3 matrix X containing the pixel values
  '''
  n = len(next(os.walk(folder))[2]) # number of files = 83
  X = np.empty([n, 3])
  i = 0

  if verbose:
    fig, ax = plt.subplots()
    h = ax.imshow(np.random.randint(255, size=(28,28,3)).astype('uint8'))

  for filename in os.listdir(folder):
    # read image
    # img = plt.imread(os.path.join(folder,filename), 0)
    img = cv2.imread(os.path.join(folder,filename))
    # convert from BGR (opencv convention) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # store pixel rgb value
    X[i] = img[0,0].astype(np.float64)/255
    i += 1

    '''
    # display
    if verbose:
      h.set_data(img)
      ax.set_title(filename)
      fig.canvas.flush_events()
      plt.show()
    '''

  return X


if __name__ == '__main__':
  folder = 'data/training'
  #get the pixel value of each training data in training sets 'red', 'green', 'blue'
  X1 = read_pixels(folder+'/red', verbose = True) #n = 1352 images
  X2 = read_pixels(folder+'/green')               #n = 1199 images
  X3 = read_pixels(folder+'/blue')                #n = 1143 images
  #calculate number of 1's (reds), 2's (greens), 3's (blues)
  y1, y2, y3 = np.full(X1.shape[0],1), np.full(X2.shape[0], 2), np.full(X3.shape[0],3)
  #concatenate X1,X2,X3 and y1,y2,y3 to form training dataset X and corresponding labels y
  X, y = np.concatenate((X1,X2,X3)), np.concatenate((y1,y2,y3))
  #below is just to see what the output looks like more clearly
  #print('X shape:',X.shape) #(3694,3)
  #print('y shape:',y.shape) #(3694,1)

  #TRAIN PIXEL CLASSIFIER - Gaussian Naive Bayes
  labels = ['red','green','blue']
  num_classes = 3
  num_X_rows = X.shape[0] #3694

  #calculate theta
  theta = np.empty((3, 1))  # kx1, k=#classes
  y_sum = np.empty((3, 1))  # store y_sum
  for k in range(num_classes): #0:2
    k_value = k+1
    for i in range(num_X_rows): #0:3693
      if y[i] == k_value:
        y_sum[k] = y_sum[k] + 1
    theta[k] = y_sum[k]/num_X_rows
  #print('theta:',theta)
  #print('theta shape:',theta.shape) #kx1 = 3x1

  #calculate mu
  mu = np.empty((3, 3))  # kxl, k=#classes, l=#columns in X
  num_X_cols = X.shape[1] #3
  for k in range(num_classes): #0:2
    for l in range(num_X_cols): #0:2
      k_value = k+1
      x_times_y_sum = 0
      for i in range(num_X_rows): #0:3693
        if y[i] == k_value:
          x_times_y_sum = x_times_y_sum + X[i,l]*1
      mu[k,l] = x_times_y_sum/y_sum[k]
  #print('mu:', mu)
  #print('mu shape:', mu.shape) #kxl = 3x3

  #calculate sigma
  sigma = np.empty((3, 3))  # kxl, k=#classes, l=#columns in X
  for k in range(num_classes): #0:2
    for l in range(num_X_cols): #0:2
      k_value = k+1
      x_minus_mu_squared_times_yi = 0
      for i in range(num_X_rows): #0:3693
        if y[i] == k_value:
          x_minus_mu_squared_times_yi = x_minus_mu_squared_times_yi + ((X[i,l]-mu[k,l])**2) * 1
      sigma[k,l] = math.sqrt(x_minus_mu_squared_times_yi/y_sum[k])
  #print('sigma:', sigma)
  #print('sigma shape:', sigma.shape) #kxl = 3x3

  # save theta, mu, and sigma parameters
  np.save('./trained_parameters/theta_rgb.npy', theta)
  np.save('./trained_parameters/mu_rgb.npy', mu)
  np.save('./trained_parameters/sigma_rgb.npy', sigma)



