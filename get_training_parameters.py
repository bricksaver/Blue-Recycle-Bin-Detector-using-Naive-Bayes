'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import math

if __name__ == '__main__':

    #import training data obtained from collect_data_roipoly.py
    # load recycling bin blue data
    recycling_bin_blue_pixels_set1 = np.load('./data/training/hsv_color_space_8_classes/recycling_bin_blue_pixels_hsv_set1.npy')
    recycling_bin_blue_pixels_set2 = np.load('./data/training/hsv_color_space_8_classes/recycling_bin_blue_pixels_hsv_set2.npy')
    recycling_bin_blue_pixels_set3 = np.load('./data/training/hsv_color_space_8_classes/recycling_bin_blue_pixels_hsv_set3.npy')
    # load sky blue data
    sky_blue_pixels_set1 = np.load('./data/training/hsv_color_space_8_classes/sky_blue_pixels_hsv_set1.npy')
    sky_blue_pixels_set2 = np.load('./data/training/hsv_color_space_8_classes/sky_blue_pixels_hsv_set2.npy')
    sky_blue_pixels_set3 = np.load('./data/training/hsv_color_space_8_classes/sky_blue_pixels_hsv_set3.npy')
    # load brown data
    brown_pixels_set1 = np.load('./data/training/hsv_color_space_8_classes/brown_pixels_hsv_set1.npy')
    brown_pixels_set2 = np.load('./data/training/hsv_color_space_8_classes/brown_pixels_hsv_set2.npy')
    brown_pixels_set3 = np.load('./data/training/hsv_color_space_8_classes/brown_pixels_hsv_set3.npy')
    # load light gray data
    light_gray_pixels_set1 = np.load('./data/training/hsv_color_space_8_classes/light_gray_pixels_hsv_set1.npy')
    light_gray_pixels_set2 = np.load('./data/training/hsv_color_space_8_classes/light_gray_pixels_hsv_set2.npy')
    light_gray_pixels_set3 = np.load('./data/training/hsv_color_space_8_classes/light_gray_pixels_hsv_set3.npy')
    # load dark gray data
    dark_gray_pixels_set1 = np.load('./data/training/hsv_color_space_8_classes/dark_gray_pixels_hsv_set1.npy')
    dark_gray_pixels_set2 = np.load('./data/training/hsv_color_space_8_classes/dark_gray_pixels_hsv_set2.npy')
    dark_gray_pixels_set3 = np.load('./data/training/hsv_color_space_8_classes/dark_gray_pixels_hsv_set3.npy')
    # load green data
    green_pixels_set1 = np.load('./data/training/hsv_color_space_8_classes/green_pixels_hsv_set1.npy')
    green_pixels_set2 = np.load('./data/training/hsv_color_space_8_classes/green_pixels_hsv_set2.npy')
    green_pixels_set3 = np.load('./data/training/hsv_color_space_8_classes/green_pixels_hsv_set3.npy')
    # load black data
    black_pixels_set1 = np.load('./data/training/hsv_color_space_8_classes/black_pixels_hsv_set1.npy')
    black_pixels_set2 = np.load('./data/training/hsv_color_space_8_classes/black_pixels_hsv_set2.npy')
    black_pixels_set3 = np.load('./data/training/hsv_color_space_8_classes/black_pixels_hsv_set3.npy')
    # load tan data
    tan_pixels_set1 = np.load('./data/training/hsv_color_space_8_classes/tan_pixels_hsv_set1.npy')
    tan_pixels_set2 = np.load('./data/training/hsv_color_space_8_classes/tan_pixels_hsv_set2.npy')
    tan_pixels_set3 = np.load('./data/training/hsv_color_space_8_classes/tan_pixels_hsv_set3.npy')

    #combine the 7 above training sets into two (kx3) numpy arrays by class
    recycling_bin_blue_pixels = np.vstack((recycling_bin_blue_pixels_set1, recycling_bin_blue_pixels_set2, recycling_bin_blue_pixels_set3))
    sky_blue_pixels = np.vstack((sky_blue_pixels_set1, sky_blue_pixels_set2, sky_blue_pixels_set3))
    brown_pixels = np.vstack((brown_pixels_set1, brown_pixels_set2, brown_pixels_set3))
    light_gray_pixels = np.vstack((light_gray_pixels_set1, light_gray_pixels_set2, light_gray_pixels_set3))
    dark_gray_pixels = np.vstack((dark_gray_pixels_set1, dark_gray_pixels_set2, dark_gray_pixels_set3))
    green_pixels = np.vstack((green_pixels_set1, green_pixels_set2, green_pixels_set3))
    black_pixels = np.vstack((black_pixels_set1, black_pixels_set2, black_pixels_set3))
    tan_pixels = np.vstack((tan_pixels_set1, tan_pixels_set2, tan_pixels_set3))

    #for easier notation for training data
    X1 = recycling_bin_blue_pixels
    X2 = sky_blue_pixels
    X3 = brown_pixels
    X4 = light_gray_pixels
    X5 = dark_gray_pixels
    X6 = green_pixels
    X7 = black_pixels
    X8 = tan_pixels
    # y: 1 = recycling bin blue, 2 = sky blue, 3 = brown, 4 = light gray, 5 = dark gray, 6 = green, 7 = black, 8 = tan
    y1, y2, y3, y4, y5, y6, y7, y8 = np.full(X1.shape[0],1), np.full(X2.shape[0],2), np.full(X3.shape[0],3), np.full(X4.shape[0],4), np.full(X5.shape[0],5), np.full(X6.shape[0],6), np.full(X7.shape[0],7), np.full(X8.shape[0],8)
    #concatenate X's and y's to form training dataset X and corresponding labels y
    X, y = np.concatenate((X1,X2,X3,X4,X5,X6,X7,X8)), np.concatenate((y1,y2,y3,y4,y5,y6,y7,y8))
    #print(X.shape) #(x,3)
    #print(y.shape) #(x,1)

    #TRAIN PIXEL CLASSIFIER - Gaussian Naive Bayes
    labels = ['recycling_bin_blue','sky_blue','brown','light_gray','dark_gray','green','black','tan']
    num_classes = 8
    num_X_rows = X.shape[0] #16517669
    print('num_X_rows:', num_X_rows)

    #calculate theta
    theta = np.zeros((num_classes, 1)) #kx1, k=#classes
    y_sum = np.zeros((num_classes, 1)) #store y_sum
    for k in range(num_classes): #0:6
        k_value = k+1
        for i in range(num_X_rows): #0:16517669
            if y[i] == k_value:
                y_sum[k] = y_sum[k] + 1
        print('y_sum[k]',y_sum[k])
        theta[k] = y_sum[k]/num_X_rows
    print('theta:',theta)
    print('theta shape:',theta.shape) #kx1 = 7x1

    #calculate mu
    num_X_cols = X.shape[1]  # 3
    mu = np.zeros((num_classes, num_X_cols)) # kxl, k=#classes, l=#columns in X
    for k in range(num_classes): #0:6
        for l in range(num_X_cols): #0:2
            k_value = k+1
            x_times_y_sum = 0
            for i in range(num_X_rows): #0:16517669
                if y[i] == k_value:
                    x_times_y_sum = x_times_y_sum + X[i,l]*1
            mu[k,l] = x_times_y_sum/y_sum[k]
    print('mu:', mu)
    print('mu shape:', mu.shape) #kxl = 7x3

    #calculate sigma
    sigma = np.zeros((num_classes,num_X_cols)) #kxl, k=#classes, l=#columns in X
    for k in range(num_classes): #0:6
        for l in range(num_X_cols): #0:2
            k_value = k+1
            x_minus_mu_squared_times_yi = 0
            for i in range(num_X_rows): #0:16517669
                if y[i] == k_value:
                    x_minus_mu_squared_times_yi = x_minus_mu_squared_times_yi + ((X[i,l]-mu[k,l])**2) * 1
            sigma[k,l] = math.sqrt(x_minus_mu_squared_times_yi/y_sum[k])
    print('sigma:', sigma)
    print('sigma shape:', sigma.shape) #kxl = 7x3

    # save theata, mu, and sigma parameters
    np.save('./trained_parameters/hsv_color_space_8_classes/theta_hsv_8_class.npy', theta)
    np.save('./trained_parameters/hsv_color_space_8_classes/mu_hsv_8_class.npy', mu)
    np.save('./trained_parameters/hsv_color_space_8_classes/sigma_hsv_8_class.npy', sigma)







