'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import cv2
from skimage.measure import label, regionprops
import math

class BinDetector():
	def __init__(self):
		'''
			Initilize your recycle-bin detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		self.theta = np.load('./trained_parameters/hsv_color_space_8_classes/theta_hsv_8_class.npy')
		self.mu = np.load('./trained_parameters/hsv_color_space_8_classes/mu_hsv_8_class.npy')
		self.sigma = np.load('./trained_parameters/hsv_color_space_8_classes/sigma_hsv_8_class.npy')

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture,
			call other functions in this class if needed

			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		'''
		# YOUR CODE HERE
		theta = self.theta
		mu = self.mu
		sigma = self.sigma

		#UNCOMMENT BELOW FOR AUTOGRADER SUBMISSION
		# convert from BGR (opencv convention) to LAB
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # CHANGE THIS LINE WHEN CHANGING COLORSPACES - Make sure if this is uncommented, then one in test_bin_detector is commented

		# Replace this with your own approach
		num_classes = 8
		X = img
		num_X_rows = X.shape[0]
		num_X_cols = X.shape[1]
		num_X_rgb_dim = X.shape[2] # 3

		# vectorize input image
		vectorized_X_length = num_X_rows * num_X_cols
		vectorized_X = np.reshape(X, (vectorized_X_length, num_X_rgb_dim))
		# print('vectorized img shape', vectorized_X.shape)
		num_vectorized_X_rows = vectorized_X.shape[0]
		num_vectorized_X_cols = vectorized_X.shape[1]

		# initialize arrays to store predicted class scores and final image mask values
		pred_class_scores = np.zeros((num_vectorized_X_rows, num_classes))  # num_classes = 6
		vectorized_mask_img = np.zeros((num_vectorized_X_rows,1))

		# Find class score for each pixel (do vectorized calculation so code runs fast)
		# for each class
		for k in range(num_classes):  # 0:6
			log_sum = 0
			# for each column of an rgb data in X
			for l in range(num_vectorized_X_cols):  # 0:2
				log_sum = log_sum + math.log((sigma[k,l]) ** 2) + (((vectorized_X[:,l] - mu[k,l]) ** 2) / (sigma[k,l] ** 2))
			pred_class_scores[:,k] = math.log(1 / (theta[k] ** 2)) + log_sum
		vectorized_mask_img = np.argmin(pred_class_scores, axis=1) + 1
		# devectorize mask_img
		mask_img = np.reshape(vectorized_mask_img, (num_X_rows, num_X_cols))
		# When testing new colorspace, uncomment below line to figure out which class to set to 255 and rest to 0 later in this code file
		#print('devectorized mask_img bin pixel value for 1st image', mask_img[192,255]) #print location in recycling bin in 1st validation image #output is 6
		#print('devectorized mask_img bin pixel value for 3rd image', mask_img[124, 231])  # print location in recycling bin in 3rd validation image #output is 6

		return mask_img

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed

			Inputs:
				img - segmented image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2]
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		# YOUR CODE HERE
		num_X_rows = img.shape[0]
		num_X_cols = img.shape[1]
		num_X_pixels = num_X_rows * num_X_cols

		#convert img to numpy array
		image = np.array(img, np.uint8)

		#Set recycling bin blue pixels to 255 and non recycling bin blue pixels to 0
		image[image==1] = 255 #recycling-bin-blue
		image[image==2] = 0
		image[image==3] = 0
		image[image==4] = 0
		image[image==5] = 0
		image[image==6] = 0
		image[image==7] = 0
		image[image==8] = 0
		#print('image:',image)
		#print(img.shape)

		#Erode and Dilate to combine segmented sections that belong together
		kernel3 = np.ones((3,3),np.uint8)
		kernel1 = np.ones((1,1),np.uint8)
		image = cv2.erode(image, kernel3, iterations=1)
		image = cv2.dilate(image, kernel1, iterations=1)

		#COMMENT OUT BELOW WINDOW POPPER BEFORE SUBMISSION TO GRADESCOPE

		#Show segmented image
		cv2.imshow('image', image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		#get contours which contain pixel locations of thresholded out recycling_bin_blue_pixels
		ret, thresh = cv2.threshold(image, 127, 255, 0)
		#Thank you to Python for this line of code below for how to use findContours
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		#print('contours:',contours)
		#print('contours shape:',np.shape(contours)) #1125 for the first image

		#get bounding box estimate
		boxes = []
		for i in range(np.shape(contours)[0]):
			#filter out small bounding boxes
			if (cv2.contourArea(contours[i]) > 0.007 * num_X_pixels):
				x, y, w, h = cv2.boundingRect(contours[i])
				#check if shape is like recycle bin dimensions
				if h < 2.5 * w and h > 1.1 * w:
					boxes.append([x, y, x + w, y + h])

		# COMMENT OUT BELOW WINDOW POPPER BEFORE SUBMISSION TO GRADESCOPE
		print(boxes)

		return boxes


