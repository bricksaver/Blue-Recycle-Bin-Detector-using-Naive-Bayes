'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


from __future__ import division

from generate_rgb_data import read_pixels
from pixel_classifier import PixelClassifier

if __name__ == '__main__':
  # test the classifier
  
  folder = 'data/validation/blue'
  X = read_pixels(folder)

  myPixelClassifier = PixelClassifier()
  y = myPixelClassifier.classify(X)
  #print('X size:', X.shape) #(83,3)
  #print('y size:',y.shape)  #(83,1)

  #'y==2' represents number of pixels classified as blue
  print('Precision: %f' % (sum(y==3)/y.shape[0])) #0.927711
  #print(y)

  
