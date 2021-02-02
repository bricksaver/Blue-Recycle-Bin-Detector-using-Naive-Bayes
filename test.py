import numpy as np

#view bin detection parameters
# 2 class rgb colorspace
theta1 = np.load('./trained_parameters/rgb_color_space_2_classes/theta_bin_rgb.npy')
mu1 = np.load('./trained_parameters/rgb_color_space_2_classes/mu_bin_rgb.npy')
sigma1 = np.load('./trained_parameters/rgb_color_space_2_classes/sigma_bin_rgb.npy')
print('theta1:',theta1)
print('mu1:',mu1)
print('sigma1:',sigma1)
# 2 class hsv colorspace
theta2 = np.load('./trained_parameters/hsv_color_space_2_classes/theta_bin_hsv.npy')
mu2 = np.load('./trained_parameters/hsv_color_space_2_classes/mu_bin_hsv.npy')
sigma2 = np.load('./trained_parameters/hsv_color_space_2_classes/sigma_bin_hsv.npy')
print('theta2:',theta2)
print('mu2:',mu1)
print('sigma2:',sigma2)
# 2 class lab colorspace
theta3 = np.load('./trained_parameters/lab_color_space_2_classes/theta_bin_lab.npy')
mu3 = np.load('./trained_parameters/lab_color_space_2_classes/mu_bin_lab.npy')
sigma3 = np.load('./trained_parameters/lab_color_space_2_classes/sigma_bin_lab.npy')
print('theta3:',theta3)
print('mu3:',mu3)
print('sigma3:',sigma3)
# 6 class lab colorspace
theta4 = np.load('./trained_parameters/lab_color_space_6_classes/theta_lab_6_class.npy')
mu4 = np.load('./trained_parameters/lab_color_space_6_classes/mu_lab_6_class.npy')
sigma4 = np.load('./trained_parameters/lab_color_space_6_classes/sigma_lab_6_class.npy')
print('theta4:',theta4)
print('mu4:',mu4)
print('sigma4:',sigma4)
#7 class lab colorspace
theta5 = np.load('./trained_parameters/lab_color_space_7_classes/theta_lab_7_class.npy')
mu5 = np.load('./trained_parameters/lab_color_space_7_classes/mu_lab_7_class.npy')
sigma5 = np.load('./trained_parameters/lab_color_space_7_classes/sigma_lab_7_class.npy')
print('theta5:',theta5)
print('mu5:',mu5)
print('sigma5:',sigma5)
#8 class hsv colorspace
theta6 = np.load('./trained_parameters/hsv_color_space_8_classes/theta_hsv_8_class.npy')
mu6 = np.load('./trained_parameters/hsv_color_space_8_classes/mu_hsv_8_class.npy')
sigma6 = np.load('./trained_parameters/hsv_color_space_8_classes/sigma_hsv_8_class.npy')
print('theta6:',theta6)
print('mu6:',mu6)
print('sigma6:',sigma6)
#8 class hsv colorspace - sky blue same class as recycling bin blue
#Same parameters as '8 class hsv colorspace'

