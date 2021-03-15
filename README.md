# Blue-Recycle-Bin-Detector-using-Naive-Bayes

pixel_classification
1. Run 'test_pixel_classifier.py'
      - performs classification

bin_detection
1. Run 'test_bin_detector.py'
    - performs classification and segmentation
    - calculates bounding boxes
    - uses functions in 'bin_detector.py'


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TA'S PLEASE READ BELOw !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!! NOTE: STUFF BELOW THIS POINT IS DOCUMENTATION IRRELEVENT FOR TAs TO RUN THE CODE !!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


Notes:
- Paths should already be for best model parameters and colorspace.
- Need to uncomment parts for autograder if want results not tailored for autograder.

HOW_TO_RUN_CODE - DATA COLLECTION, PARAMETER TRAINING, TESTING (COMPLETE INSTRUCTIONS)

pixel_classification
  1. Run 'generate_rgb_data.py'
    - calculates and saves trained parameters
  2. Run 'test_pixel_classifier.py'
      - performs classification

bin_detection
  1. Run 'collect_data_roipoly.py'
    - used to hand label training data
    - run 40 times for 8 class hsv classifier using checklist below    
  2. Run 'get_training_parameters.py'
    - calculates and saves trained parameters
  3. Run 'test_bin_detector.py'
    - performs classification and segmentation
    - calculates bounding boxes
    - uses functions in 'bin_detector.py'
Note: - For all the files, be sure to adjust paths and colorspaces when using different trained parameters.
        These are located throughout all the files.
      - Current code runs most optimized parameters and number of classes.


DATA COLLECTION CHECKLIST FOR 8 CLASS HSV CLASSIFIER

Run 1 - Recycling Bin Blue - Loop 1 (Done)
Run 2 - Recycling Bin Blue - Loop 2 (Done)
Run 3 - Recycling Bin Blue - Loop 3 (Done)
Run 4 - Recycling Bin Blue - Loop 4 (Done)
Run 5 - Recycling Bin Blue - Loop 5 (Done)
Run 6 - Sky Blue - Loop 1 (Done)
Run 7 - Sky Blue - Loop 2 (Done)
Run 8 - Sky Blue - Loop 3 (Done)
Run 9 - Sky Blue - Loop 4 (Done)
Run 10 - Sky Blue - Loop 5 (Done)
Run 11 - Brown - Loop 1 (Done)
Run 12 - Brown - Loop 2 (Done)
Run 13 - Brown - Loop 3 (Done)
Run 14 - Brown - Loop 4 (Done)
Run 15 - Brown - Loop 5 (Done)
Run 16 - Light Gray - Loop 1 (Done)
Run 17 - Light Gray - Loop 2 (Done)
Run 18 - Light Gray - Loop 3 (Done)
Run 19 - Light Gray - Loop 4 (Done)
Run 20 - Light Gray - Loop 5 (Done)
Run 21 - Dark Gray - Loop 1 (Done)
Run 22 - Dark Gray - Loop 2 (Done)
Run 23 - Dark Gray - Loop 3 (Done)
Run 24 - Dark Gray - Loop 4 (Done)
Run 25 - Dark Gray - Loop 5 (Done)
Run 26 - Green - Loop 1 (Done)
Run 27 - Green - Loop 2 (Done)
Run 28 - Green - Loop 3 (Done)
Run 29 - Green - Loop 4 (Done)
Run 30 - Green - Loop 5 (Done)
Run 31 - Black - Loop 1 (Done)
Run 32 - Black - Loop 2 (Done)
Run 33 - Black - Loop 3 (Done)
Run 34 - Black - Loop 4 (Done)
Run 35 - Black - Loop 5 (Done)
Run 36 - Tan - Loop 1 (Done)
Run 37 - Tan - Loop 2 (Done)
Run 38 - Tan - Loop 3 (Done)
Run 39 - Tan - Loop 4 (Done)
Run 40 - Tan - Loop 5 (Done)


DESCRIPTION OF TRAINED MODEL PARAMETERS

model parameters - bin_detection
  - RGB Color Space - 3 classes (red, green, blue) corresponding to (1, 2, 3)
    - location: ./pixel_classification/trained_parameters/
    - theta_rgb.npy
    - mu_rgb.npy
    - sigma_rgb.npy

model parameters - pixel_classification  (note: color space doesn't really matter since only pixel location is recorded)
  - RGB Color Space - 2 Classes (recycling bin blue, non recycling bin blue) - strict on only labelling only recycling bin blue color
    - location: ./bin_detection/trained_parameters/rgb_color_space_2_classes
    - theta_bin_rgb.npy
    - mu_bin_rgb.npy
    - sigma_bin_rgb.npy
  - HSV Color Space - 2 Classes (recycling bin blue, non recycling bin blue)
    - location: ./bin_detection/trained_parameters/hsv_color_space_2_classes
    - theta_bin_hsv.npy
    - mu_bin_hsv.npy
    - sigma_bin_hsv.npy
  - LAB Color Space - 2 Classes (recycling bin blue, non recycling bin blue)
    - location: ./bin_detection/trained_parameters/lab_color_space_2_classes
    - theta_bin_lab.npy
    - mu_bin_lab.npy
    - sigma_bin_lab.npy
  - LAB Color Space - 6 Classes (recycling bin blue, sky blue, brown, light gray, dark gray, green)
    - location: ./bin_detection/trained_parameters/lab_color_space_6_classes
    - theta_lab_6_class.npy
    - mu_lab_6_class.npy
    - sigma_lab_6_class.npy
  - LAB Color Space - 7 Classes (recycling bin blue, sky blue, brown, light gray, dark gray, green, black)
    - location: ./bin_detection/trained_parameters/lab_color_space_7_classes
    - theta_lab_7_class.npy
    - mu_lab_7_class.npy
    - sigma_lab_7_class.npy
    - note: accidentally deleted dark gray and recycling bin blue data when training 7 class HSV model
  - HSV Color Space - 6 Classes (recycling bin blue, brown, light gray, dark gray, green, black)
    - location: ./bin_detection/trained_parameters/hsv_color_space_6_classes
    - theta_hsv_6_class_v2.npy
    - mu_hsv_6_class_v2.npy
    - sigma_hsv_6_class_v2.npy
  - HSV Color Space - 8 Classes (recycling bin blue, sky blue, brown, light gray, dark gray, green, black, tan)
    - location: ./bin_detection/trained_parameters/hsv_color_space_8_classes
    - theta_hsv_8_class.npy
    - mu_hsv_8_class.npy
    - sigma_hsv_8_class.npy
    - note: accidentally deleted dark gray and recycling bin blue data when training 8 class HSV model

RESULTS

pixel classification
  - Model: RGB Color Space - 3 Classes (red, green, blue)
    0.927711 = 92.7711% accuracy on test set

bin_detection
  - Model: RGB Color Space - 2 Classes (recycling bin blue, non recycling bin blue)
    3/10 accuracy on test images
  - Model: HSV Color Space - 2 Classes (recycling bin blue, green, brown, red)
    5/10 accuracy on test images
  - Model: LAB Color Space - 2 Classes (recycling bin blue, green, brown, red)
    15/10 accuracy on test images
  - Model: LAB Color Space - 6 Classes (recycling bin blue, sky blue, brown, light gray, dark gray, green)
    9/10 accuracy on test images, 3/10 on autograder
  - Model: LAB Color Space - 7 Classes (recycling bin blue, sky blue, brown, light gray, dark gray, green, black)
    9/10 accuracy on test images, 3.25/10 on autograder
  - Model: HSV Color Space - 6 Classes (recycling bin blue, brown, light gray, dark gray, green, black)
    9/10 accuracy on test images, 3/10 on autograder
  - Model: HSV Color Space - 8 Classes (recycling bin blue, sky blue, brown, light gray, dark gray, green, black, tan)
    10/10 accuracy on test images, 7.25/10 on autograder (BEST MODEL)
    - counting sky blue as same as recycling bin blue for above
      10/10 accuracy on test images, 6.25/10 on autograder

