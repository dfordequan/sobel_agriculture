import sys
sys.path.append('/Users/dequanou/Developer/orbital_eye/reduce_false/blog/')


import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from Computer_Vision.Sobel_Edge_Detection.convolution import convolution
from Computer_Vision.Sobel_Edge_Detection.gaussian_smoothing import gaussian_blur

import os


def sobel_edge_detection(image, filter, verbose=False):
    new_image_x = convolution(image, filter, verbose)

    if verbose:
        plt.imshow(new_image_x, cmap='gray')
        plt.title("Horizontal Edge")
        plt.show()

    new_image_y = convolution(image, np.flip(filter.T, axis=0), verbose)

    if verbose:
        plt.imshow(new_image_y, cmap='gray')
        plt.title("Vertical Edge")
        plt.show()

    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

    gradient_magnitude *= 255.0 / gradient_magnitude.max()




    # threshold
    gradient_magnitude[gradient_magnitude < 10] = 0

    # gradient_magnitude[gradient_magnitude > 50] = 0

    # set the non-zero to 255
    gradient_magnitude[gradient_magnitude > 0] = 255


    if verbose:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Gradient Magnitude")
        plt.show()

    return gradient_magnitude


if __name__ == '__main__':
    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True, help="Path to the image")
    # args = vars(ap.parse_args())

    # image = cv2.imread(args["image"])

    folder = '/Users/dequanou/Developer/orbital_eye/reduce_false/test/im1/'  # before image folder
    output_folder = '/Users/dequanou/Developer/orbital_eye/reduce_false/test/im1_sobel/' # before image sobel folder for output
    mask_folder = '/Users/dequanou/Developer/orbital_eye/reduce_false/test/change_mask/' # changes mask folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(folder):
        print(image_name)
        if image_name == '.DS_Store':
            continue
        image = cv2.imread(folder + image_name)
        mask = cv2.imread(mask_folder + image_name)
        # only keep the masked region
        image = cv2.bitwise_and(image, mask)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    

        image = gaussian_blur(image, 9, verbose=False)
        image = sobel_edge_detection(image, filter, verbose=False)
        cv2.imwrite(output_folder + image_name, image)