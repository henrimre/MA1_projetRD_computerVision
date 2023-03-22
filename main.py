import cv2
import matplotlib.pyplot as plt
import numpy as np

from colorDetection import *

from state import*
#from objectDetection import *

path = r'C:\Users\henri\Documents\HELHa\ProjetRD_image\resistor_1_a.jpg'
path_background = r'C:\Users\henri\Documents\HELHa\ProjetRD_image\scan_background.jpg'
label = "t bo"
img_res = cv2.imread(path)
# img_res = cv2.GaussianBlur(img_res, (3, 3), cv2.BORDER_DEFAULT)
img_background = cv2.imread(path_background)


state = State.vignetting
img_masked = 0
img_cropped = 0
color_array = 0
color_array_treated = 0
boucle = True

while boucle:
    if state == State.init:
        print("Init state")

    elif state == State.detect:
        print("Detect state")
        state, img_masked = detect_resistor(img_res, img_background)
        display_image('img corrected', img_masked)
        state = State.check_crop

    elif state == State.check_crop:
        print("Check and crop state")
        img_masked = apply_correct_vignetting_on_cropped(img_res, img_background)
        state, img_cropped = check_crop_img(img_masked)
        display_image("nul", img_cropped)

    elif state == State.calculate:
        print("Calculate state")
        img_cropped = brown_background.delete_color(img_cropped, img_preprocessing(img_cropped))
        display_image(label, img_cropped)
        color_array = locate_color(img_cropped)
        color_array_treated = get_linear_regression(color_array)
        calculate_resistor(color_array_treated)
        boucle = False

    elif state == State.reject:
        print("rejet résistance")
        boucle = False
        #envoie d'une valeur de rejet par I2C

    elif state == State.vignetting:
        img_corrected = correct_vignetting(cv2.imread(r'C:\Users\henri\Documents\HELHa\ProjetRD_image\photo calibration\res_yellow'))
        cv2.imwrite("all_ressistor_corrected.jpg", img_corrected)

    else:
        print("error no state")



'''
"""1) Détecter la présence d'une résistance"""
nbre_resistor, img_masked = detect_resistor(img_res, img_background)

if nbre_resistor == 1:
    """2) Rogner l'image et déterminer si la résistance reçue est bien une brune"""
    img_cropped = check_crop_img(img_masked)
    if img_cropped is not None:
        """3) Analyser les couleurs sur la résistance pour déterminer calculer la valeur de la résistance"""
        #display_image('image_cropped', img_cropped)
        """black.get_center(img, img_hsv)
        brow.get_center(img, img_hsv)
        red.get_center(img, img_hsv)
        orange.get_center(img, img_hsv)
        yellow.get_center(img, img_hsv)
        green.get_center(img, img_hsv)
        blue.get_center(img, img_hsv)
        violet.get_center(img, img_hsv)
        grey.get_center(img, img_hsv)
        white.get_center(img, img_hsv)"""

        """color_array = np.array([black.get_color_array_format(img, img_hsv),
                                brow.get_color_array_format(img, img_hsv),
                                red.get_color_array_format(img, img_hsv),
                                orange.get_color_array_format(img, img_hsv),
                                yellow.get_color_array_format(img, img_hsv),
                                green.get_color_array_format(img, img_hsv),
                                blue.get_color_array_format(img, img_hsv),
                                violet.get_color_array_format(img, img_hsv),
                                grey.get_color_array_format(img, img_hsv),
                                white.get_color_array_format(img, img_hsv),
                                gold.get_color_array_format(img, img_hsv)])

        print(color_array)

        color_array_treated = get_linear_regression(color_array, True)
        calculate_resistor(color_array_treated)"""

        #envoie de la valeur de la résistance par I2C

'''



<<<<<<< HEAD
print("j'ai fini")



# coucou Henri
=======
>>>>>>> resistorReshape
