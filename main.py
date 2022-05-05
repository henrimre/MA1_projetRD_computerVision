import cv2
import matplotlib.pyplot as plt
import numpy as np

from colorDetection import *

from state import*
#from objectDetection import *

path = r'C:\Users\henri\Documents\HELHa\ProjetRD_image\resistor_1.jpg'
path_background = r'C:\Users\henri\Documents\HELHa\ProjetRD_image\scan_background.jpg'
label = "t bo"
img_res = cv2.imread(path)
# img_res = cv2.GaussianBlur(img_res, (3, 3), cv2.BORDER_DEFAULT)
img_background = cv2.imread(path_background)

"""black = Color("noir", 0, 1, np.array([0, 0, 0]), np.array([0, 0, 0]))
brow = Color("brun", 1, 10, np.array([15, 110, 40]), np.array([50, 40, 173]))
red = Color("rouge", 2, 100, np.array([160, 20, 70]), np.array([190, 255, 255]))
orange = Color("orange", 3, 1e3, np.array([0, 0, 0]), np.array([0, 0, 0]))
yellow = Color("jaune", 4, 10e3, np.array([0, 0, 0]), np.array([0, 0, 0]))
green = Color("vert", 5, 100e3, np.array([0, 0, 0]), np.array([0, 0, 0]))
blue = Color("bleu", 6, 1e6, np.array([110, 50, 50]), np.array([130, 255, 255]))
violet = Color("violet", 7, 10e6, np.array([0, 0, 0]), np.array([0, 0, 0]))
grey = Color("gris", 8, 100e6, np.array([0, 0, 80]), np.array([179, 35, 130]))
white = Color("blanc", 9, 1e9, np.array([0, 0, 0]), np.array([0, 0, 0]))
gold = Color("gold", -1, 0, np.array([15, 108, 44]), np.array([35, 128, 124]))
brown_background = Color("brown_background", -3, 0, np.array([9, 50, 50]), np.array([29, 70, 150]))
blue_background = Color("blue_background", -4, 0, np.array([[87, 99, 75]]), np.array([113, 164, 159]))
green_background = Color("green_background", -5, 0, np.array([50, 46, 38]), np.array([76, 66, 118]))
rose_background = Color("rose background", -6, 0, np.array([125, 26, 65]), np.array([154, 57, 180]))"""


'''def detect_resistor(img, img_background):
    """1.1) Retirer l'arrière plan"""
    img_background_masked = remove_background(img, img_background)
    display_image(label, img_background_masked)

    """ 1.2) Supprimer le reso résiduel restant sur la photo """
    img_background_masked = rose_background.delete_color(img_background_masked,
                                                         img_preprocessing(img_background_masked))
    display_image(label, img_background_masked)

    """ 1.3) Compter le nombre de pixel restant pour déterminer le nombre de résistance """
    # une résistance : 4312, 3703 pixel
    # deux résistances : 7700, 8000
    gray_difference = cv2.cvtColor(img_background_masked, cv2.COLOR_BGR2GRAY)
    mask_ret, mask_thresh = cv2.threshold(gray_difference, 10, 255, cv2.THRESH_BINARY)
    resistor_pixel = int(np.sum(mask_thresh) / 255)
    print("Nombre de pixel restant : " + str(resistor_pixel))

    """ 1.4) Agir en fonction du nombre de pixel restant sur la photo"""
    if resistor_pixel <= 1500:
        print("No resistor")
        return 0, 0
        # mettre à jour le background
    elif 1500 < resistor_pixel <= 4500:
        print("1 résistance")
        # envoyer une information à l'Arduino pour prévenir la détection d'une résistance
        return 1, cv2.bitwise_and(img_res, img_res, mask=mask_thresh)
    else:
        print("2 résistances")
        return 2, 0'''


'''def check_crop_img(img):
    img_hsv = img_preprocessing(img)
    """2.1) Vérification de la présence de résistance avec fond vert ou bleu"""
    if blue_background.detect_number_resistor(img, img_hsv) == 0 and green_background.detect_number_resistor(img,img_hsv) == 0:
        print("Resistance e12 détectée")
        """2.1 Résistanc OK, on peut rogner la photo pour se concentrer sur celle-ci"""
        return brown_background.reshape_resistor(img, img_hsv)
    else:
        print("mauvaise résistance")
        #envoyer code arduino pour ejecter la résistance
        return None'''


state = State.detect
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

    elif state == State.check_crop:
        print("Check and crop state")
        state, img_cropped = check_crop_img(img_masked)

    elif state == State.calculate:
        print("Calculate state")
        display_image(label, img_cropped)
        color_array = locate_color(img_cropped)
        color_array_treated = get_linear_regression(color_array)
        calculate_resistor(color_array_treated)
        boucle = False

    elif state == State.reject:
        print("rejet résistance")
        #envoie d'une valeur de rejet par I2C

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



