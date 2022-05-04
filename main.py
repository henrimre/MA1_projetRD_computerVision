import cv2
import matplotlib.pyplot as plt
import numpy as np

from colorDetection import *
from objectDetection import *

# img = substract_background(path, path_background)

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
gold = Color("gold", -1, 0, np.array([15, 108, 44]), np.array([35, 128, 124]))"""
brown_background = Color("brown_background", -3, 0, np.array([9, 50, 50]), np.array([29, 70, 150]))
blue_background = Color("blue_background", -4, 0, np.array([[87, 99, 75]]), np.array([113, 164, 159]))
green_background = Color("green_background", -5, 0, np.array([50, 46, 38]), np.array([76, 66, 118]))
rose_background = Color("rose background", -6, 0, np.array([125, 26, 65]), np.array([154, 57, 180]))


def detect_resistor(img, img_background):
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
        return 2, 0


def check_crop_img(img):
    img_hsv = img_preprocessing(img)
    """2.1) Vérification de la présence de résistance avec fond vert ou bleu"""
    if blue_background.detect_number_resistor(img, img_hsv) == 0 and green_background.detect_number_resistor(img,img_hsv) == 0:
        print("Resistance e12 détectée")
        """2.1 Résistanc OK, on peut rogner la photo pour se concentrer sur celle-ci"""
        return brown_background.reshape_resistor(img, img_hsv)
    else:
        print("mauvaise résistance")
        #envoyer code arduino pour ejecter la résistance
        return None





"""1) Détecter la présence d'une résistance"""
nbre_resistor, img_masked = detect_resistor(img_res, img_background)

if nbre_resistor == 1:
    """2) Rogner l'image et déterminer si la résistance reçue est bien une brune"""
    img_cropped = check_crop_img(img_masked)
    if img_cropped is not None:
        """3) Analyser les couleurs sur la résistance pour déterminer calculer la valeur de la résistance"""
        #display_image('image_cropped', img_cropped)

"""gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
plt.show()
blur = cv2.GaussianBlur(gray, (5,5),0)
plt.imshow(blur)
plt.show()
canny = cv2.Canny(blur, 80, 110)
plt.imshow(canny)
plt.show()
sum_pixel_edge = int(np.sum(canny)/255)
print("sum pixel edge :" + str(sum_pixel_edge))
print("e 12 brown")
test_color_resistor = brown_background.detect_number_resistor(img, img_hsv, 2)
print("e 12 blue")
test_color_blue_resistor = blue_background.detect_number_resistor(img, img_hsv, 2)

test_color_green_resistor = green_background.detect_number_resistor(img, img_hsv, 2)

if 10 < sum_pixel_edge < 1800 and test_color_blue_resistor == 0 and test_color_green_resistor == 0 and test_color_resistor == 1:
    if test_color_resistor == 1:
        print("1 resistor e12")
    else:
        print("1 resistor no e12")
else:
    print("ko resistor")"""

# 1 résistance = 1331
# 2 résistance = 2093
# np.set_printoptions(threshold=np.inf)
# print(canny)
"""
plt.imshow(canny, cmap='gray')
plt.show()
dilated = cv2.dilate(canny,(1,1), iterations=0)
(cnt, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
plt.imshow(rgb)
plt.show()
print("resistor in the image :",len(cnt))"""

"""red.get_masked_image(img, img_hsv, True)
blue.get_masked_image(img, img_hsv, True)
grey.get_masked_image(img, img_hsv, True)
gold.get_masked_image(img, img_hsv, True)
gold.get_center(img, img_hsv, 2)"""

"""a = black.get_center(img, img_hsv)
b = brow.get_center(img, img_hsv)
red.get_center(img, img_hsv, 2)
orange.get_center(img, img_hsv)
yellow.get_center(img, img_hsv)
green.get_center(img, img_hsv)
blue.get_center(img, img_hsv, 2)
violet.get_center(img, img_hsv)
grey.get_center(img, img_hsv, 2)
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
