import cv2
import numpy as np
import matplotlib as plt


def display_image(label, image):
    cv2.imshow(label, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


path = r'C:\Users\MediMonster\Documents\HELHa\ProjetRD\openCV\resistor_project\R6800.jpeg'
path_2 = r'C:\Users\MediMonster\Documents\HELHa\ProjetRD\openCV\resistor_project\color_sample.png'

img = cv2.imread(path)

# display_image("R6800", img)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

dark_blue = np.array([110, 50, 50])
light_blue = np.array([130, 255, 255])

dark_red = np.array([160, 20, 70])
light_red = np.array([190, 255, 255])

"""
dark_yellow = np.array([23, 220, 220])
light_yellow = np.array([25, 255, 230])
"""

dark_braun = np.array([15, 110, 40])
light_braun = np.array([25, 255, 180])

# Gris à refaire
dark_gray = np.array([45, 9, 83])
light_gray = np.array([50, 40, 173])

red_mask = cv2.inRange(img_hsv, dark_red, light_red)
blue_mask = cv2.inRange(img_hsv, dark_blue, light_blue)
# yellow_mask = cv2.inRange(img_hsv, dark_yellow, light_yellow)
braun_mask = cv2.inRange(img_hsv, dark_braun, light_braun)
gray_mask = cv2.inRange(img_hsv, dark_gray, light_gray)

# output_blue = cv2.bitwise_and(img, img, mask = blue_mask)
output_red = cv2.bitwise_and(img, img, mask=red_mask)
# output_yellow = cv2.bitwise_and(img, img, mask = yellow_mask)
# output_braun = cv2.bitwise_and(img, img, mask = braun_mask)
# output_gray = cv2.bitwise_and(img, img, mask = gray_mask)

display_image("Masked image", np.hstack((img, output_red)))

print("j'ai fini")
