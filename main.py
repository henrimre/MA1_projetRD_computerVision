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



print("j'ai fini")