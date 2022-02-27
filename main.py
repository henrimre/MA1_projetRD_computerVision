import cv2
import numpy as np
import matplotlib.pyplot as plt
from colorDetection import *

path = r'C:\Users\MediMonster\Documents\HELHa\ProjetRD\openCV\resistor_project\R6800.jpeg'
path_2 = r'C:\Users\MediMonster\Documents\HELHa\ProjetRD\openCV\resistor_project\color_sample.png'
img, img_hsv = img_preprocessing(path)

black = Color("noir", 0, 1, np.array([0, 0, 0]), np.array([0, 0, 0]))
brow = Color("brun", 1, 10, np.array([15, 110, 40]), np.array([50, 40, 173]))
red = Color("rouge", 2, 100, np.array([160, 20, 70]), np.array([190, 255, 255]))
orange = Color("orange", 3, 1e3, np.array([0, 0, 0]), np.array([0, 0, 0]))
yellow = Color("jaune", 4, 10e3, np.array([0, 0, 0]), np.array([0, 0, 0]))
green = Color("vert", 5, 100e3, np.array([0, 0, 0]), np.array([0, 0, 0]))
blue = Color("bleu", 6, 1e6, np.array([110, 50, 50]), np.array([130, 255, 255]))
violet = Color("violet", 7, 10e6, np.array([0, 0, 0]), np.array([0, 0, 0]))
grey = Color("gris", 8, 100e6, np.array([0, 0, 0]), np.array([0, 0, 0]))
white = Color("blanc", 0, 1e9, np.array([0, 0, 0]), np.array([0, 0, 0]))


red.img_masked = red.get_masked_image(img, img_hsv)

display_image("Red masked", red.img_masked)

display_image("Red masked", img, red.img_masked)

red.get_center(red.img_masked, 2)

print("j'ai fini")