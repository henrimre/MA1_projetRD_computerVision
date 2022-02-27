import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(label, image, img_masked = None):
    """
    Display an image in a new windows
    :param label: Title of the windows
    :param image: image to display
    :return:
    """

    if img_masked is not None:
        cv2.imshow(label, np.hstack((image, img_masked)))
    else:
        cv2.imshow(label, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def img_preprocessing(path):
    """
    Transform the image
    :param path:
    :return: img, img_hsv
    """
    return cv2.imread(path), cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV)


def init_color_object():
    black = Color(0, 1, np.array([0, 0, 0]), np.array([0, 0, 0]))
    brow = Color(1, 10, np.array([15, 110, 40]), np.array([50, 40, 173]))
    red = Color(2, 100, np.array([160, 20, 70]), np.array([190, 255, 255]))
    orange = Color(3, 1e3, np.array([0, 0, 0]), np.array([0, 0, 0]))
    yellow = Color(4, 10e3, np.array([0, 0, 0]), np.array([0, 0, 0]))
    green = Color(5, 100e3, np.array([0, 0, 0]), np.array([0, 0, 0]))
    blue = Color(6, 1e6, np.array([110, 50, 50]), np.array([130, 255, 255]))
    violet = Color(7, 10e6, np.array([0, 0, 0]), np.array([0, 0, 0]))
    grey = Color(8, 100e6, np.array([0, 0, 0]), np.array([0, 0, 0]))
    white = Color(0, 1e9, np.array([0, 0, 0]), np.array([0, 0, 0]))


class Color :
    def __init__(self, color_name, value, multiplier, dark_color, light_color):
        """
        Constructor of the object Color
        :param value: The numeric value of the color
        :param multiplier: The multiplier value of the color
        :param dark_color: The dark_color as np.array (HSV format)
        :param light_color: The light_color as np.array (HSV format)
        """
        self.value = value
        self.multiplier = multiplier
        self.dark_color = dark_color
        self.light_color = light_color
        self.color_name = color_name

    def get_masked_image(self, img, img_hsv):
        """
        Give the img with only the desired color
        :param img: orignal image
        :param img_hsv: img at the HSV format
        :return: img
        """
        """Get the image masked (image with only the color wanted"""
        color_mask = cv2.inRange(img_hsv, self.dark_color, self.light_color)
        return cv2.bitwise_and(img, img, mask=color_mask)

    def get_center(self, img_masked, display = None):
        """
        Give the center of all the pixel on an image
        :param img_masked:
        :return: center_x, center_y
        """
        output_color = cv2.cvtColor(img_masked, cv2.COLOR_HSV2BGR)
        gray_output_color = cv2.cvtColor(output_color, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(src=gray_output_color, thresh=127, maxval=255, type=0)
        m = cv2.moments(thresh)
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        if display is not None:
            print("Centroïde", self.color_name," : ", cx, ",", cy)
            cv2.circle(output_color, (cx, cy), 2, (255, 255, 0), -1)
            #cv2.putText(output_color, "centroid", (cx - 25, cy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if display == 1 :
                display_image("Centroïde", output_color)
            else :
                plt.imshow(output_color)
                plt.show()

        return cx, cy





