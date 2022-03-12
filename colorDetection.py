import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def get_linear_regression(color_array, display=None):
    creation_array = False
    for i in range(len(color_array[:, 0])):
        # print(i)
        if color_array[i, 0] == 0:
            if not creation_array:
                list_elem_delete = np.array(i)
                creation_array = True
            else:
                list_elem_delete = np.append(list_elem_delete, i)
            # print("delete index")

    color_array_without_0 = np.delete(color_array, list_elem_delete, axis=0)
    # print("color_array_without_0 v2")
    # print(color_array_without_0)

    x = color_array_without_0[:, 0]
    x = x[:, np.newaxis]
    # print(x.shape)
    y = color_array_without_0[:, 1]
    # print (y.shape)

    model = LinearRegression()
    model.fit(x, y)

    #version2
    for i in range(len(color_array_without_0[:,0])):
        m_2 = -1/model.coef_[0]
        p_2 = color_array_without_0[i, 1] - (m_2*color_array_without_0[i, 0])
        a = np.array([[1.0, -1*model.coef_[0]],
                      [1.0, -1*m_2]])
        b = np.array([model.intercept_, p_2])
        color_array_without_0[i, 5], color_array_without_0[i, 4] = np.linalg.solve(a, b)

        if display == 'projection':
            print("point à projeter : ", color_array_without_0[i, 0], ";", color_array_without_0[i, 1])
            print("point de projection: ", color_array_without_0[i, 4], ";", color_array_without_0[i, 5])

    if display is not None:
        x_fit = np.linspace(0, 50, 1000)
        # 2eme facon de creer le vecteur de 1000 valeurs entre 0 et 10 régulierement espacees
        x_fit = x_fit[:, np.newaxis]
        x_fit_s2 = x_fit.shape
        y_fit = model.predict(x_fit)  # Creation vecteur y_fit à partir de x_fit par prediction
        print("y = ", model.coef_[0], " x + ", model.intercept_)
        plt.scatter(x, y)
        plt.scatter(24.65, 42.61)
        plt.scatter(color_array_without_0[:, 4], color_array_without_0[:, 5])
        plt.plot([x_fit[0, 0], x_fit[999, 0]], [y_fit[0], y_fit[999]], 'r', )
        plt.axis('equal')
        plt.show()

    # il faut parcourir la droite afin de déterminer l'ordre des points qu'on rencontre au fur et à mesure que l'on traverse la droite
    # taille de l'image 60x83

    """   k = 0
    for i in range(0, 83):
        y_output = i*model.coef_[0] + model.intercept_
        #print(y_output)
        for j in range(len(color_array_without_0[:, 1])):
            if y_output == color_array_without_0[j, 1]:
                color_array_without_0[j, 3] = k
                print("centroide détecté ", k)
                k += 1
    color_array_without_0 = color_array_without_0[color_array_without_0[:, 3].argsort()]
    print(color_array_without_0)
"""


def find_contour_image(img):
    output_color = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    gray_output_color = cv2.cvtColor(output_color, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(src=gray_output_color, thresh=90, maxval=255, type=cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros(img.shape)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 1)
    plt.imshow(img_contours)
    plt.show()


def calculate_resistor(array):
    print("hello")


def display_image(label, image, img_masked=None):
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


'''
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
'''


class Color:
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
        self.cx = 0
        self.cy = 0
        self.img_masked = 0
        self.order = 0
        self.projection_linear_reg = 0
        self.cx_proj = 0.0
        self.cy_proj = 0.0

    def get_masked_image(self, img, img_hsv, display=None):
        """
        Give the img with only the desired color
        :param img: orignal image
        :param img_hsv: img at the HSV format
        :return: img
        """
        """Get the image masked (image with only the color wanted"""
        color_mask = cv2.inRange(img_hsv, self.dark_color, self.light_color)
        self.img_masked = cv2.bitwise_and(img, img, mask=color_mask)
        if display == 0:
            display_image(self.color_name + " masked", self.img_masked)
        elif display == 1:
            display_image(self.color_name + " masked", img, self.img_masked)
        # return cv2.bitwise_and(img, img, mask=color_mask)

    def get_nonzero_pixel(self):
        output_color = cv2.cvtColor(self.img_masked, cv2.COLOR_HSV2BGR)
        gray_output_color = cv2.cvtColor(output_color, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(src=gray_output_color, thresh=0, maxval=255, type=cv2.THRESH_BINARY)
        points = cv2.findNonZero(thresh)
        display_image("blue tresh", thresh)
        print(points)
        # bound_rect = cv2.boundingRect(points)

    def get_center(self, img, img_hsv, display=None):
        """
        Calculate the centroïde of the masked_image
        :param display: !=1 : matplotlib display
        :return:
        """
        self.get_masked_image(img, img_hsv)
        output_color = cv2.cvtColor(self.img_masked, cv2.COLOR_HSV2BGR)
        gray_output_color = cv2.cvtColor(output_color, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(src=gray_output_color, thresh=0, maxval=255, type=0)
        m = cv2.moments(thresh)
        if m["m00"] != 0:
            self.cx = int(m["m10"] / m["m00"])
            self.cy = int(m["m01"] / m["m00"])
        else:
            self.cx, self.cy = 0, 0
            print("The color", self.color_name, " is not on the resistor")
            return self.cx, self.cy

        print("Centroïde", self.color_name, " : ", self.cx, ",", self.cy)
        if display is not None:
            cv2.circle(output_color, (self.cx, self.cy), 2, (255, 255, 0), -1)
            if display == 1:
                display_image("Centroïde", output_color)
            else:
                plt.imshow(output_color)
                plt.show()

        return self.cx, self.cy

    def get_color_array_format(self, img, img_hsv):
        self.get_center(img, img_hsv)
        return self.cx, self.cy, self.value, self.order, self.cx_proj, self.cy_proj
