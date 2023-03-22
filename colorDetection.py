import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math


def remove_background(img, img_background):
    difference = cv2.absdiff(img, img_background)
    # display_image(label, difference)
    gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    mask_ret, mask_thresh = cv2.threshold(gray_difference, 10, 255, cv2.THRESH_BINARY)
    # display_image(label, mask_thresh)
    return cv2.bitwise_and(img, img, mask=mask_thresh)


def img_preprocessing(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def first_reshape(img_original, display_cropped = False) :
    img = np.copy(img_original)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    #display_image(label, s)
    s_ret, s_thresh = cv2.threshold(s, 100, 255, cv2.THRESH_BINARY)
    #display_image(label, s_thresh)
    img = cv2.bitwise_and(img, img, mask=s_thresh)
    #display_image(label, v)
    v_ret, v_thresh = cv2.threshold(v, 70, 255, cv2.THRESH_BINARY)
    #display_image(label, v_thresh)
    img = cv2.bitwise_and(img, img, mask=v_thresh)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), cv2.BORDER_DEFAULT)
    img_edge = cv2.Canny(img_gray, 50, 150)
    half_size = 150
    all_center_x = np.empty(10, dtype=int)
    all_center_y = np.empty(10, dtype=int)
    all_h = np.empty(10, dtype=int)
    all_w = np.empty(10, dtype=int)

    all_new_min_x = np.empty(10, dtype=int)
    all_new_min_y = np.empty(10, dtype=int)

    for i in range(0, 10):
        #print(i)
        M = cv2.moments(img_edge)
        # print(M)
        if M["m00"] == 0:
            #print("error : m00 = 0")
            break

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        """print("center found :")
        print(cX)
        print(cY)"""

        all_center_x.itemset(i, cX)
        all_center_y.itemset(i, cY)
        img_edge_centroid = np.copy(img_edge)

        new_min_x = cX - half_size if cX - half_size > 0 else 0
        new_max_x = cX + half_size if cX + half_size < img_edge.shape[0] else img_edge.shape[1] - 1

        new_min_y = cY - half_size if cY - half_size > 0 else 0
        new_max_y = cY + half_size if cY + half_size < img_edge.shape[1] else img_edge.shape[0] - 1

        """print("X limits : " + str(new_min_x) + ", " + str(new_max_x))
        print("Y limits : " + str(new_min_y) + ", " + str(new_max_y))"""

        all_new_min_x.itemset(i, new_min_x)
        all_new_min_y.itemset(i, new_min_y)
        img_edge = img_edge[new_min_y:new_max_y, new_min_x:new_max_x]
        img_edge_centroid = img_edge_centroid[new_min_y:new_max_y, new_min_x:new_max_x]
        cv2.circle(img_edge_centroid, (cX, cY), 5, (255, 255, 255), -1)
        all_w.itemset(i, img_edge.shape[0])
        all_h.itemset(i, img_edge.shape[1])

        # display_image(label,img_edge_centroid)
        half_size = half_size - 10

    cv2.circle(img_edge, (int(img.shape[0] / 2), int(img.shape[1] / 2)), 5, (255, 255, 255), -1)
    # display_image(label,img_edge)
    center_x = 0
    center_y = 0
    for i in range(0, 10):
        #print(str(center_x) + "+" + str(all_new_min_x[i]) + "=" + str(center_x + all_new_min_x[i]))
        center_x = center_x + all_new_min_x[i]
        #print(center_x)
        center_y = center_y + all_new_min_y[i]

    img_point = np.copy(img_original)
    cv2.circle(img_point, (int(center_x), int(center_y)), 5, (255, 255, 255), -1)
    cv2.rectangle(img_point, (center_x, center_y), (center_x + all_w[9], center_y + all_h[9]), (0, 255, 0), 2)
    img_cropped = img_original[center_y : center_y + all_h[9], center_x : center_x + all_w[9]]
    path = r'C:\Users\henri\Documents\HELHa\ProjetRD_image\resistor_cropped.jpg'
    cv2.imwrite(path, img_cropped)
    #print(img_point.shape)
    #display_image("allez", img_point)
    return img_cropped


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

    # version2
    for i in range(len(color_array_without_0[:, 0])):
        m_2 = -1 / model.coef_[0]
        p_2 = color_array_without_0[i, 1] - (m_2 * color_array_without_0[i, 0])
        a = np.array([[1.0, -1 * model.coef_[0]],
                      [1.0, -1 * m_2]])
        b = np.array([model.intercept_, p_2])
        color_array_without_0[i, 5], color_array_without_0[i, 4] = np.linalg.solve(a, b)
        if display == 'projection':
            print("point à projeter : ", color_array_without_0[i, 0], ";", color_array_without_0[i, 1])
            print("point de projection: ", color_array_without_0[i, 4], ";", color_array_without_0[i, 5])

    if np.min(color_array_without_0[:, 4]) > np.min(color_array_without_0[:, 5]):
        # si la distance sur les x est plus grande que sur les y alors on trie selon les x
        axis = 4
    else:
        # sinon la distance sur les y est plus grande que sur les x donc on fixe le tri sur les y
        axis = 5
    color_array_without_0 = color_array_without_0[color_array_without_0[:, axis].argsort()]

    print(color_array_without_0)
    print("minimum value", np.min(color_array_without_0[:, 4]))

    if display is not None:
        x_fit = np.linspace(0, 50, 1000)
        # 2eme facon de creer le vecteur de 1000 valeurs entre 0 et 10 régulierement espacees
        x_fit = x_fit[:, np.newaxis]
        x_fit_s2 = x_fit.shape
        y_fit = model.predict(x_fit)  # Creation vecteur y_fit à partir de x_fit par prediction
        # print("y = ", model.coef_[0], " x + ", model.intercept_)
        plt.scatter(x, y)
        plt.scatter(color_array_without_0[:, 4], color_array_without_0[:, 5])
        plt.plot([x_fit[0, 0], x_fit[999, 0]], [y_fit[0], y_fit[999]], 'b', )
        plt.axis('equal')
        plt.show()

    return color_array_without_0


def find_contour_image(img):
    output_color = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    gray_output_color = cv2.cvtColor(output_color, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(src=gray_output_color, thresh=90, maxval=255, type=cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros(img.shape)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 1)
    plt.imshow(img_contours)
    plt.show()


def calculate_resistor(color_array_treated):
    # on dispose d'un tableau dont l'ordre correspond à l'ordre des couleurs, il faut maintenant calculer la valeur
    # de la résistance à partir des valeurs du tableau
    # ! Il faut encore regarder dans quel ordre on lit les valeurs : pour le moment on se refère à la couleur gold, qui
    # donne la valeur de la précision de la résistance et qui n'est pas utilisée pour les valeurs de la résistance
    resistor_value = 0
    order = 0
    # Déterminer l'emplacement de la couleur gold :
    if len(color_array_treated) == 4:
        if color_array_treated[len(color_array_treated) - 1, 2] == -1:
            order = 1
        elif color_array_treated[0, 2] == -1:
            order = 1
            color_array_treated = np.flip(color_array_treated, axis=0)
            #print("flip color_array_treated")
            # print(color_array_treated)

        if order != 0:
            for i in range(len(color_array_treated[:, 0]) - 2):
                # print("valeur à ajouter : " + str(color_array_treated[i,2]))
                resistor_value += color_array_treated[i, 2] * math.pow(10, len(color_array_treated[:, 2]) - i - 3)
                # print("degré du 10 :", len(color_array_treated[:, 2]) -i -3)
                # print(resistor_value)
            resistor_value *= math.pow(10, color_array_treated[len(color_array_treated[:, 2]) - 2, 2])
            print("Valeur de la résistance : ", resistor_value)
        else:
            print("impossible de calculer la valeur de la résistance")
    else:
        print("Erreur dans la reconnaissance des couleurs")
        #envoyer code erreur I2C


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
    def __init__(self, color_name, value, multiplier, lower_color, upper_color):
        self.value = value
        self.multiplier = multiplier
        self.lower_color = lower_color
        self.upper_color = upper_color
        self.color_name = color_name
        self.cx = 0
        self.cy = 0
        self.img_masked = 0
        self.order = 0
        self.projection_linear_reg = 0
        self.cx_proj = 0.0
        self.cy_proj = 0.0

    def delete_color(self, img, img_hsv):
        color_mask = cv2.inRange(img_hsv, self.lower_color, self.upper_color)
        self.img_masked = cv2.bitwise_and(img, img, mask=color_mask)
        return img - self.img_masked

    def get_masked_image(self, img, img_hsv, display=None):
        """
        Give the img with only the desired color
        :param img: orignal image
        :param img_hsv: img at the HSV format
        :return: img
        """
        """Get the image masked (image with only the wanted color)"""
        color_mask = cv2.inRange(img_hsv, self.lower_color, self.upper_color)
        self.img_masked = cv2.bitwise_and(img, img, mask=color_mask)
        #print(color_mask.shape)
        if display == 0:
            display_image(self.color_name + " masked", self.img_masked)
        elif display == 1:
            display_image(self.color_name + " masked", img, self.img_masked)
        elif display == 'plot':
            plt.imshow(self.img_masked)
            plt.show()
        elif display == 'return':
            return self.img_masked
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
                display_image("Centroïde", thresh)
            elif display == 2:
                np.set_printoptions(threshold=np.inf)
                print(thresh)
                # print('sum :' + str(np.sum(output_color)))
            elif display == 3:
                print("shape : " + str(thresh.shape))
                print("sum : " + str(np.sum(output_color) / 255))
            else:
                plt.imshow(output_color)
                plt.show()

        return self.cx, self.cy

    def get_color_array_format(self, img, img_hsv):
        self.get_center(img, img_hsv,4)
        return self.cx, self.cy, self.value, self.order, self.cx_proj, self.cy_proj

    def detect_number_resistor(self, img, img_hsv, display=None):
        self.get_masked_image(img, img_hsv)
        output_color = cv2.cvtColor(self.img_masked, cv2.COLOR_HSV2BGR)
        gray_output_color = cv2.cvtColor(output_color, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(src=gray_output_color, thresh=0, maxval=255, type=0)
        resistor_pixel = int(np.sum(thresh) / 255)
        if display == 1:
            display_image("masque", thresh)
        elif display == 2:
            print("resistor pixel " + self.color_name + " " + str(resistor_pixel))

        if resistor_pixel <= 100:
            #print("no resistance")
            return 0
        elif 100 < resistor_pixel < 500:
            #print("1 resistor detected")
            return 1

    def reshape_resistor(self, img, img_hsv, save = None):
        self.get_masked_image(img, img_hsv)
        """plt.imshow(self.img_masked)
        plt.show()"""
        img_masked_non_zero = np.nonzero(self.img_masked)
        #print(img_masked_non_zero)
        self.img_masked[img_masked_non_zero]
        img_masked_transpose_non_zero = np.transpose(np.nonzero(self.img_masked))
        img_masked_transpose_non_zero_delete = np.delete(img_masked_transpose_non_zero, 2, 1)
        max_border = np.amax(img_masked_transpose_non_zero_delete, axis = 0)
        """print("amax")
        print(max_border)"""
        min_border = np.amin(img_masked_transpose_non_zero_delete, axis = 0)
        """print("amin")
        print(min_border)"""
        border_accuracy = 4
        img_cropped = img[min_border[0] - border_accuracy: max_border[0] + border_accuracy, min_border[1] - border_accuracy : max_border[1] + border_accuracy]
        if save is not None :
            path = r'C:\Users\henri\Documents\HELHa\ProjetRD_image\resistor_cropped_2.jpg'
            cv2.imwrite(path, img_cropped)
        return img_cropped

