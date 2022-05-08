from enum import Enum
from colorDetection import*


black = Color("noir", 0, 1, np.array([3, 67, 30]), np.array([16, 92, 73]))  # mis à jour 1.0
brown = Color("brun", 1, 10, np.array([3, 91, 52]), np.array([13, 101, 97]))  # mis à jour 1.0
red = Color("rouge", 2, 100, np.array([169, 104, 44]), np.array([185, 255, 255]))  # mis à jour 1.0
orange = Color("orange", 3, 1e3, np.array([3, 91, 88]), np.array([14, 255, 255]))  # mis à jour 1.0
yellow = Color("jaune", 4, 10e3, np.array([0, 0, 0]), np.array([0, 0, 0]))
green = Color("vert", 5, 100e3, np.array([21, 39, 62]), np.array([31, 49, 102]))  # mis à jour 1.0
blue = Color("bleu", 6, 1e6, np.array([140, 28, 46]), np.array([168, 45, 117]))  # mis à jour 1.0
violet = Color("violet", 7, 10e6, np.array([0, 0, 0]), np.array([0, 0, 0]))  #
grey = Color("gris", 8, 100e6, np.array([11, 44, 74]), np.array([17, 57, 94]))  # mis à jour 1.0
white = Color("blanc", 9, 1e9, np.array([16, 15, 105]), np.array([26, 25, 145]))  # mis à jour 1.0
gold = Color("gold", -1, 0, np.array([2, 70, 81]), np.array([6, 80, 106]))  # mis à jour 1.0

# background colors
brown_background = Color("brown_background", -3, 0, np.array([7, 64, 75]), np.array([17, 75, 130]))
blue_background = Color("blue_background", -4, 0, np.array([[87, 99, 75]]), np.array([113, 164, 159]))
green_background = Color("green_background", -5, 0, np.array([50, 46, 38]), np.array([76, 66, 118]))
rose_background = Color("rose background", -6, 0, np.array([148, 31, 73]), np.array([175, 88, 170]))
rose_background_not_corrected = Color("rose background", -7, 0, np.array([126, 33, 49]), np.array([156, 61, 129]))
grey_reflect = Color("grey reflect", -9, 0, np.array([2, 32, 70]), np.array([22, 52, 150]))

label = "img"
color_array = 0

img_ref = cv2.imread(r'C:\Users\henri\Documents\HELHa\ProjetRD_image\vignettage picam.jpg')
img_ref = np.intc(img_ref)
correction = np.intc(img_ref-128)


class State(Enum):
    init = 0
    detect = 1
    reject = 2
    check_crop = 3
    no_resistor = 4
    calculate = 5
    vignetting = 6


def detect_resistor(img, img_background):
    """1.1) Retirer l'arrière plan"""
    img_background_masked = remove_background(img, img_background)
    display_image("arrière plan retiré", img_background_masked)

    """ 1.2) Supprimer le rose résiduel restant sur la photo """
    img_background_masked = rose_background_not_corrected.delete_color(img_background_masked,img_preprocessing(img_background_masked))
    #display_image("rose took off", img_background_masked)

    #img_2 = brown_background.delete_color(img_background_masked,img_preprocessing(img_background_masked))
    #display_image(label, img_2)

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
        return State.no_resistor, 0
        # mettre à jour le background
    elif 2000 < resistor_pixel <= 4500:
        print("1 résistance")
        return State.check_crop, cv2.bitwise_and(img, img, mask=mask_thresh)
    else:
        print("2 résistances")
        return State.reject, 0


def check_crop_img(img):
    img_hsv = img_preprocessing(img)
    """2.1) Vérification de la présence de résistance avec fond vert ou bleu"""
    if blue_background.detect_number_resistor(img, img_hsv) == 0 and green_background.detect_number_resistor(img, img_hsv) == 0:
        print("Resistance correcte détectée")
        """2.1 Résistanc OK, on peut rogner la photo pour se concentrer sur celle-ci"""
        return State.calculate, brown_background.reshape_resistor(img, img_hsv)
    else:
        print("mauvaise résistance")
        #envoyer code arduino pour ejecter la résistance
        return State.reject, 0


def locate_color(img):
    img_hsv = img_preprocessing(img)
    '''black.get_center(img, img_hsv)
    brow.get_center(img, img_hsv)
    red.get_center(img, img_hsv)
    orange.get_center(img, img_hsv)
    yellow.get_center(img, img_hsv)
    green.get_center(img, img_hsv)
    blue.get_center(img, img_hsv)
    violet.get_center(img, img_hsv)
    grey.get_center(img, img_hsv)
    white.get_center(img, img_hsv)'''

    color_array = np.array([black.get_color_array_format(img, img_hsv),
                            brown.get_color_array_format(img, img_hsv),
                            red.get_color_array_format(img, img_hsv),
                            orange.get_color_array_format(img, img_hsv),
                            yellow.get_color_array_format(img, img_hsv),
                            green.get_color_array_format(img, img_hsv),
                            blue.get_color_array_format(img, img_hsv),
                            violet.get_color_array_format(img, img_hsv),
                            grey.get_color_array_format(img, img_hsv),
                            white.get_color_array_format(img, img_hsv),
                            gold.get_color_array_format(img, img_hsv)])

    return color_array


def correct_vignetting(img):
    new_image = np.add(img, -correction)
    return np.ubyte(new_image)

def apply_correct_vignetting_on_cropped(img, img_background):
    img_masked = remove_background(correct_vignetting(img), correct_vignetting(img_background))
    display_image('vignettage corrigé', img_masked)
    img_masked = grey_reflect.delete_color(img_masked, img_preprocessing(img_masked))
    return rose_background.delete_color(img_masked, img_preprocessing(img_masked))

