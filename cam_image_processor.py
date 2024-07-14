import yaml
import cv2 
from matplotlib import pyplot as plt
import numpy as np
import math

def draw_rectangle(image, rectangle_origin, rectangle_size):
    color = (0, 255, 255) 
    thickness = 2

    start_point = (rectangle_origin[0], rectangle_origin[1])
    end_point = (rectangle_origin[0] + rectangle_size[0], rectangle_origin[1] + rectangle_size[1])

    return cv2.rectangle(image, start_point, end_point, color, thickness)

def crop_rectangle(image, rectangle_origin, rectangle_size):
    return image[rectangle_origin[1]:rectangle_origin[1]+rectangle_size[1], rectangle_origin[0]:rectangle_origin[0]+rectangle_size[0]]

def annotate_led(img, origin, radius, color):
    leds_visual = img.copy()    
    cv2.circle(leds_visual, (origin[0], origin[1]), radius, color, 1)
    return leds_visual

def get_led_color(img, origin, radius):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_empty_indicator = np.zeros_like(gray)
    cv2.circle(mask_empty_indicator, (origin[0], origin[1]), radius, 255, -1)
    return cv2.mean(img, mask=mask_empty_indicator)[:3]

def get_led_status(image, return_annotated=False):

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    leds = crop_rectangle(image, config["leds"]["rectangle_origin"], config["leds"]["rectangle_size"])
    lcd = crop_rectangle(image, config["lcd"]["rectangle_origin"], config["lcd"]["rectangle_size"])

    leds_dict = { 
        "K1": (255, 0, 0), 
        "K2": (200, 20, 20), 
        "K3": (150, 50, 50),
        "S1": (0, 255, 0 ),
        "S2": (40, 200, 40 ),
        "S3": (80, 150, 80 ),
        "S4": (120, 100, 120 ),
        "Status": (0, 200, 200 ),
        "Black": (0, 0, 200)
        }

    led_colors = {}
    for led in leds_dict:
        led_colors[led] = get_led_color(leds, config["leds"][led]["origin"], config["leds"][led]["radius"])

    return_json = "{ "
    for color in led_colors:
        euclidian_distance = math.dist(led_colors[color], tuple(config["leds"][color]["color"]))
        status = "1"
        if euclidian_distance < 100: 
            status = "0"
        return_json = return_json + (f"\"{color}\": {status}, ")

    return_json = return_json + " }"

    if (return_annotated):
        image = draw_rectangle(image, config["leds"]["rectangle_origin"], config["leds"]["rectangle_size"])
        image = draw_rectangle(image, config["lcd"]["rectangle_origin"], config["lcd"]["rectangle_size"])

        annotated_leds = leds.copy()

        for led in leds_dict:
            annotated_leds = annotate_led(annotated_leds, config["leds"][led]["origin"], config["leds"][led]["radius"], leds_dict[led])

        return annotated_leds
    else:
        return return_json

