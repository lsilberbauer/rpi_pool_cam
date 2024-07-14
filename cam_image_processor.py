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

    led_states = {}
    for led in list(config["leds"].keys())[2:]:
        color = get_led_color(leds, config["leds"][led]["origin"], config["leds"][led]["radius"])
        euclidian_distance = math.dist(color, tuple(config["leds"][led]["color"]))
        led_states[led] = True if euclidian_distance < 100 else False

    fill_level = 0

    if led_states["S1"]:
        fill_level = 1
    elif led_states["S2"]:
        fill_level = 0.75
    elif led_states["S3"]:
        fill_level = 0.5
    elif led_states["S4"]:
        fill_level = 0.25

    valid = False

    if led_states["Status"] and led_states["Black"]:
        valid = True

    return_json = f"{{ \"Valid\": \"{valid}\", \"K1\": \"{led_states['K1']}\", \"K2\": \"{led_states['K2']}\", \"K3:\" \"{led_states['K3']}\", \"Error\": \"{led_states['Error']}\", \"FillLevel\": \"{fill_level}\" }}"

    if (return_annotated):
        annotated_leds = leds.copy()

        draw_rectangle(annotated_leds, config["leds"]["rectangle_origin"], config["leds"]["rectangle_size"])
        draw_rectangle(annotated_leds, config["lcd"]["rectangle_origin"], config["lcd"]["rectangle_size"])        

        for led in list(config["leds"].keys())[2:]:
            annotated_leds = annotate_led(annotated_leds, config["leds"][led]["origin"], config["leds"][led]["radius"], (200, 0, 200))

        return annotated_leds
    else:
        return return_json

