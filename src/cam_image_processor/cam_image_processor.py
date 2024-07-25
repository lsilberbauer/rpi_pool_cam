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

class CamImageProcessor:
    def __init__(self, config, img):
        self.config = config

        self.img = img
        self.lcd = crop_rectangle(img, self.config["lcd"]["rectangle_origin"], self.config["lcd"]["rectangle_size"])

        gray = cv2.cvtColor(self.lcd, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 1:
            raise Exception("Multiple contours for LCD found, aborting.")

        peri = cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], 0.04 * peri, True)

        lcd_offset = approx[0][0] - (17, 17)
        offsetted_origin = tuple(map(sum, zip(self.config["leds"]["rectangle_origin"], lcd_offset)))
        self.leds = crop_rectangle(img, offsetted_origin, self.config["leds"]["rectangle_size"])

    def get_led_dict(self):

        led_states = {}
        for led in list(self.config["leds"].keys())[2:]:
            color = get_led_color(self.leds, self.config["leds"][led]["origin"], self.config["leds"][led]["radius"])
            euclidian_distance = math.dist(color, tuple(self.config["leds"][led]["color"]))
            led_states[led] = True if euclidian_distance < 100 else False

        valid = True if led_states["Status"] and led_states["Black"] else False

        if valid:
            return led_states
        else:
            return {}        

    def get_led_json(self):
        led_states = self.get_led_dict()

        if len(led_states) > 0:
            fill_level = 0

            if led_states["S1"]:
                fill_level = 1
            elif led_states["S2"]:
                fill_level = 0.75
            elif led_states["S3"]:
                fill_level = 0.5
            elif led_states["S4"]:
                fill_level = 0.25

            return f"{{ \"Valid\": \"True\", \"K1\": \"{led_states['K1']}\", \"K2\": \"{led_states['K2']}\", \"K3\": \"{led_states['K3']}\", \"Error\": \"{led_states['Error']}\", \"FillLevel\": \"{fill_level}\" }}"

        else: return ""

    def get_led_annotations(self):
        
        annotated_leds = self.leds.copy()
        draw_rectangle(annotated_leds, self.config["leds"]["rectangle_origin"], self.config["leds"]["rectangle_size"])        

        for led in list(self.config["leds"].keys())[2:]:
            annotated_leds = annotate_led(annotated_leds, self.config["leds"][led]["origin"], self.config["leds"][led]["radius"], (200, 0, 200))

        return annotated_leds
