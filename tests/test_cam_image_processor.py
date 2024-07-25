import sys
sys.path.append('./')
import unittest
from src.cam_image_processor import CamImageProcessor
import os
import cv2
import yaml


class TestCIP(unittest.TestCase):

    def setUp(self):
        directory = "data"
        self.data = []

        with open('config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
            
        for yaml_file in os.listdir(directory):
            if yaml_file.endswith(".yaml"):
                image_file = os.path.join(directory, yaml_file.replace(".yaml", ".jpg"))
                if os.path.isfile(image_file):
                    self.data.append((os.path.join(directory, yaml_file), image_file))
                continue
            else:
                continue

    def test_leds(self):
        led_names = ["S1", "S2", "S3", "S4", "K1", "K2", "K3", "Status"]

        for yaml_file, image_file in self.data:
            print(f"Checking image {image_file} with yaml {yaml_file}")
            with open(yaml_file, 'r') as file:
                image_data = yaml.safe_load(file)

            image = cv2.imread(image_file)
            cip = CamImageProcessor(self.config, image)
            led_states = cip.get_led_dict()

            led_states_correct = True

            if image_data != None or len(led_states) != 0: # if both are zero, we detected a faulty picture alright
                for led in led_names:

                    print(f"Checking led {led} with color {led_states[led]} and expected color {image_data[led]}")

                    if led_states[led] != image_data[led]:
                        led_states_correct = False
                        break
            else:
                print("Successfully detected faulty picture.")

            self.assertTrue(led_states_correct)

if __name__ == '__main__':
    unittest.main()

