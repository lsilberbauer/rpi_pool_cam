import time
import io
import threading
import picamera
import datetime as dt
from picamera import PiCamera
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from fractions import Fraction
import numpy as np
import cv2
import cam_image_processor
from time import localtime, strftime

# Portnummer für den Webserver
PORT_NUMBER = 8000

# Klassendefinition für den Bild-Stream
class ImageStream:
    def __init__(self):
        self.frame = None
        self.frame_lock = threading.Lock()

    def update_frame(self, frame):
        with self.frame_lock:
            self.frame = frame
        print("Frame updated")

    def get_frame(self):
        with self.frame_lock:
            return self.frame        

# Bild-Stream-Objekt erstellen
image_stream = ImageStream()

# Handler-Klasse für den HTTP-Server
class MyHandler(BaseHTTPRequestHandler):
    # GET-Anfrage verarbeiten
    def do_GET(self):
        global image_stream
        
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            # HTML-Antwort senden, die das aktuelle Bild anzeigt
            self.wfile.write('<html><body>'.encode('utf-8'))
            self.wfile.write('<img src="image.jpg">'.encode('utf-8'))
            self.wfile.write('<img src="digital.jpg">'.encode('utf-8'))
            self.wfile.write('<img src="leds_annotated.jpg">'.encode('utf-8'))
            self.wfile.write('</body></html>'.encode('utf-8'))
        elif self.path == '/image.jpg':
            self.send_response(200)
            self.send_header('Cache-Control', 'no-store, must-revalidate')
            self.send_header('Expires', '0')
            self.send_header('Content-type', 'image/jpeg')
            self.end_headers()
            # Aktuelles Bild als Antwort senden
            frame = image_stream.get_frame()            
            if frame is not None:
                jpg = cv2.imencode('.jpg', frame)[1]
                self.wfile.write(jpg)
        elif self.path == '/digital.jpg':
            self.send_response(200)
            self.send_header('Cache-Control', 'no-store, must-revalidate')
            self.send_header('Expires', '0')
            self.send_header('Content-type', 'image/jpeg')
            self.end_headers()
            # Aktuelles Bild als Antwort senden
            frame = image_stream.get_frame()            
            if frame is not None:
                digital = frame[1157:1157+80, 1835:1835+115].copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                time = strftime("%H:%M:%S", localtime())
                digital = cv2.putText(digital, time, 
                    (55, 75), font, 0.4,
                    (210, 155, 155), 1, cv2.LINE_8)
                jpg = cv2.imencode('.jpg', digital)[1]
                self.wfile.write(jpg)              
        elif self.path == '/leds.jpg':
            self.send_response(200)
            self.send_header('Cache-Control', 'no-store, must-revalidate')
            self.send_header('Expires', '0')
            self.send_header('Content-type', 'image/jpeg')
            self.end_headers()
            # Aktuelles Bild als Antwort senden
            frame = image_stream.get_frame().copy()            
            if frame is not None:
                leds = frame[759:759+80, 377:377+115]
                font = cv2.FONT_HERSHEY_SIMPLEX
                time = strftime("%H:%M:%S", localtime())
                leds = cv2.putText(leds, time, 
                    (55, 75), font, 0.4,
                    (210, 155, 155), 1, cv2.LINE_8)
                jpg = cv2.imencode('.jpg', leds)[1]
                self.wfile.write(jpg)                    
        elif self.path == '/leds_annotated.jpg':
            self.send_response(200)
            self.send_header('Cache-Control', 'no-store, must-revalidate')
            self.send_header('Expires', '0')
            self.send_header('Content-type', 'image/jpeg')
            self.end_headers()

            frame = image_stream.get_frame().copy()
            if frame is not None:
                leds_annotated = cam_image_processor.get_led_status(frame, True)
                font = cv2.FONT_HERSHEY_SIMPLEX
                time = strftime("%H:%M:%S", localtime())
                leds = cv2.putText(leds_annotated, time, 
                    (55, 75), font, 0.4,
                    (210, 155, 155), 1, cv2.LINE_8)
                jpg = cv2.imencode('.jpg', leds_annotated)[1]
                self.wfile.write(jpg)
        elif self.path == '/leds_json':
            self.send_response(200)
            self.send_header('Cache-Control', 'no-store, must-revalidate')
            self.send_header('Expires', '0')
            self.send_header('Content-type', 'text/xml')
            self.end_headers()

            frame = image_stream.get_frame()
            if frame is not None:
                leds_status = cam_image_processor.get_led_status(frame, False)
                self.wfile.write(leds_status.encode('utf-8'))         
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write('404 Not Found'.encode('utf-8'))

# Funktion zum kontinuierlichen Aufnehmen von Bildern
def capture_image():
    global image_stream       
   
    while True:

        camera = PiCamera(
            resolution=(2592, 1952),
            framerate=1,
            sensor_mode=3)
        camera.led=False    
        camera.shutter_speed = 1000000
        camera.iso = 800
        camera.rotation = 180
        time.sleep(5)
        camera.exposure_mode = 'off'

        image = np.empty((1952 * 2592 * 3,), dtype=np.uint8)
        camera.annotate_text = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')       

        camera.capture(image, format='bgr')
        image = image.reshape((1952, 2592, 3))
        image_stream.update_frame(image)
        camera.close()
        time.sleep(60)

# Hauptfunktion
def main():
    # Thread für das kontinuierliche Aufnehmen von Bildern starten
    image_thread = Thread(target=capture_image)
    image_thread.daemon = True
    image_thread.start()

    # Webserver starten
    try:
        server = HTTPServer(('', PORT_NUMBER), MyHandler)
        print('Webserver läuft auf Port {}'.format(PORT_NUMBER))
        server.serve_forever()
    except KeyboardInterrupt:
        print('^C erhalten, Server wird heruntergefahren')
        server.socket.close()

if __name__ == '__main__':
    main()
