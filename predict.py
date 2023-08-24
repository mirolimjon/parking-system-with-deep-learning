from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import pytesseract as pt
import plotly.express as px
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet
from PIL import Image
import serial
import json
import requests
import time
import cv2
import os

pt.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Load model
model = tf.keras.models.load_model('./object_detection.h5')
print('Model loaded Sucessfully')
# List of allowed cars
allowed = [
    "MH 20 EE 0943\n",
    "M778.\n",
]

# Port connecction and 
def main():
    # Connecting to serial port, find the port name in your microchip programmer IDE
    sp = serial.Serial(port='COM5', baudrate=9600, timeout=1)

    # Check if connected successfully, or return
    if sp.is_open:
        print("Connected")
    else:
        print("Failed to open port :(")
        return

    # Loop while the port is still connected
    while sp.is_open:
        # Change the timeout settings
        sp.timeout = 1
        # Read the incoming stream

        # incoming_message = sp.read_until().decode().strip()
        incoming_message = '1'

        # Skip if there is no incoming message
        if not incoming_message:
            print("not incoming_message")
            continue

        try:
            # Switch the incoming messages to carry out different tasks
            if incoming_message == "1":
                # Capture a photo from the camera
                image = capture()
                
                # Predict vehicle's number plate coord
                image, cods = object_detection(image)
                # Pass the photo to check car plate numbers and if they are allowed
                is_allowed = check(image)
                print("Allowed: {}".format(is_allowed))
                if is_allowed == True:
                    # Send 0 to Arduino, which opens the gate
                    sp.write(bytes([0]))
                    print("Gate is opened")
                    sp.flush()
                    # Wait for 2 seconds until the car passes
                    time.sleep(3)
                    # Send 1 to Arduino, which closes the gate
                    sp.write(bytes([1]))
                    print("Gate has been closed")
                    # Wait for 2 seconds until the car passes
                    time.sleep(2)
                    sp.flush()
        except Exception as e:
            print(e)

    print("Port is closed :)")
    


# Capture vehicle image
def capture():
    print("Capturing photo")
    try:
        # Create a VideoCapture object with the default camera
        capture = cv2.VideoCapture(0)
        # Capture a frame
        ret, frame = capture.read()

        # Save the frame as an image file
        image_path = "image.jpeg"
        cv2.imwrite(image_path, frame)
        capture.release()

        return image_path
    except Exception as e:
        print(e)
         

# Create pipeline

def object_detection(path):
    # Read image
    image = load_img(path) # PIL object
    image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    image1 = load_img(path,target_size=(224,224))
    
    # Data preprocessing
    image_arr_224 = img_to_array(image1)/255.0 # Convert to array & normalized
    h,w,d = image.shape
    test_arr = image_arr_224.reshape(1,224,224,3)
    
    # Make predictions
    coords = model.predict(test_arr)
    
    # Denormalize the values
    denorm = np.array([w,w,h,h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    
    # Draw bounding on top the image
    xmin, xmax,ymin,ymax = coords[0]
    pt1 =(xmin,ymin)
    pt2 =(xmax,ymax)
    print(pt1, pt2)
    cv2.rectangle(image,pt1,pt2,(0,255,0),3)
    return image, coords




def check(image):
    path = 'C:/Users/User/Desktop/Final/image.jpeg'
    image, cods = object_detection(path)

    fig = px.imshow(image)
    fig.update_layout(width=700, height=500, margin=dict(l=10, r=10, b=10, t=10),xaxis_title='Figure 14')

    img = np.array(load_img(path))
    xmin ,xmax,ymin,ymax = cods[0]
    roi = img[ymin:ymax,xmin:xmax]
    fig = px.imshow(roi)
    fig.update_layout(width=350, height=250, margin=dict(l=10, r=10, b=10, t=10),xaxis_title='Figure 15 Cropped image')
    roi_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = roi_gray[ymin:ymax,xmin:xmax]


    #memory usage with image i.e. adding image to memory
    filename = "{}.jpeg".format(os.getpid())
    cv2.imwrite(filename, roi)
    text = pt.image_to_string(Image.open(filename))
    # os.remove(filename)
    print(text)
    try:
        response_data =  {"vehicles":[{
             "plate":{"plate_text":text}
            }]}
        print(response_data)
        vehicles = response_data["vehicles"]
        is_allowed = any(vehicle["plate"]["plate_text"] in allowed for vehicle in vehicles)
        # Check if there are any allowed cars
        
        return is_allowed
    except Exception as e:
        print("No cars detected")
        return False

main()