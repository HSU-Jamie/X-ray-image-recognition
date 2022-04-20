from ctypes import *  # Import libraries
import math
import random
import os
import cv2
import numpy as np
import time
import darknet


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    # Colored labels dictionary
    color_dict = {
        'none': [255, 0, 0], 'bad': [0, 255, 0], 'good': [0, 255, 0]
    }
    for detection in detections:
        x, y, w, h = detection[2][0], \
                     detection[2][1], \
                     detection[2][2], \
                     detection[2][3]
        name_tag = str(detection[0])
        for name_key, color_val in color_dict.items():
            if name_key == name_tag:
                color = color_val
                xmin, ymin, xmax, ymax = convertBack(
                    float(x), float(y), float(w), float(h))
                pt1 = (xmin, ymin)
                pt2 = (xmax, ymax)
                cv2.rectangle(img, pt1, pt2, color, 1)
                cv2.putText(img,
                            detection[0] +
                            " [" + detection[1] + "]",
                            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 2)
    return img


def YOLO():
    # global metaMain, netMain, altNames
    configPath = "cfg_mask/yolov3.cfg"  # Path to cfg
    weightPath = "cfg_mask/yolov3_1100.weights"  # Path to weights
    metaPath = "cfg_mask/obj.data"  # Path to meta data
    network, class_names, class_colors = darknet.load_network(
        configPath,
        metaPath,
        weightPath,
        batch_size=1
    )

    # cap = cv2.VideoCapture(0)                                      # Uncomment to use Webcam
    cap = cv2.VideoCapture("mask_video/3578.MOV")  # Local Stored video detection - Set input video
    frame_width = int(cap.get(3))  # Returns the width and height of capture video
    frame_height = int(cap.get(4))
    # Set out for video writer
    out = cv2.VideoWriter(  # Set the Output path for video writer
        "mask_video/3578.avi", cv2.VideoWriter_fourcc(*"MJPG"), 60.0,
        (frame_width, frame_height))
    print("Starting the YOLO loop...")
    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(frame_width, frame_height, 3)  # Create image according darknet for compatibility of network

    while True:  # Load the input frame and write output frame.
        prev_time = time.time()
        ret, frame_read = cap.read()  # Capture frame and return true if frame present
        # For Assertion Failed Error in OpenCV
        if not ret:  # Check if frame present otherwise he break the while loop
            break

        frame_rgb = cv2.cvtColor(frame_read,
                                 cv2.COLOR_BGR2RGB)  # Convert frame into RGB from BGR and resize accordingly
        frame_resized = cv2.resize(frame_rgb,
                                   (frame_width, frame_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())  # Copy that frame bytes to darknet_image

        detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.6)
        if any(detections):
            det.append(detections)
        # Detection occurs at this line and return detections, for customize we can change the threshold.

        image = cvDrawBoxes(detections, frame_resized)  # Call the function cvDrawBoxes() for colored bounding box per class
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1 / (time.time() - prev_time))
        cv2.imshow('Demo', image)  # Display Image window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.waitKey(3)
        out.write(image)  # Write that frame into output video
    cap.release()  # For releasing cap and out.
    out.release()
    cv2.destroyAllWindows()
    print(":::Video Write Completed")

det = []
if __name__ == "__main__":
    YOLO()