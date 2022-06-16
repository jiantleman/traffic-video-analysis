"""
References
- https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
- https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/README.md

YOLOv3 model download: https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
TinyYOLOv3 model download: https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo-tiny.h5
"""

import cv2 as cv
import numpy as np
import argparse
from imageai.Detection import ObjectDetection
import os
import _thread

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

################################ Parameters ################################
# Motion detection
BACK_SUB_THRESHOLD = 256
FG_MASK_THRESHOLD = 0.0005
# Object detection
MIN_PERCENTAGE_PROB = 40
DETECTION_SPEED = "normal" # Options: normal/fast/faster/fastest/flash

# Subroutine for object detection executed on new thread
def detect_objects(detector, custom, frame, results):
    returned_image, detections = detector.detectObjectsFromImage(
                                            input_type="array",
                                            input_image=frame, 
                                            output_type="array", 
                                            minimum_percentage_probability=MIN_PERCENTAGE_PROB,
                                            custom_objects = custom)
    results["returned_image"] = returned_image
    results["detections"] = detections

def main():
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='traffic.mp4')
    args = parser.parse_args()

    # Set up ImageAI object detector
    detector = ObjectDetection()
    # detector.setModelTypeAsYOLOv3()
    # detector.setModelPath( "yolo.h5")
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath( "yolo-tiny.h5")
    detector.loadModel(detection_speed=DETECTION_SPEED)
    custom = detector.CustomObjects(person=True, car=True, truck=True, motorcycle=True)
    results = {}

    # Set up OpenCV background subtractor
    backSub = cv.createBackgroundSubtractorMOG2(varThreshold = BACK_SUB_THRESHOLD, detectShadows=False)
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
    if not capture.isOpened():
        print('Unable to open: ' + args.input)
        exit(0)

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        # Preprocess frame 
        frame = cv.resize(frame, (480,270))
        processed_frame = cv.GaussianBlur(src=frame, ksize=(5,5), sigmaX=0)
        fgMask = backSub.apply(processed_frame)
        
        # Threshold ratio of difference in entire image
        diffRatio = np.count_nonzero(fgMask)/fgMask.size

        # Display information
        cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        cv.rectangle(frame, (10, 22), (200,42), (255,255,255), -1)
        cv.putText(frame, "Motion detected: "+ str(diffRatio>FG_MASK_THRESHOLD), (15, 37),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        # Run object detection in new thread every 2 seconds
        if capture.get(cv.CAP_PROP_POS_FRAMES) == 1:
            returned_image, detections = detector.detectObjectsFromImage(
                                            input_type="array",
                                            input_image=frame, 
                                            output_type="array", 
                                            minimum_percentage_probability=MIN_PERCENTAGE_PROB,
                                            custom_objects = custom)
            results["returned_image"] = returned_image
            results["detections"] = detections
        elif capture.get(cv.CAP_PROP_POS_FRAMES)%60 == 0:
            _thread.start_new_thread(detect_objects, (detector, custom, frame, results,))

        # Display frames 
        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', fgMask)
        cv.imshow('Detection',results["returned_image"])
        
        keyboard = cv.waitKey(30)
        if keyboard == ord("q"):
            break

if __name__ == "__main__":
    main()