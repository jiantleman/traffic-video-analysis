"""
References
- https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
- https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/README.md
YOLOv3 model download: https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
TinyYOLOv3 model download: https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo-tiny.h5
"""

import argparse
import cv2
from imageai.Detection import ObjectDetection
import numpy as np
import _thread

################################ Parameters ################################
FRAME_SIZE = (640,360)
HALF_FRAME_SIZE = (320,180)
OUTPUT_SIZE = (640,540)
# Motion detection
GAUSSIAN_KERNEL = (5,5)
GAUSSIAN_STDDEV = 2
BACK_SUB_THRESHOLD = 512
# Object detection
MIN_PERCENTAGE_PROB = 40
DETECTION_SPEED = "normal" # Options: normal/fast/faster/fastest/flash
############################################################################

def get_detection_area(detections):
    area = 0
    for object in detections:
        x1, y1, x2, y2 = object["box_points"]
        area += (x2-x1)*(y2-y1)
    return area

def detect_objects(detector, custom, motion_30frames, frame, results):
    returned_image, detections = detector.detectObjectsFromImage(
                                    input_type="array",
                                    input_image=frame, 
                                    output_type="array", 
                                    minimum_percentage_probability=MIN_PERCENTAGE_PROB,
                                    custom_objects = custom)
    results["returned_image"] = returned_image
    detection_area = get_detection_area(detections)
    results['congestion_5s'].pop(0)
    if detection_area != 0:
        print(sum(motion_30frames)/len(motion_30frames)/detection_area)
    if detection_area == 0 or sum(motion_30frames)/len(motion_30frames) > 0.1*detection_area:
        results['congestion_5s'].append(0)
    else:
        results['congestion_5s'].append(1)


def main():
    parser = argparse.ArgumentParser(description='This program combines video stabilization, motion detection, and object \
                                        detection to perform traffic video analysis.')
    parser.add_argument('--input', type=str, help='Path to input video.', default='video/combined.mp4')
    parser.add_argument('--output', type=str, help='Path to save output.', default='output/combined1.mp4')
    args = parser.parse_args()

    # Set up OpenCV background subtractor
    capture = cv2.VideoCapture(args.input)
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold = BACK_SUB_THRESHOLD, detectShadows=False)

    # Set up ImageAI object detector
    detector = ObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath( "model/yolo-tiny.h5")
    detector.loadModel(detection_speed=DETECTION_SPEED)
    custom = detector.CustomObjects(car=True, truck=True, motorcycle=True)
    results = {}
    motion_30frames = []

    # Save output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output,fourcc, 30, OUTPUT_SIZE)

    if not capture.isOpened():
        print('Unable to open: ' + args.input)
        exit(0)
    
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        
        # Preprocessing and background subtraction
        frame = cv2.resize(frame, FRAME_SIZE)
        pp_frame = cv2.GaussianBlur(src=frame, ksize=GAUSSIAN_KERNEL, sigmaX=GAUSSIAN_STDDEV)
        fgMask = backSub.apply(pp_frame)
        
        if capture.get(cv2.CAP_PROP_POS_FRAMES) == 1:
            results["returned_image"] = frame.copy()
            results["congestion_5s"] = [0]*5
        motion_30frames.append(np.count_nonzero(fgMask))

        # Run object detection in new thread every 30 frames
        if (capture.get(cv2.CAP_PROP_POS_FRAMES)-1) % 30 == 0:
            _thread.start_new_thread(detect_objects, (detector, custom, motion_30frames, frame, results))
            motion_30frames = []

        # Display information
        cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        cv2.putText(frame, "Congestion detected: "+str(sum(results['congestion_5s'])>=3), (10, fgMask.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (0,255,0), 2)
        
        # Write output frame to file
        fgMask = np.stack((fgMask,)*3, axis=-1)
        info = np.concatenate((cv2.resize(fgMask, HALF_FRAME_SIZE),cv2.resize(results["returned_image"], HALF_FRAME_SIZE)), axis=1)
        output_frame = np.concatenate((frame, info), axis=0)
        out.write(output_frame)

        # cv2.imshow('Frame', frame)
        # cv2.imshow('FG Mask', fgMask)
        # cv2.imshow('Object', results["returned_image"])

        # keyboard = cv2.waitKey(30)
        # if keyboard == ord("q"):
        #     break


if __name__ == "__main__":
    main()

