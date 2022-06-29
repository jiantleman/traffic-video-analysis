"""
Implementation of Li and Dai, "Real-time Congestion Detection Based on Image Texture Analysis",
https://doi.org/10.1016/j.proeng.2016.01.250
"""

import cv2
import numpy as np
import argparse
from skimage.feature import graycomatrix, graycoprops
from skimage.exposure import rescale_intensity
from skimage.measure import shannon_entropy

S_THRESHOLD = 6.5

def main():

    parser = argparse.ArgumentParser(description='This program combines motion detection and object \
                                        detection to perform traffic video analysis.')
    parser.add_argument('--input', type=str, help='Path to input video.', default='video/demo.mp4')
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.input)
    bbox = None
    hist = [0]*30

    if not capture.isOpened():
        print('Unable to open: ' + args.input)
        exit(0)

    while True:
        _, frame = capture.read()
        if frame is None:
            break
        
        frame = cv2.resize(frame, (960,640))

        # Allow user to select region of interest
        if bbox == None:
            cv2.rectangle(frame, (330, 0), (960, 30), color=(255,255,255), thickness=-1)
            cv2.putText(frame, "Select region of interest and press Enter. Press \"c\" to cancel.", (frame.shape[0]-300, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (0,0,0), 1)
            bbox = cv2.selectROI("Select Region of Interest",frame, False)
            if bbox == (0,0,0,0):
                exit(0)
            cv2.destroyWindow("Select Region of Interest")

        # Preprocessing
        # 1. Crop to region of interest
        # 2. Convert to grayscale
        # 3. Reduce pixel intensity from 256-level to 32-level
        processed = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        processed = rescale_intensity(processed, in_range=(0,255), out_range=(0, 31)).astype(np.int8)

        # Calculate GLCM, entropy S_p, and energy feature S'_g
        glcm = graycomatrix(processed, distances=[1], 
                            angles=[0,np.pi/4, np.pi/2, np.pi*3/4], levels=32)
        energy = -np.log(graycoprops(glcm,prop='ASM')[0])
        entropy = np.zeros(4)
        for i in range(4):
            entropy[i] = shannon_entropy(glcm[:,:,:,i])
        
        S = sum(energy)/4 + sum(entropy)/4

        # Record S values in last 30 frames       
        hist.pop(0)
        hist.append(S)
        
        # Congestion detected if mean S is greater than threshold
        if sum(hist)/30 >= S_THRESHOLD:
            cv2.putText(frame, "Congestion detected", (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (0,0,255), 2)
        else:
            cv2.putText(frame, "No congestion detected", (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (0,255,0), 2)

        cv2.rectangle(frame, (330, 0), (960, 30), color=(255,255,255), thickness=-1)
        cv2.putText(frame, "Press \"s\" to select a new region of interest. Press \"c\" to exit.", (frame.shape[0]-300, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (0,0,0), 1)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color=(255,0,0), thickness=2)

        # Display frame
        cv2.imshow('Video', frame)
        keyboard = cv2.waitKey(30)
        if keyboard == ord("c"):
            break
        if keyboard == ord("s"):
            cv2.destroyWindow("Video")
            bbox = None

if __name__ == "__main__":
    main()