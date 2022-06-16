"""
References
- https://github.com/abhiTronix/vidgear#videogear
- https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
- https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/README.md

YOLOv3 model download: https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
TinyYOLOv3 model download: https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo-tiny.h5
"""

import os
import argparse
import cv2
import numpy as np
from imageai.Detection import ObjectDetection
from vidgear.gears import VideoGear

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

################################ Parameters ################################
FRAME_SIZE = (480,270)
# Video stabilization
options = {
    'SMOOTHING_RADIUS': 250,
    'BORDER_SIZE': 64,
    'CROP_N_ZOOM': True
}
# Motion detection
GAUSSIAN_KERNEL = (11,11)
GAUSSIAN_STDDEV = 10
BACK_SUB_THRESHOLD = 512
FG_MASK_THRESHOLD = 0.0005
# Object detection
MIN_PERCENTAGE_PROB = 40
DETECTION_SPEED = "normal" # Options: normal/fast/faster/fastest/flash
############################################################################

def main():
    parser = argparse.ArgumentParser(description='This program combines video stabilization, motion detection, and object \
                                        detection to perform traffic video analysis.')
    parser.add_argument('--input', type=str, help='Path to input video.', default='video/10.avi')
    parser.add_argument('--output', type=str, help='Path to save output.', default='output/10.avi')
    parser.add_argument('--display', type=bool, help='Display analysis output.', default=False)
    args = parser.parse_args()

    # Open video stream with stabilization enabled
    stream_stab = VideoGear(source=args.input, stabilize=True, **options).start()
    stream_org = VideoGear(source=args.input).start()
    frame_num = 0

    # Set up ImageAI object detector
    detector = ObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath( "model/yolo-tiny.h5")
    detector.loadModel(detection_speed=DETECTION_SPEED)
    custom = detector.CustomObjects(person=True, car=True, truck=True, motorcycle=True)

    # Set up OpenCV background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold = BACK_SUB_THRESHOLD, detectShadows=False)

    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc('M','J','P','G'),3,(960,540))

    while True:
        frame_num += 1
        frame_processed = stream_stab.read()
        if frame_processed is None:
            break
        frame_org = stream_org.read()
        if (frame_num-1)%10 != 0:
            continue
        # Resize frame 
        frame_stab = cv2.resize(frame_processed, FRAME_SIZE)
        frame_org = cv2.resize(frame_org, FRAME_SIZE)
        
        # Motion detection: 
        # 1. Apply Gaussian blur
        # 2. Perform background subtraction 
        # 3. Calculate ratio of difference in entire image
        processed_frame = cv2.GaussianBlur(src=frame_stab, ksize=GAUSSIAN_KERNEL, sigmaX=GAUSSIAN_STDDEV)
        fgMask = backSub.apply(processed_frame)
        diffRatio = np.count_nonzero(fgMask)/fgMask.size        

        # Run object detection in new thread every 2 seconds
        returned_image, detections = detector.detectObjectsFromImage(
                                        input_type="array",
                                        input_image=frame_org, 
                                        output_type="array", 
                                        minimum_percentage_probability=MIN_PERCENTAGE_PROB,
                                        custom_objects = custom)

        
        # Display frame information
        fgMask = np.stack((fgMask,)*3, axis=-1)
        cv2.rectangle(frame_org, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(frame_org, str(frame_num), (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        cv2.putText(frame_org, "Original", (10, frame_org.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (0,255,0), 2)
        cv2.putText(frame_stab, "Stabilized", (10, frame_stab.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (0,255,0), 2)
        cv2.putText(fgMask, "Motion detected", (10, fgMask.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (0,255,0), 2)
        cv2.putText(returned_image, "Object detected", (10, returned_image.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (0,255,0), 2)
            
        # Concatenate and ahow output window
        top_row = np.concatenate((frame_org, frame_stab), axis=1)
        bottom_row = np.concatenate((fgMask, returned_image), axis=1)
        output_frame = np.concatenate((top_row, bottom_row), axis=0)
        if args.display:
            cv2.imshow("Traffic video analysis", output_frame)
        out.write(output_frame)
            
        keyboard = cv2.waitKey(0)
        if keyboard == ord("q"):
            break
    

    cv2.destroyAllWindows()
    out.release()
    stream_stab.stop()
    stream_org.stop()
        

if __name__ == "__main__":
    main()