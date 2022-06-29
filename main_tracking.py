import cv2
import numpy as np
import argparse

FPS = 30

def count_objects(trackers):
    count = 0
    for item in trackers.getObjects():
        if not np.all(item == [0,0,0,0]):
            count += 1
    return count

def main():

    parser = argparse.ArgumentParser(description='This program combines motion detection and object \
                                        detection to perform traffic video analysis.')
    parser.add_argument('--input', type=str, help='Path to input video.', default='video/6-jam.mp4')
    parser.add_argument('--output', type=str, help='Path to save output.', default='output/combined.mp4')
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.input)
    previous_frame = None
    trackers = None
    num_start = 0
    num_tracked = 0
    ratio = 0
    min_size = 0

    frame_num = 0
    while True:
        _, frame = capture.read()
        if frame is None:
            break
        
        frame = cv2.resize(frame, (960,640))
        prepared_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5,5), sigmaX=0)

        if (previous_frame is None):
            # First frame; there is no previous one yet
            bbox = cv2.selectROI("Select",frame, False)
            cv2.destroyWindow('Select')
            previous_frame = prepared_frame
            min_size = 0.05*bbox[2]*bbox[3]
            continue

        frame_num += 1

        # calculate difference and update previous frame
        diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
        previous_frame = prepared_frame

        # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
        kernel = np.ones((5, 5))
        diff_frame = cv2.dilate(diff_frame, kernel, 1)

        # 5. Only take different areas that are different enough (>20 / 255)
        thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        
        num_start = 0
        for contour in contours:
            if cv2.contourArea(contour) < min_size:
                # too small: skip!
                continue
            num_start += 1
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
        
        if (frame_num-1) % (5 * FPS) == 0:
            if num_start != 0:
                ratio = num_tracked/(29*num_start)
            trackers = cv2.legacy.MultiTracker_create()
            num_start = 0
            num_tracked = 0
            # print(len(contours))
            for contour in contours:
                if cv2.contourArea(contour) < min_size:
                    # print(cv2.contourArea(contour), min_size)
                    # too small: skip!
                    continue
                num_start += 1
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(img=thresh_frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
                trackers.add(cv2.legacy.TrackerKCF_create(), frame, (x, y, w, h))
            if num_start == 0:
                frame_num -= 1
        elif (frame_num-1) % 5 == 0:
            (success, boxes) = trackers.update(frame)
            num_tracked += count_objects(trackers)
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    
        if ratio > 0.3:
            cv2.putText(frame, "Congestion detected", (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (0,0,255), 2)
        else:
            cv2.putText(frame, "No congestion", (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (0,255,0), 2)

        
        cv2.imshow('Frame', frame)
        keyboard = cv2.waitKey(30)
        if keyboard == ord("q"):
            break

if __name__ == "__main__":
    main()
