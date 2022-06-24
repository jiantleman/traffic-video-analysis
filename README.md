# Traffic Video Analysis

## How to use

See [`output/combined.mp4`](output/combined.mp4) for a sample output of the program. 

To run the program on your own video file:
1. Install [Python 3.7.6](https://www.python.org/downloads/release/python-376/).
2. Open the Terminal on your computer and run the following:
    ```
    pip install tensorflow==2.4.0
    pip install keras==2.4.3 numpy==1.19.3 pillow==7.0.0 scipy==1.4.1 h5py==2.10.0 matplotlib==3.3.2 opencv-python keras-resnet==0.2.0
    pip install imageai --upgrade
    ```
3. Run the following command in your terminal:
    ```
    python main.py --input <path to video file> --output <path to save output>
    ```


## Program Details

This program is a proof-of-concept for the use of unmanned aerial vehicle (UAV) footage with computer vision to identify traffic congestion.

The program combines the following techniques:
- Background subtraction ([OpenCV](https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html)): Detects motion by assuming that the background remains constant and finding the difference between consecutive frames.
- Objection detection ([ImageAI](https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/README.md)): Uses the Tiny-YOLOv3 deep learning model to detect vehicles within a frame.

We identify a potential traffic congestion when the amount of motion is significantly lesser than the area of the detected vehicles across multiple seconds, indicating slow-moving vehicles.

**Limitations**

This program, as a simple proof-of-concept, has the following limitations that must be addressed before it can be deployed:
- _Significant computational resources required_: Object detection using deep learning is a computationally-intensive task and requires the use of a GPU for fast completion. Without adequate computational resources, there is a tradeoff between speed, recall (fraction of vehicles that are correctly identified), and frequency of updates. Currently, processing one 640x360 frame per second of a 1 min 30 second video on Google Colaboratory takes approximately 11 minutes with a GPU and 15 minutes without a GPU, which is way too slow for real-time applications. The poor recall, i.e. failure to accurately detect all vehicles present, could also lead to a traffic congestion being undetected, for instance in the 4th sequence in "output.mp4". 
    - The use of a more powerful GPU could enable faster processing with a better deep learning model and larger frame sizes, which is necessary for accurate real-time congestion detection.
- _Sensitivity to jitter and background motion_: The use of background subtraction for motion detection assumes that the background is constant, which requires that the camera has a fixed pose. While most commercial drones today have stabilization, there might nonetheless be some camera jitter which could result in motion being incorrectly detected due to changes to the background. Furthermore, there might be background motion (e.g. of plants, people, etc.) that might be incorrectly detected as vehicular motion. 
    - Low-frequency jitter could be mitigated using additional preprocessing techniques like video stabilization (e.g. using [VidGear](https://github.com/abhiTronix/vidgear#videogear)).
    - Currently, various techniques like Gaussian blurring, finding the mean amount of motion across multiple frames, and checking for potential congestion across multiple seconds serve to reduce the impact of camera jitter and background motion on the program's accuracy. The parameters used in these techniques could be finetuned to further mitigate the impact of camera jitter and background motion. 

This program presents one possible, rudimentary approach to congestion detection using traffic videos. Other more sophisticated image processing techniques that are not based on background subtraction and deep learning (e.g. [image texture analysis](https://www.sciencedirect.com/science/article/pii/S1877705816002630)) could also be explored to address the limitations of this program. 
