# Traffic Video Analysis

## How to Use

To run the program on your own video file:
1. Install [Git](https://git-scm.com/downloads) and [Python](https://www.python.org/downloads). 
2. Open the terminal on your computer from the folder you would like to save the program in. Download this repository by running the following:
    ```
    git clone https://github.com/jiantleman/traffic-video-analysis
    ```
3. Run the following commands in your terminal:
    ```
    pip install numpy
    pip install opencv-python
    pip install scikit-image
    ```
4. Run the following command in your terminal:
    ```
    python main.py --input <path to video file>
    ```
   You can also save the video output by running the following command:
    ```
    python main.py --input <path to video file> --output <path to save output>
    ```
4. Select the region of interest and press Enter to start the analysis. Press 's' to select a new region of interest and 'c' to exit.

## Program Details
 
This program is an implementation of the paper [*Real-time Congestion Detection Based on Image Texture Analysis*](https://doi.org/10.1016/j.proeng.2016.01.250) by Li Wei and Dai Hong-ying in Python using OpenCV and scikit-image. It estimates the vehicle density in a region of interest in a video frame by looking at its texture, which is a measure of how ordered and regular objects in an image are. 

The camera position and region of interest should be carefully selected to ensure the accuracy of the program. The region of interest should be a segment of the road that would contain multiple vehicles if there is a congestion and should not contain other reccuring patterns, e.g. road markings, trees, traffic barrier. 

See [`demo.mp4`](demo.mp4) for a demonstration of the program and examples of how the region of interest could be selected. 
