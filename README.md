# Traffic Video Analysis

This program is a proof-of-concept for the use of unmanned aerial vehicle (UAV) footage with computer vision to automate two separate tasks:
- Identification of traffic congestions
- Identification of intruders

The program combines the following techniques and libraries:
- Video stabilization ([VidGear](https://github.com/abhiTronix/vidgear#videogear)): Stabilizes the UAV footage across frames to enable motion detection using background subtraction.
- Background subtraction ([OpenCV](https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html])): Detects motion by finding the difference between a particular frame and a fixed background.
- Object detection ([ImageAI](https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/README.md)): Uses the Tiny-YOLOv3 deep learning model to detect specific objects (e.g. cars, person) within a frame.

How the above techniques could be used to accomplished the desired tasks depends on the specific use cases and context in which the UAV is deployed. As an example, an alert for a traffic congestion could be triggered when cars are detected but no motion is detected for a certain number of frames, while an alert for an intruder could be triggered when a person and motion is detected. 

**Limitations**
This program, as a simple proof-of-concept, has the following limitations that must be addressed before it can be deployed:
- _Significant computational resources required_: Object detection uses a computationally-intensive algorithm that requires the use of a GPU for fast completion. Without adequate computational resources, there is a tradeoff between speed, recall (fraction of objects that are correctly identified), and frequency of updates.
- _Sensitivity to high-frequency jitter_: The use of background subtraction for motion detection relies on the camera having a fixed pose such that the background remains constant, which is not possible with a UAV. While low-frequency jitter can be mitigated using video stabilization and other pre-processing techniques, high-frequency jitter could result in motion being incorrectly detected due to changes to the background. 
- _Sensitivity to operating conditions_: Video stabilization and motion detection involves various parameters (e.g. threshold of difference in background that constitutes motion, amount of stabilization) that might need to be finetuned for the program to work in different operating conditions such as amount of camera jitter, size of objects in the frame, and amount of background motion (e.g. rustling of leaves, swaying of trees).