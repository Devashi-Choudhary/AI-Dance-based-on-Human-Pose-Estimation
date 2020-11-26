# AI-Dance-based-on-Human-Pose-Estimation
A Human Pose Skeleton represents the orientation of a person in a graphical format. Essentially, it is a set of coordinates that can be connected to describe the pose of the person. Each co-ordinate in the skeleton is known as a part (or a joint, or a keypoint). A valid connection between two parts is known as a pair (or a limb).

# Dependencies
 - [Keras](https://pypi.org/project/Keras/)
 - [Librosa](https://pypi.org/project/librosa/)
 - [Moviepy](https://pypi.org/project/moviepy/)
 - [OpenCV](https://pypi.org/project/opencv-python/)
 - [Pytube3](https://pypi.org/project/pytube/)
 - [Tensorflow](https://www.tensorflow.org/install/pip)
 
 # Flow of Human Pose Estimation
 
- Extraction of pose coordinates from dance videos using openpose human pose estimation.
- Training LSTM network on extracted coordinates using video as input and coordinates as output. 
- Display output videos by joining predicted coordinates to generate dancing human stick figures.

# How to Execute Code 

1. You will first have to download the repository and then extract the contents into a folder.
2. Make sure you have the correct version of Python installed on your machine. This code runs on Python 3.6 above.
3. Now, install the required libraries. 
4. Now go to src folder and run extract_data.py to download videos and audios to data folder. You can add youtube videos links to "video_links.txt" file for downloading. 
5. Download pretrained weights for pose estimation from [here](https://drive.google.com/file/d/1WYWwZR_mtUSfRCR-Rwi0mDGNlL_Uvbei/view?usp=sharing). Download pose_iter_440000.caffemodel and save it in "models" folder.
6. Run demo.py to train LSTM and display predicted dance video.
> `python main.py --video data/video.mp4 --audio data/audio.wav --background data/bg1.jpg `

# Results

1. **Pose estimation using openpose on image**

![image](https://github.com/Devashi-Choudhary/AI-Dance-based-on-Human-Pose-Estimation/blob/main/outputs/1.JPG)

2. **Pose estimation using openpose on video** 

[![](http://i3.ytimg.com/vi/JrekUhYvGt4/hqdefault.jpg)](https://youtu.be/JrekUhYvGt4)

**NOte** : For more information about implementation details, please go through [AI Dance based on Human Pose Estimation](https://devashi-choudhary.medium.com/ai-dance-based-on-human-pose-estimation-738ac2ff6d1f)

# References

1. [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 
2. [Moviepy](https://zulko.github.io/moviepy/)
3. [Pytube](https://python-pytube.readthedocs.io/en/latest/)
4. [Dancing AI](https://github.com/keshavoct98/DANCING-AI)
