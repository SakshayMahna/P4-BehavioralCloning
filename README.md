# Project: Behavioral Cloning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, we use deep neural networks and convolutional neural networks to clone driving behavior. The model is trained, validated and tested using Keras. The model outputs a steering angle to an autonomous vehicle.

The autonomous vehicle is provided as a simulator. Image data and steering angles are used to train a neural network and drive the simulation car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

The project contains the following files:
* [model.py](model.py) (script used to create the model)
* [train.py](train.py) (script used to train the model)
* [drive.py](drive.py) (script to drive the car - feel free to modify this file)
* [model.h5](model.h5) (a trained Keras model)
* [writeup file](writeup.md) 
* [video.mp4](video.mp4) (a video recording of the vehicle driving autonomously around the track for one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Getting Started
---

The project has been developed on a Linux machine with Python 3.6 with a Tesla K10 GPU. The system was provided by Udacity for this particular project.

The following steps have been provided to run the program on a Windows 10 machine with Python 3.8.8.

## Prerequisites
Install dependencies

```bash
pip install -r requirements.txt
```

Dataset
---

The dataset has been generated by using the Car Simulator provided [here](https://github.com/udacity/self-driving-car-sim). The training can be done directly by running the simulator.

The compiled and collected dataset is available [here](data).

Training
---

The training can be done by using the command

```bash
python train.py
```

Testing
---

The trained model can be used to run the autonomous car on simulator. Start the simulator in autonomous mode and use the following command to start the code.

```bash
python drive.py model.h5
```

Output
---

[video.mp4](video.mp4)

[Youtube](https://youtu.be/7qR-o3wB3r8) video for the same.
