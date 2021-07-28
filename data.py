"""
Python module to load the data for training
"""
import cv2
import csv
import numpy as np
import concurrent.futures
from tqdm import tqdm

# Function to load dataset
def load_dataset(csv_path, relative_path):
    """
    Inputs
    ---
    csv_path: path to training data csv
    relative_path: relative path to training data

    Outputs
    ---
    X: Training data numpy array
    y: Training labels numpy array
    """
    lines = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        print("Loading CSV File ...")
        for line in tqdm(reader):
            lines.append(line)
    
    images = []; measurements = []
    print("Loading Data ...")


    for line in tqdm(lines):
        # Center Image
        image, measurement = _load_image(line, 0, relative_path)
        images.append(image)
        measurements.append(measurement)

        image_flipped = np.fliplr(image)
        images.append(image_flipped)

        measurement_flipped = -1 * measurement
        measurements.append(measurement_flipped)

        # Left Image
        image, measurement = _load_image(line, 1, relative_path)
        images.append(image)
        measurements.append(measurement)

        image_flipped = np.fliplr(image)
        images.append(image_flipped)

        measurement_flipped = -1 * measurement
        measurements.append(measurement_flipped)

        # Right Image
        image, measurement = _load_image(line, 2, relative_path)
        images.append(image)
        measurements.append(measurement)

        image_flipped = np.fliplr(image)
        images.append(image_flipped)

        measurement_flipped = -1 * measurement
        measurements.append(measurement_flipped)

    X = np.array(images)
    y = np.array(measurements)

    return X, y

# Private function to load image
def _load_image(line, index, relative_path):
    """
    Inputs
    ---
    line: csv line to read data from
    index: decides left, right or center
    relative_path: relative path of the data

    Outputs
    ---
    image: output image
    measurement: output measurement
    """
    source_path = line[index]
    filename = source_path.split('\\')[-1]
    current_path = relative_path + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if index == 1:
        # Left Image
        correction = 0.2
    elif index == 2:
        # Right Image
        correction = -0.2
    else:
        # Center Image
        correction = 0

    measurement = float(line[3]) + correction

    return image, measurement