"""
Python file for model definition
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Flatten, Dense, Lambda,
                                    Conv2D, MaxPool2D, Cropping2D)

# Function to generate model
def generate_model(input_shape = (160, 320, 3)):
    """
    Inputs
    ---
    input_shape: Shape of the input

    Outputs
    ---
    model: Keras model to train
    """
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape = input_shape))
    
    # Generate intermediate model
    model = generate_lenet_model(model)

    model.add(Dense(1))

    return model

# Function to generate LeNet model
def generate_lenet_model(model):
    """
    Inputs
    ---
    model: model with input defined

    Outputs
    ---
    model: model with LeNet architecture defined
    """
    # Convolution Block 1
    model.add(Conv2D(10, (5, 5), padding = 'valid'))    
    model.add(MaxPool2D((2, 2), strides = (2, 2)))      

    # Convolution Block 2
    model.add(Conv2D(32, (5, 5), padding = 'valid'))    
    model.add(MaxPool2D((2, 2), strides = (2, 2)))      

    # Flatten
    model.add(Flatten())

    # Dense Layers
    model.add(Dense(32))
    model.add(Dense(8))

    return model