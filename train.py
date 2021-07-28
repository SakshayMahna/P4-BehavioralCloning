"""
Main python file
"""
from math import ceil
from model import generate_model
from data import load_dataset, load_generator

# Load the data
# X, y = load_dataset("data/driving_log.csv", "data/IMG/")

# Load the data generator
BATCH_SIZE = 5
train_generator, validation_generator, train_length, validation_length = load_generator("data/driving_log.csv", "data/IMG/", BATCH_SIZE)

# Get the model
model = generate_model()
model.compile(loss = 'mse', optimizer = 'adam')
# model.fit(X, y, validation_split = 0.2, shuffle = True, epochs = 4)
model.fit(
    train_generator, steps_per_epoch = ceil(train_length / BATCH_SIZE),
    validation_data = validation_generator, validation_steps = ceil(validation_length / BATCH_SIZE),
    epochs = 5, verbose = 1
)

model.save('model.h5')