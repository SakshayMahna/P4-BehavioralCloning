"""
Main python file
"""
from model import generate_model
from data import load_dataset

# Load the data
X, y = load_dataset("data/driving_log.csv", "data/IMG/")

# Get the model
model = generate_model()
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X, y, validation_split = 0.2, shuffle = True, epochs = 4)

model.save('model.h5')