import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ref process.py: 0.839506 using {'batch_size': 16, 'epochs': 20, 'neurons': 64, 'optimizer': 'adam'}
upstairs_data = pd.read_csv("f2.csv")
downstairs_data = pd.read_csv("f1.csv")

# create labels
data = pd.concat([upstairs_data, downstairs_data], axis=0)
labels = np.concatenate([np.zeros(len(upstairs_data)), np.ones(len(downstairs_data))])

# Normalize
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

# Create a feedforward neural network model with the best optimization parameters
model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(data.shape[1],)))  # Neurons updated to 64
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fnn
history = model.fit(x_train, y_train, epochs=20, batch_size=16, validation_split=0.2)  # Best batch size and epochs used
model.evaluate(x_test, y_test)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower left')

plt.show()
