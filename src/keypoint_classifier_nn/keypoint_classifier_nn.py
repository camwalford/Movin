import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import pandas as pd
import os

path = './classifier_input/berkeley/'

full_data = pd.DataFrame()

# grab full data
for entry in sorted(os.listdir(path)):
    print(entry)
    if os.path.isfile(os.path.join(path, entry)):
        if entry.endswith('.txt'):
            data = pd.read_csv(path + entry, sep=' ', header=None)
            data.drop([129, 130], inplace=True, axis=1)
            data['classs'] = entry[-10:-8]
            full_data = pd.concat([full_data, data], ignore_index=True)

# Split into data and label
x = full_data.drop(["classs"], axis=1)
y = full_data.classs.values

# Label t-pose as class type 12
y = pd.DataFrame(y)
y.iloc[:, 0] = y.iloc[:, 0].str.replace('t', '1')
y.iloc[:, 0] = y.iloc[:, 0].str.replace('-', '2')
y.astype('int32')

# Split into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                    shuffle=True)
print('Shape of train data is : ', x_train.shape)
print('Shape of label data is : ', y_train.shape)

# Convert labels into categorical features
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define model architecture
early_stop = EarlyStopping(monitor='loss', patience=2)

model = keras.Sequential()

model.add(keras.layers.Input(shape=(129,)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(13, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model and visualize training
hist = model.fit(x_train, y_train, epochs=6, validation_split=0.20,
                 batch_size=128, callbacks=[early_stop])
plt.plot(hist.history['loss'], label='Train loss')
plt.plot(hist.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

# Save the model
model.save('nn_berkeley.keras')

# Test the model and display best predictions
pred = model.predict(x_test)
top_5_classes = []
top_5_probs = []

for row in pred:
    top_5_indices = np.argsort(row)[::-1][:5]
    top_5_classes.append(top_5_indices)
    top_5_probs.append(row[top_5_indices])

# Create a DataFrame to display the results in a table format
df = pd.DataFrame({
    'Top 5 Classes': top_5_classes,
    'Top 5 Probabilities': top_5_probs
})

print(df)
