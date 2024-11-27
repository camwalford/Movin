import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt


class DNN:

    def __init__(self):
        self.structure_model()

    def structure_model(self):
        self.model = keras.Sequential()

        self.model.add(keras.layers.Input(shape=(99,)))
        self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dense(256, activation='relu'))
        self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dense(256, activation='relu'))
        self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dense(64, activation='relu'))
        self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dense(256, activation='relu'))
        self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dense(64, activation='relu'))
        self.model.add(keras.layers.Dense(4, activation='softmax'))

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train_model(self, x_train, y_train, output_path, show_hist=False):
        hist = self.model.fit(
            x_train, y_train, epochs=6, validation_split=0.20, batch_size=128,
            callbacks=[EarlyStopping(monitor='loss', patience=2)]
        )
        plt.plot(hist.history['loss'], label='Train loss')
        plt.plot(hist.history['val_loss'], label='Validation loss')
        plt.legend()
        plt.show()

        self.model.save(output_path)

    def predict_model(self, x_test):
        pred = self.model.predict(x_test)
        top_5_classes = []
        top_5_probs = []

        for row in pred:
            top_5_indices = np.argsort(row)[::-1][:5]
            top_5_classes.append(top_5_indices)
            top_5_probs.append(row[top_5_indices])

        df = pd.DataFrame({
            'Class Probability Ranking': top_5_classes,
            'Associated Probabilities': top_5_probs
        })
        print(df)

