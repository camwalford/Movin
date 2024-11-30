import numpy as np
import tensorflow as tf

class Classifier:

    def __init__(self, classifier_path, label_encoder_path=None):
        if label_encoder_path:
            self.class_names = np.load(label_encoder_path, allow_pickle=True).tolist()
        self.classifier = tf.keras.models.load_model(classifier_path)

    def predict(self, landmarks):
        pred = self.classifier.predict(landmarks)
        if self.class_names:
            pred_class = self.class_names[np.argmax(pred)]
            probability = np.max(pred)
            return pred_class, probability
        return "jumping_jacks", 0.95
