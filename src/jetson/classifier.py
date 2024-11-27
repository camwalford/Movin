import tensorflow as tf

class Classifier:

    def __init__(self, classifier_path):
        self.classifier = tf.keras.models.load_model(classifier_path)

    def predict(self, landmarks):
        # pred = self.classifier.predict(landmarks)
        return "jumping_jacks", 0.95
