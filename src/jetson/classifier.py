class Classifier:

    def __init__(self, classifier):
        print("Classifier path:", classifier)
        self.classifier = classifier

    def predict(self, landmarks):
        print("Classifier received landmarks:", landmarks)
        print("Predicting on landmarks.")
        return "jumping_jacks"
