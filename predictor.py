import pickle


class ImageClassifier:
    def __init__(self):
        pass

    def deserialize(self):
        with open('model.pkl', 'rb') as handle:
            model = pickle.load(handle)

        return model

    def predict(self, input_image):
        model = self.deserialize()
        return model.predict(input_image)

