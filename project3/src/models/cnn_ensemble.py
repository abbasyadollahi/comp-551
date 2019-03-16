from keras.models import Model, Input
from keras.layers import average

class Ensemble():
    def __init__(self, models, model_input):
        models = [model(model_input) for model in models]
        y = average(models)
        self.model = Model(inputs=model_input, outputs=y, name='ensemble')

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, verbose=0)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, filename):
        self.model.save(filename)

    def summary(self):
        return self.model.summary()

    def predict_classes(self, x):
        return self.model.predict_classes(x)
