from keras.models import load_model, Input
from keras.utils import to_categorical, plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from models.cnn_ensemble import Ensemble
from data import load_train, load_test, predictions_to_csv

model_names = ['cnn_ens1_96.02%', 'cnn_ens2_96.2%', 'cnn_ens3_96.45%']
models = []
for i, name in enumerate(model_names):
    model = load_model(f'./project3/trained_models/{name}.h5')
    model.name = f'ensemble_{i+1}'
    models.append(model)

model_input = Input(shape=(64, 64, 1))
model = Ensemble(models, model_input)
model_name = 'ensemble'

print(model.summary())
plot_model(model.model, to_file=f'./project3/figures/{model_name}_arch.png', show_shapes=True)

num_classes = 10
img_x, img_y = 64, 64

# Load data
train_images, train_labels = load_train()
x_test = load_test()

_, x_valid, _, y_valid = train_test_split(train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

# Reshape and normalize
x_valid = x_valid.reshape(x_valid.shape[0], img_x, img_y, 1)
x_valid = x_valid.astype('float32')
x_valid /= 255.
print('Validation dim: ', x_valid.shape)

x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
x_test = x_test.astype('float32')
x_test /= 255.
print('Test dim: ', x_test.shape)

y_valid = to_categorical(y_valid, num_classes)

# score = model.evaluate(x_valid, y_valid)

# print(f'Validation Loss: {score[0]}')
# print(f'Validation Accuracy: {score[1]}')

# Generate test labels
y_test = model.predict(x_test)
y_test = np.argmax(y_test, axis=1)
print(y_test.shape)
predictions_to_csv(y_test, f'{model_name}_pred.csv')

plt.show()
