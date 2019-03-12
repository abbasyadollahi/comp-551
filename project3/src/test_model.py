from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from data import load_train, load_test, predictions_to_csv
import numpy as np

model = load_model('./project3/cnn_89.8%.h5')

train_images, train_labels = load_train()
x_test = load_test()

_, x_valid, _, y_valid = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

x_valid = x_valid.reshape(x_valid.shape[0], 64, 64, 1)
x_valid = x_valid.astype('float32')
x_valid /= 255

x_test = x_test.reshape(x_test.shape[0], 64, 64, 1)
x_test = x_test.astype('float32')
x_test /= 255
print(x_test.shape)

y_valid = to_categorical(y_valid, 10)
score = model.evaluate(x_valid, y_valid)

print(f'Validation Loss: {score[0]}')
print(f'Validation Accuracy: {score[1]}')

y_test = model.predict(x_test)
y_test = np.argmax(y_test, axis=1)
print(y_test.shape)
predictions_to_csv(y_test, 'cnn_1.csv')
