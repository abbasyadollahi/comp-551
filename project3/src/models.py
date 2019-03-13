import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Adadelta, Nadam
from sklearn.model_selection import train_test_split
from keras import backend as K
from matplotlib import pyplot as plt

from data import load_train

print(K.tensorflow_backend._get_available_gpus())

num_classes = 10
img_x, img_y = 64, 64

train_images, train_labels = load_train()
x_train, x_valid, y_train, y_valid = train_test_split(train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels)
print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_valid = x_valid.reshape(x_valid.shape[0], img_x, img_y, 1)
x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_train /= 255.
x_valid /= 255.
print(x_train.shape)
print(y_train.shape)

y_train = to_categorical(y_train, num_classes)
y_valid = to_categorical(y_valid, num_classes)
print(y_train.shape)
print(x_train[0].shape)

# Best: Ghetto vgg10 (half layer dimensions), Nadam optimizer
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_x, img_y, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# optimizer = Adadelta()
# optimizer = Adam()
optimizer = Nadam()

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

batch_size = 256  # best 64
# num_steps = x_train.shape[0] // batch_size
num_steps = 1000
epochs = 200

data_generator = ImageDataGenerator(
    rotation_range=12,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=1,
    zoom_range=0.12)

annealer = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5)
early_stop = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
checkpoint = ModelCheckpoint(
    filepath=f'./project3/models/checkpoints/cnn_steps={num_steps}_batch={batch_size}.h5',
    verbose=1,
    monitor='val_acc',
    save_best_only=True)

history = model.fit_generator(
    data_generator.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=num_steps,
    epochs=epochs,
    verbose=1,
    validation_data=(x_valid, y_valid),
    callbacks=[annealer, early_stop, checkpoint])

score = model.evaluate(x_valid, y_valid, verbose=0)

print(f'Validation Loss: {score[0]}')
print(f'Validation Accuracy: {score[1]}')

history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

if score[1] >= 0.96:
    model.save(f'./project3/models/cnn_{round(score[1]*100, 2)}%.h5')
else:
    print('Model did not exceed baseline validation accuracy, not saving.')

epcs = range(1, len(acc) + 1)

plt.plot(epcs, acc, 'bo', label='Training acc')
plt.plot(epcs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'./project3/figures/cnn_{round(score[1]*100, 2)}%.png')
plt.show()
