import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

class VGG():
    def __init__(self, input_shape, num_classes, optimizer):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
        print(self.model.summary())

    def train(self, x, y, x_valid, y_valid, batch_size=64, epochs=60, datagen=True, num_steps='auto', callbacks=None):
        if num_steps == 'auto':
            num_steps = x.shape[0] // batch_size
        if datagen:
            data_generator = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=1,
                zoom_range=0.1
            )
            self.history = self.model.fit_generator(
                data_generator.flow(x, y, batch_size=batch_size),
                steps_per_epoch=num_steps,
                epochs=epochs,
                verbose=1,
                validation_data=(x_valid, y_valid),
                callbacks=callbacks
            )
        else:
            self.history = self.model.fit(
                x, y,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_valid, y_valid),
                callbacks=callbacks
            )
        return self.history

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, verbose=0)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, filename):
        self.model.save(filename)
