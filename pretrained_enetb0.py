from keras.applications import EfficientNetB0
from keras import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D
from keras.utils import image_dataset_from_directory
from keras.callbacks import TensorBoard
from datetime import datetime


class PretrainedEfficientNetB0:
    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape
        self.model = self._build()

    def _build(self, activation='relu'):
        # EfficientNetB0 already handles the input, no need for separate Input layer
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=self.input_shape)
        base_model.trainable = False

        # Create Sequential model without explicit Input layer
        model = Sequential([
            base_model,
            Conv2D(32, (3,3), activation='relu'),
            GlobalAveragePooling2D(),
            Dense(128, activation=activation),
            Dropout(0.5),
            Dense(4, activation='softmax')
        ])

        return model
    
    def get_model(self):
        return self.model


def get_dataset(dataset_path):
        ds = image_dataset_from_directory(
            directory=dataset_path,
            labels='inferred',
            label_mode='int',
            batch_size=64,
            image_size=image_size,  # Note: This should match your input_shape
        )
        return ds


if __name__ == "__main__":
    train_path = "scans/Training"
    test_path = "scans/Testing"
    image_size = (224, 224)  
    input_shape = image_size + (3,)  

    print('loading training images')
    train_ds = get_dataset(train_path)
    print('processing testing images')
    test_ds = get_dataset(test_path)

    log_dir = "logs/fit/" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    enetb0 = PretrainedEfficientNetB0(input_shape=input_shape)
    model = enetb0.get_model()
    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(train_ds, epochs=10, validation_data=test_ds, callbacks=[tensorboard_callback])