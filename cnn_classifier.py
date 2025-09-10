from keras import Sequential
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Activation
from keras.utils import image_dataset_from_directory
from keras.callbacks import TensorBoard
from datetime import datetime

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class CNNModel:
    def __init__(self, shape=(256, 256, 3), num_conv_layers=1, hidden_layer=False, normalize=True):
        self.model = Sequential()
        self.model.add(Input(shape=shape))
        self.num_conv_layers = num_conv_layers
        self.hidden_layer = hidden_layer
        self.normalize = normalize

    def _add_conv_layer(self, filters, kernel_size=(3, 3), activation='relu', normalize=True, pool_size=(2, 2)):
        # Slides filters across the height and width of an image to produce feature maps.
        self.model.add(Conv2D(filters, kernel_size))
        if normalize:
            self.model.add(BatchNormalization())
        self.model.add(Activation(activation))

        # Takes max value in each patch of each feature map to highlight the most important features
        self.model.add(MaxPooling2D(pool_size=pool_size))
    
    def build(self):
        for i in range(self.num_conv_layers):
            self._add_conv_layer(filters=(2**i)*32, normalize=self.normalize)
        
        self.model.add(Flatten())

        if self.hidden_layer:
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dropout(0.5))
        
        self.model.add(Dense(4, activation='softmax')) # separate into the 4 classes

        return self.model
    
def get_dataset(dataset_path):
    ds = image_dataset_from_directory(
        directory=dataset_path,
        labels='inferred',
        label_mode='int',
        batch_size=64,
        image_size=image_size,
    )
    return ds
    
if __name__ == "__main__":
    train_path =  "scans/Training"
    test_path =  "scans/Testing"
    image_size = (256, 256)  
    input_shape = image_size + (3,)

    print('loading traing images')
    train_ds = get_dataset(train_path)
    print('processing testing images')
    test_ds = get_dataset(test_path)

    log_dir = "logs/fit/" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    cnn = CNNModel(num_conv_layers=1, hidden_layer=False, normalize=False)
    model = cnn.build()
    model.summary()

    # compile and train
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(train_ds, epochs=20, validation_data=test_ds, callbacks=[tensorboard_callback])

    # plot
    # plt.title('Training Accuracy vs Validation Accuracy')

    # print(history.history['accuracy'])
    # print(history.history['val_accuracy'])

    # plt.plot(history.history['accuracy'], color='red',label='Train')
    # plt.plot(history.history['val_accuracy'], color='blue',label='Validation')

    # plt.legend()
    # plt.show()