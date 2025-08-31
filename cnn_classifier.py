import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Activation
from keras.utils import image_dataset_from_directory
from keras.callbacks import TensorBoard
import datetime

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


train_path =  "scans/Training"
test_path =  "scans/Testing"

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

def get_dataset(dataset_path):
    ds = image_dataset_from_directory(
        directory=dataset_path,
        labels='inferred',
        label_mode='int',
        batch_size=64,
        image_size=(256, 256),
    )
    return ds


print('loading traing images')
train_ds = get_dataset(train_path)
print('processing testing images')
test_ds = get_dataset(test_path)



# barebones CNN
model = Sequential(
    [
        Input(shape=(256, 256, 3)),

        # Slides filters across the height and width of an image to produce feature maps.
        Conv2D(32, (3, 3)),  
        BatchNormalization(),
        Activation('relu'),

        # Takes max value in each patch of each feature map to highlight the most important features
        MaxPooling2D((2, 2)),    

        Flatten(),                
        Dense(4, activation='softmax') # seperate into 4 classes
    ]
)

# # barebones CNN with dropout and another dense layer
# model = Sequential(
#     [
#         Input(shape=(256, 256, 3)),

#         # Slides filters across the height and width of an image to produce feature maps.
#         Conv2D(32, (3, 3)),  
#         BatchNormalization(),
#         Activation('relu'),

#         # Takes max value in each patch of each feature map to highlight the most important features
#         MaxPooling2D((2, 2)),    

#         Flatten(),                
#         Dense(4, activation='softmax') # seperate into 4 classes
#     ]
# )

# # CNN with more layers 
# model = Sequential(
#     [
#         Input(shape=(256, 256, 3)),

#         # Slides filters across the height and width of an image to produce feature maps.
#         Conv2D(32, (3, 3)),  
#         BatchNormalization(),
#         Activation('relu'),

#         # Takes max value in each patch of each feature map to highlight the most important features
#         MaxPooling2D((2, 2)),    

#         # Slides filters across the height and width of an image to produce feature maps.
#         Conv2D(64, (3, 3)),  
#         BatchNormalization(),
#         Activation('relu'),

#         # Takes max value in each patch of each feature map to highlight the most important features
#         MaxPooling2D((2, 2)),   

#         Flatten(),                
#         Dense(4, activation='softmax') # seperate into 4 classes
#     ]
# )

# # CNN with more layers and dense layer
# model = Sequential(
#     [
#         Input(shape=(256, 256, 3)),

#         # Slides filters across the height and width of an image to produce feature maps.
#         Conv2D(32, (3, 3)),  
#         BatchNormalization(),
#         Activation('relu'),

#         # Takes max value in each patch of each feature map to highlight the most important features
#         MaxPooling2D((2, 2)),    

#         # Slides filters across the height and width of an image to produce feature maps.
#         Conv2D(64, (3, 3)),  
#         BatchNormalization(),
#         Activation('relu'),

#         # Takes max value in each patch of each feature map to highlight the most important features
#         MaxPooling2D((2, 2)),   

#         Flatten(),                
#         Dense(4, activation='softmax') # seperate into 4 classes
#     ]
# )


model.summary()

# compile and train
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
history = model.fit(train_ds, epochs=1, validation_data=test_ds, callbacks=[tensorboard_callback])

# plot
# plt.title('Training Accuracy vs Validation Accuracy')

# print(history.history['accuracy'])
# print(history.history['val_accuracy'])

# plt.plot(history.history['accuracy'], color='red',label='Train')
# plt.plot(history.history['val_accuracy'], color='blue',label='Validation')

# plt.legend()
# plt.show()