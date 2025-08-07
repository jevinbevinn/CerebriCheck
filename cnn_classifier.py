import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Flatten,Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.utils import image_dataset_from_directory

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


train_path =  "/Users/kevinpham/radiology/scans/Training"
test_path =  "/Users/kevinpham/radiology/scans/Testing"

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

# build the model
model = Sequential(
    [
        Input(shape=(256, 256, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(4, activation='softmax')
    ]
)

# model = Sequential(
#     [
#         Input(shape=(256, 256, 3)),
#         Conv2D(32, (3, 3), activation='relu'),
#         # Conv2D(64, (3, 3), activation='relu'),
#         BatchNormalization(), 
#         MaxPooling2D((2, 2)),

#         # Conv2D(128, kernel_size=(3, 3), activation='relu'),
#         # BatchNormalization(), 
#         # MaxPooling2D((2, 2)),

#         # Conv2D(256, kernel_size=(3, 3), activation='relu'),
#         # BatchNormalization(), 
#         # MaxPooling2D((2, 2)),

#         Flatten(),
#         # Dense(256, activation='softmax'),
#         # Dense(128, activation='softmax'),
#         # Dense(64, activation='softmax'),
#         Dense(4, activation='softmax')
#     ]
# )

model.summary()

# compile and train
model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
history = model.fit(train_ds, epochs=10, validation_data=test_ds)

# plot
plt.title('Training Accuracy vs Validation Accuracy')

plt.plot(history.history['accuracy'], color='red',label='Train')
plt.plot(history.history['val_accuracy'], color='blue',label='Validation')

plt.legend()
plt.show()