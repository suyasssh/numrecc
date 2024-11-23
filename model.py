import os
import cv2
import keras

is_google_drive: bool = False

if is_google_drive:
    # from google.colab import drive
    # drive.mount('/content/drive')

    # Save to Google Drive
    # model.save('/content/drive/My Drive/my_model.h5')
    pass

mnist = keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

# model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(728, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model=tf.keras.models.load_model('model1.h5')

model.fit(x_train, y_train, epochs=8, batch_size=32)

if is_google_drive:
    model.save('/content/drive/My Drive/model1.h5')
else:
    model.save("model.keras")
