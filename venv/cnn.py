import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


def set_up_model(input_shape):
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    # final dense layer must have 10 neurons because we have 10 number classes (0,1,2,...,10)
    model.add(Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def save_model(model):
    model_json = model.to_json()

    with open('model.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights('model.h5')
    print('model saved')


def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('model.h5')

    return loaded_model


def model_predict(model, image):
    model_pred = model.predict(image.reshape(1, 28, 28, 1))


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # reshaping array to 4 dimensions to work with Keras
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    input_shape = (28, 28, 1)

    # making sure that the values we are working with are floats so we can get decimal points
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # normalizing the RGB codes by dividing it to the max RGB value
    x_train /= 255
    x_test /= 255

    # model = set_up_model(input_shape)

    # model.fit(x=x_train, y=y_train, epochs=10)
    # model.evaluate(x_test, y_test)

    # save_model(model)

    loaded_model = load_model()

    img_idx = 4444
    model_predict(loaded_model, x_test[img_idx])


if __name__ == '__main__':
    main()
