from keras import Model
from keras.layers import BatchNormalization, Input, concatenate, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam

NUM_CHANNELS = 4
HEIGHT, WIDTH = 256, 256
LEARNING_RATE = 0.1


def create_model():
    x = Input(shape=(HEIGHT, WIDTH, NUM_CHANNELS))
    # first convolutional layer
    l1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same",
                input_shape=(HEIGHT, WIDTH, NUM_CHANNELS), activation="relu")(x)

    # second convolutional layer
    l2_b1a = BatchNormalization(momentum=0.01)(l1)
    l2_b1b = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l2_b1a)
    l2_b2a = BatchNormalization(momentum=0.01)(l2_b1b)
    l2_b2b = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l2_b2a)
    l2_b3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(l2_b2b)

    # third convolutional layer
    l3_b1a = BatchNormalization(momentum=0.01)(l2_b3)
    l3_b1b = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l3_b1a)
    l3_b2a = BatchNormalization(momentum=0.01)(l3_b1b)
    l3_b2b = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l3_b2a)
    l3_b3a = BatchNormalization(momentum=0.01)(l3_b2b)
    l3_b3b = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l3_b3a)
    l3_b3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(l3_b3b)

    # fourth convolutional layer
    l4_b1a = BatchNormalization(momentum=0.01)(l3_b3)
    l4_b1b = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l4_b1a)
    l4_b2a = BatchNormalization(momentum=0.01)(l4_b1b)
    l4_b2b = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l4_b2a)
    l4_b3a = BatchNormalization(momentum=0.01)(l4_b2b)
    l4_b3b = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l4_b3a)
    l4_b4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(l4_b3b)

    # fifth convolutional layer
    l5_b1a = BatchNormalization(momentum=0.01)(l4_b4)
    l5_b1b = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l5_b1a)
    l5_b2a = BatchNormalization(momentum=0.01)(l5_b1b)
    l5_b2b = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l5_b2a)
    l5_b3a = BatchNormalization(momentum=0.01)(l5_b2b)
    l5_b3b = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l5_b3a)
    l5_b4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(l5_b3b)

    # sixth convolutional layer
    l6_b1a = BatchNormalization(momentum=0.01)(l5_b4)
    l6_b1b = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l6_b1a)
    l6_b2a = BatchNormalization(momentum=0.01)(l6_b1b)
    l6_b2b = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l6_b2a)
    l6_b3a = BatchNormalization(momentum=0.01)(l6_b2b)
    l6_b3b = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l6_b3a)
    l6_b4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(l6_b3b)

    # seventh convolutional layer
    l7_b1a = BatchNormalization(momentum=0.01)(l6_b4)
    l7_b1b = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l7_b1a)
    l7_b2a = BatchNormalization(momentum=0.01)(l7_b1b)
    l7_b2b = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l7_b2a)
    l7_b3a = BatchNormalization(momentum=0.01)(l7_b2b)
    l7_b3b = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(l7_b3a)

    # eigth convolutional layer
    l8_b1 = concatenate([l6_b2b, l7_b3b])
    l8_b2a = BatchNormalization(momentum=0.01)(l8_b1)
    l8_b2b = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l8_b2a)
    l8_b3a = BatchNormalization(momentum=0.01)(l8_b2b)
    l8_b3b = Conv2D(96, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l8_b3a)
    l8_b4a = BatchNormalization(momentum=0.01)(l8_b3b)
    l8_b4b = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(l8_b4a)

    # nineth convolutional layer
    l9_b1 = concatenate([l5_b2b, l8_b4b])
    l9_b2a = BatchNormalization(momentum=0.01)(l9_b1)
    l9_b2b = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l9_b2a)
    l9_b3a = BatchNormalization(momentum=0.01)(l9_b2b)
    l9_b3b = Conv2D(96, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l9_b3a)
    l9_b4a = BatchNormalization(momentum=0.01)(l9_b3b)
    l9_b4b = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(l9_b4a)

    # tenth convolutional layer
    l10_b1 = concatenate([l4_b2b, l9_b4b])
    l10_b2a = BatchNormalization(momentum=0.01)(l10_b1)
    l10_b2b = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l10_b2a)
    l10_b3a = BatchNormalization(momentum=0.01)(l10_b2b)
    l10_b3b = Conv2D(96, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l10_b3a)
    l10_b4a = BatchNormalization(momentum=0.01)(l10_b3b)
    l10_b4b = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(l10_b4a)

    # eleventh convolutional layer
    l11_b1 = concatenate([l3_b2b, l10_b4b])
    l11_b2a = BatchNormalization(momentum=0.01)(l11_b1)
    l11_b2b = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l11_b2a)
    l11_b3a = BatchNormalization(momentum=0.01)(l11_b2b)
    l11_b3b = Conv2D(96, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l11_b3a)
    l11_b4a = BatchNormalization(momentum=0.01)(l11_b3b)
    l11_b4b = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(l11_b4a)

    # twelfth convolutional network
    l12_b1 = concatenate([l2_b1b, l11_b4b])
    l12_b2a = BatchNormalization(momentum=0.01)(l12_b1)
    l12_b2b = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l12_b2a)
    l12_b3a = BatchNormalization(momentum=0.01)(l12_b2b)
    l12_b3b = Conv2D(96, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(l12_b3a)
    l12_b4 = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding="valid", activation="sigmoid")(l12_b3b)
    y = Reshape((HEIGHT, WIDTH))(l12_b4)

    adam_optimizer = Adam(lr=LEARNING_RATE)
    model = Model(inputs=x, outputs=y)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['categorical_accuracy'])
    print(model.summary())
    return model