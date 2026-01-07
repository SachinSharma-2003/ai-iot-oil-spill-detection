import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    UpSampling2D, Concatenate
)
from tensorflow.keras.models import Model


def build_unet(input_shape=(256, 256, 1)):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
    c1 = Conv2D(64, 3, activation="relu", padding="same")(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(128, 3, activation="relu", padding="same")(p1)
    c2 = Conv2D(128, 3, activation="relu", padding="same")(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(256, 3, activation="relu", padding="same")(p2)
    c3 = Conv2D(256, 3, activation="relu", padding="same")(c3)
    p3 = MaxPooling2D()(c3)

    # Bottleneck
    c4 = Conv2D(512, 3, activation="relu", padding="same")(p3)
    c4 = Conv2D(512, 3, activation="relu", padding="same")(c4)

    # Decoder
    u5 = UpSampling2D()(c4)
    u5 = Concatenate()([u5, c3])
    c5 = Conv2D(256, 3, activation="relu", padding="same")(u5)
    c5 = Conv2D(256, 3, activation="relu", padding="same")(c5)

    u6 = UpSampling2D()(c5)
    u6 = Concatenate()([u6, c2])
    c6 = Conv2D(128, 3, activation="relu", padding="same")(u6)
    c6 = Conv2D(128, 3, activation="relu", padding="same")(c6)

    u7 = UpSampling2D()(c6)
    u7 = Concatenate()([u7, c1])
    c7 = Conv2D(64, 3, activation="relu", padding="same")(u7)
    c7 = Conv2D(64, 3, activation="relu", padding="same")(c7)

    # IMPORTANT: float32 output for stability
    outputs = Conv2D(
        1, 1, activation="sigmoid", dtype="float32"
    )(c7)

    return Model(inputs, outputs)
