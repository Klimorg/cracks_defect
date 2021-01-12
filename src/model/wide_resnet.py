from typing import List

import tensorflow as tf
import yaml
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    GlobalAvgPool2D,
    Input,
    ReLU,
)

params = yaml.safe_load(open("configs/params.yml"))["wide_resnet"]

repetitions = params["repetitions_block"]
n_classes = params["n_classes"]
width_factor = params["width_factor"]
img_shape = params["img_shape"]


def wide_bn_relu_conv(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: List[int],
    strides: List[int],
    width_factor: int,
) -> tf.Tensor:
    """[summary]

    Args:
        tensor (tf.Tensor): [description]
        filters (int): [description]
        kernel_size (List[int]): [description]
        strides (List[int]): [description]

    Returns:
        tf.Tensor: [description]
    """

    # shortcut strem
    shortcut = Conv2D(
        filters=filters * width_factor,
        kernel_size=(1, 1),
        padding="same",
        strides=strides,
        kernel_initializer="he_normal",
        use_bias=False,
    )(tensor)

    x = BatchNormalization()(tensor)
    x = ReLU()(x)
    x = Conv2D(
        filters=filters * width_factor,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
    )(x)

    x = Add()([x, shortcut])

    return x


def wide_resnet_block(
    tensor: tf.Tensor, filters: int, width_factor: int, strides: List[int]
) -> tf.Tensor:
    """[summary]

    Args:
        tensor (tf.Tensor): [description]
        filters (int): [description]

    Returns:
        tf.Tensor: [description]
    """
    # main stream
    x = wide_bn_relu_conv(
        tensor,
        filters,
        strides=strides,
        width_factor=width_factor,
        kernel_size=(3, 3),
    )
    x = wide_bn_relu_conv(
        x,
        filters,
        width_factor=width_factor,
        strides=strides,
        kernel_size=(3, 3),
    )

    return x


def wide_block(
    tensor: tf.Tensor,
    filters: int,
    width_factor: int,
    strides: List[int],
    repets: int,
) -> tf.Tensor:

    x = wide_resnet_block(tensor, filters, width_factor, strides)

    if repets > 1:
        for _ in range(repets - 1):
            x = wide_resnet_block(x, filters, width_factor, strides=(1, 1))

    return x


def get_wide_resnet(
    img_shape: List[int] = img_shape,
    n_classes: int = n_classes,
    width_factor: int = width_factor,
    repets: int = repetitions,
) -> tf.keras.Model:

    input = Input(img_shape)

    x = wide_bn_relu_conv(
        input, filters=16, kernel_size=(3, 3), strides=(1, 1), width_factor=1
    )

    x = wide_block(
        x,
        filters=16,
        width_factor=width_factor,
        repets=repets,
        strides=(1, 1),
    )

    x = wide_block(
        x,
        filters=32,
        width_factor=width_factor,
        repets=repets,
        strides=(2, 2),
    )
    x = wide_block(
        x,
        filters=64,
        width_factor=width_factor,
        repets=repets,
        strides=(2, 2),
    )

    x = GlobalAvgPool2D()(x)
    x = Dense(n_classes)(x)
    output = Activation("softmax")(x)

    model = Model(input, output)
    return model


if __name__ == "__main__":
    model = get_wide_resnet()
    model.summary()
