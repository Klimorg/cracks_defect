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

params = yaml.safe_load(open("configs/params.yml"))["resnet"]

repetitions = params["repetitions"]
n_classes = params["n_classes"]
img_shape = params["img_shape"]


def bn_relu_conv(
    tensor: tf.Tensor, filters: int, kernel_size: List[int], strides: List[int]
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
    x = BatchNormalization()(tensor)
    x = ReLU()(x)
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
    )(x)

    return x


def resnet_block(tensor: tf.Tensor, filters: int) -> tf.Tensor:
    """[summary]

    Args:
        tensor (tf.Tensor): [description]
        filters (int): [description]

    Returns:
        tf.Tensor: [description]
    """

    inner_filters = filters // 4
    x = bn_relu_conv(tensor, inner_filters, kernel_size=(1, 1), strides=(1, 1))
    x = bn_relu_conv(x, inner_filters, kernel_size=(3, 3), strides=(1, 1))
    x = bn_relu_conv(x, filters, kernel_size=(1, 1), strides=(1, 1))

    x = Add()([x, tensor])

    return x


def proj_block(
    tensor: tf.Tensor, filters: int, strides: List[int]
) -> tf.Tensor:
    """[summary]

    Args:
        tensor (tf.Tensor): [description]
        filters (int): [description]
        strides (List[int]): [description]

    Returns:
        tf.Tensor: [description]
    """

    inner_filters = filters // 4

    x = BatchNormalization()(tensor)
    out1 = ReLU()(x)

    # shortcut
    out2 = Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        padding="same",
        strides=strides,
        kernel_initializer="he_normal",
        use_bias=False,
    )(out1)

    # main stream
    out3 = bn_relu_conv(
        out1, inner_filters, kernel_size=(1, 1), strides=strides
    )
    out3 = bn_relu_conv(
        out3, inner_filters, kernel_size=(3, 3), strides=(1, 1)
    )
    out3 = bn_relu_conv(out3, filters, kernel_size=(1, 1), strides=(1, 1))

    out = Add()([out2, out3])

    return out


def bottleneck_block(
    tensor: tf.Tensor, filters: int, repets: int, strides: List[int]
) -> tf.Tensor:
    """[summary]

    Args:
        tensor (tf.Tensor): [description]
        filters (int): [description]
        repets (int): [description]
        strides (List[int]): [description]

    Returns:
        tf.Tensor: [description]
    """

    x = proj_block(tensor, filters, strides)
    for _ in range(repets - 1):
        x = resnet_block(x, filters=filters)

    return x


def get_resnet(
    img_shape: List[int] = img_shape,
    n_classes: int = n_classes,
    repets: int = repetitions,
) -> tf.keras.Model:
    """[summary]

    Args:
        img_shape (List[int], optional): [description]. Defaults to img_shape.
        n_classes (int, optional): [description]. Defaults to n_classes.
        repets (int, optional): [description]. Defaults to repetitions.

    Returns:
        tf.keras.Model: [description]
    """

    input = Input(img_shape)

    x = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
    )(input)

    x = bottleneck_block(x, filters=64, repets=repets, strides=(1, 1))
    x = bottleneck_block(x, filters=128, repets=repets, strides=(2, 2))
    x = bottleneck_block(x, filters=256, repets=repets, strides=(2, 2))
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(n_classes)(x)
    output = Activation("softmax")(x)

    model = Model(input, output)
    return model


if __name__ == "__main__":

    model = get_resnet()
    model.summary()
    model.save("model.h5")
