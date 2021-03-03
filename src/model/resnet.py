from typing import List, Optional, Tuple

import tensorflow as tf
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

size_one = (1, 1)


def bn_relu_conv(
    tensor: tf.Tensor,
    filters: int,
    kernel_size: Tuple[int, int],
    strides: Tuple[int, int],
) -> tf.Tensor:
    """[summary].

    Args:
        tensor (tf.Tensor): [description]
        filters (int): [description]
        kernel_size (List[int]): [description]
        strides (List[int]): [description]

    Returns:
        tf.Tensor: [description]
    """
    img = BatchNormalization()(tensor)
    img = ReLU()(img)

    return Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
    )(img)


def resnet_block(tensor: tf.Tensor, filters: int) -> tf.Tensor:
    """[summary].

    Args:
        tensor (tf.Tensor): [description]
        filters (int): [description]

    Returns:
        tf.Tensor: [description]
    """
    inner_filters = filters // 4
    img = bn_relu_conv(tensor, inner_filters, kernel_size=size_one, strides=size_one)
    img = bn_relu_conv(img, inner_filters, kernel_size=(3, 3), strides=size_one)
    img = bn_relu_conv(img, filters, kernel_size=size_one, strides=size_one)

    return Add()([img, tensor])


def proj_block(tensor: tf.Tensor, filters: int, strides: Tuple[int, int]) -> tf.Tensor:
    """[summary].

    Args:
        tensor (tf.Tensor): [description]
        filters (int): [description]
        strides (List[int]): [description]

    Returns:
        tf.Tensor: [description]
    """
    inner_filters = filters // 4

    img = BatchNormalization()(tensor)
    out1 = ReLU()(img)

    # shortcut
    out2 = Conv2D(
        filters=filters,
        kernel_size=size_one,
        padding="same",
        strides=strides,
        kernel_initializer="he_normal",
        use_bias=False,
    )(out1)

    # main stream
    out3 = bn_relu_conv(out1, inner_filters, kernel_size=size_one, strides=strides)
    out3 = bn_relu_conv(out3, inner_filters, kernel_size=(3, 3), strides=size_one)
    out3 = bn_relu_conv(out3, filters, kernel_size=size_one, strides=size_one)

    return Add()([out2, out3])


def bottleneck_block(
    tensor: tf.Tensor,
    filters: int,
    strides: Tuple[int, int],
    repets: int,
) -> tf.Tensor:
    """[summary].

    Args:
        tensor (tf.Tensor): [description]
        filters (int): [description]
        repets (int): [description]
        strides (List[int]): [description]

    Returns:
        tf.Tensor: [description]
    """
    img = proj_block(tensor, filters, strides)
    for _ in range(repets - 1):
        img = resnet_block(img, filters=filters)

    return img


def get_cnn(
    img_shape: List[int],
    n_classes: Optional[int],
    repets: int,
) -> tf.keras.Model:
    """[summary].

    Args:
        img_shape (List[int], optional): [description]. Defaults to img_shape.
        n_classes (int, optional): [description]. Defaults to n_classes.
        repets (int): [description]. Defaults to repetitions.

    Returns:
        tf.keras.Model: [description]
    """
    img_input = Input(img_shape)

    img = Conv2D(
        filters=16,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
    )(img_input)

    img = bottleneck_block(img, filters=64, repets=repets, strides=size_one)
    img = bottleneck_block(img, filters=128, repets=repets, strides=(2, 2))
    img = bottleneck_block(img, filters=256, repets=repets, strides=(2, 2))
    img = BatchNormalization()(img)
    img = ReLU()(img)

    img = GlobalAvgPool2D()(img)
    img = Dense(n_classes)(img)
    output = Activation("softmax")(img)

    return Model(img_input, output)
