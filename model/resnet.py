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


def bn_relu_conv(tensor, filters, kernel_size, strides):
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


def resnet_block(tensor, filters):

    inner_filters = filters // 4
    x = bn_relu_conv(tensor, inner_filters, kernel_size=(1, 1), strides=(1, 1))
    x = bn_relu_conv(x, inner_filters, kernel_size=(3, 3), strides=(1, 1))
    x = bn_relu_conv(x, filters, kernel_size=(1, 1), strides=(1, 1))

    x = Add()([x, tensor])

    return x


def proj_block(tensor, filters, strides):

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


def bottleneck_block(tensor, filters, repets, strides):

    x = proj_block(tensor, filters, strides)
    for _ in range(repets - 1):
        x = resnet_block(x, filters=filters)

    return x


def get_resnet(img_shape, n_classes, repets=5):

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
    x = Activation("softmax")(x)

    model = Model(input, x)
    return model


if __name__ == "__main__":

    img_shape = (256, 256, 3)
    model = get_resnet(img_shape=img_shape, n_classes=2)

    model.summary()
